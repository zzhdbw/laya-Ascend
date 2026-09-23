#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Laya NPU inference demo.

Run with environment:
    ASCEND_RT_VISIBLE_DEVICES=0 python infer-npu.py

By default three local checkpoints under ./models are loaded onto Ascend NPU.
Override the device with LAYA_NPU_DEVICE, e.g. LAYA_NPU_DEVICE=npu:1.
Set LAYA_NPU_PRELOAD=0 to load each checkpoint lazily on first use.
"""

import os
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    import torch_npu  # noqa: F401  # registers the Ascend NPU backend in torch
except ImportError as exc:  # pragma: no cover - depends on the runtime environment
    raise SystemExit(
        "未检测到 torch_npu。请先安装与 torch 匹配的 torch-npu 包，"
        "并确认 CANN/Ascend 驱动已正确配置。"
    ) from exc

if not torch.npu.is_available():
    raise SystemExit(
        "当前环境没有可用的 Ascend NPU。请运行 `npu-smi info` 检查设备，"
        "并确认 ASCEND_RT_VISIBLE_DEVICES 设置正确。"
    )

# 默认使用第 0 张卡，可通过环境变量切换
NPU_DEVICE = os.environ.get("LAYA_NPU_DEVICE", "npu:0")
torch.npu.set_device(NPU_DEVICE)

# PyTorch 的 aten::_transformer_encoder_layer_fwd 在当前 torch_npu 上没有 NPU kernel，
# 会整层回退到 CPU。这里用 NPU 支持的 SDPA 路径实现等价的 TransformerEncoderLayer
# forward，确保 Laya 的决策头也真正运行在 NPU 上；数值结果与原实现一致。
def _npu_mha_forward(attn, query, key_padding_mask=None, attn_mask=None, is_causal=False):
    bsz, tgt_len, embed_dim = query.shape
    src_len = query.shape[1]
    head_dim = embed_dim // attn.num_heads

    if not attn.batch_first or attn.in_proj_weight is None:
        raise RuntimeError("infer-npu.py expects a batch-first TransformerEncoderLayer")

    qkv = F.linear(query, attn.in_proj_weight, attn.in_proj_bias)
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.reshape(bsz, tgt_len, attn.num_heads, head_dim).transpose(1, 2)
    k = k.reshape(bsz, src_len, attn.num_heads, head_dim).transpose(1, 2)
    v = v.reshape(bsz, src_len, attn.num_heads, head_dim).transpose(1, 2)

    attn_bias = None
    if key_padding_mask is not None:
        attn_bias = torch.zeros(bsz, 1, 1, src_len, dtype=q.dtype, device=q.device)
        attn_bias = attn_bias.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            mask = torch.zeros(bsz, 1, tgt_len, src_len, dtype=q.dtype, device=q.device)
            mask = mask.masked_fill(attn_mask, float("-inf"))
        else:
            mask = attn_mask
        attn_bias = mask if attn_bias is None else attn_bias + mask

    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_bias, dropout_p=0.0, is_causal=is_causal
    )
    out = out.transpose(1, 2).reshape(bsz, tgt_len, embed_dim)
    return attn.out_proj(out)


def _npu_sa_block(layer, x, attn_mask, key_padding_mask, is_causal=False):
    x = _npu_mha_forward(
        layer.self_attn,
        x,
        key_padding_mask=key_padding_mask,
        attn_mask=attn_mask,
        is_causal=is_causal,
    )
    return layer.dropout1(x)


def _npu_transformer_encoder_layer_forward(
    self, src, src_mask=None, src_key_padding_mask=None, is_causal=False
):
    x = src
    if self.norm_first:
        x = x + _npu_sa_block(self, self.norm1(x), src_mask, src_key_padding_mask, is_causal)
        x = x + self._ff_block(self.norm2(x))
    else:
        x = self.norm1(x + _npu_sa_block(self, x, src_mask, src_key_padding_mask, is_causal))
        x = self.norm2(x + self._ff_block(x))
    return x


torch.nn.TransformerEncoderLayer.forward = _npu_transformer_encoder_layer_forward

from laya import Router  # noqa: E402

MODELS = Path(__file__).resolve().parent / "models"
PRELOAD = os.environ.get("LAYA_NPU_PRELOAD", "1").lower() not in ("0", "false", "no")

print("NPU device   :", NPU_DEVICE)
print("Devices      :", torch.npu.device_count(), "x", torch.npu.get_device_name(torch.npu.current_device()))
print("Preload      :", PRELOAD)

# 使用本地已下载好的三个 checkpoint，避免从 Hugging Face 重新下载
router = Router(
    models={
        "english": str(MODELS / "laya"),
        "multilingual": str(MODELS / "laya-multilingual"),
        "typed-decisions": str(MODELS / "laya-typed-decisions"),
    },
    device=NPU_DEVICE,
    max_loaded=3,
    preload=PRELOAD,
)

# 1. State in any language or schema
state = {
    "from": "user@acme.com",
    "subject": "Duplicate charge on invoice #4411",
    "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.",
}

# 2. Define your typed questions
questions = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "sales": "pricing, new contracts",
            "other": "everything else",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"],
    },
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the user threaten to cancel or leave?",
    },
    "refund_requested": {
        "type": "noul",
        "instructions": "Does the user explicitly request a refund?",
    },
}

# 3. English state -> automatically routed to laya
res_en = router.predict(state, questions)
print("\n[English]")
print("  Department :", res_en["answers"]["department"]["choice"])
print("  Routing    :", res_en["routing"]["model"])
print("  Device     :", NPU_DEVICE)

# 4. Hindi state -> automatically routed to laya-multilingual
res_hi = router.predict(
    {"body": "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।"}, questions
)
print("\n[Hindi]")
print("  Department :", res_hi["answers"]["department"]["choice"])
print("  Routing    :", res_hi["routing"]["model"])
print("  Device     :", NPU_DEVICE)

# 5. Explicit override when you want a specific checkpoint
res_td = router.predict(state, questions, model="typed-decisions")
print("\n[Explicit typed-decisions]")
print("  Department :", res_td["answers"]["department"]["choice"])
print("  Routing    :", res_td["routing"]["model"])
print("  Device     :", NPU_DEVICE)
