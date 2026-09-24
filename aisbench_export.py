"""Static ONNX export and ATC compilation for the three ModernBERT checkpoints."""
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from laya.agent import Agent
from laya.aisbench import INPUT_NAMES


class _ModernBertForExport(nn.Module):
    """Build explicit masks instead of tracing transformers' vmap mask machinery."""

    def __init__(self, encoder, seq_len):
        super().__init__()
        self.encoder = encoder
        self.config = encoder.config
        positions = torch.arange(seq_len)
        local = (positions[:, None] - positions[None, :]).abs() <= encoder.config.sliding_window
        self.register_buffer("local_mask", local[None, None], persistent=False)

    def forward(self, input_ids, attention_mask):
        allowed = attention_mask[:, None, None, :].bool()
        full = torch.zeros_like(allowed, dtype=torch.float32).masked_fill(~allowed, torch.finfo(torch.float32).min)
        local = full.expand(-1, 1, input_ids.shape[1], -1).masked_fill(
            ~self.local_mask, torch.finfo(torch.float32).min
        )
        return self.encoder(input_ids=input_ids, attention_mask={
            "full_attention": full,
            "sliding_attention": local,
        })


class _HeadLayerForExport(nn.Module):
    """Equivalent eval-only head attention using portable MatMul/Softmax operators."""

    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def _attention(self, x, padding):
        attn = self.layer.self_attn
        batch, length, hidden = x.shape
        dim = hidden // attn.num_heads
        q, k, v = F.linear(x, attn.in_proj_weight, attn.in_proj_bias).chunk(3, dim=-1)
        q, k, v = (t.reshape(batch, length, attn.num_heads, dim).transpose(1, 2) for t in (q, k, v))
        scores = torch.matmul(q, k.transpose(-2, -1)) * (dim ** -0.5)
        scores = scores.masked_fill(padding[:, None, None, :], float("-inf"))
        out = torch.matmul(torch.softmax(scores, -1), v).transpose(1, 2).reshape(batch, length, hidden)
        return attn.out_proj(out)

    def forward(self, src, src_key_padding_mask):
        layer = self.layer
        if layer.norm_first:
            x = src + self._attention(layer.norm1(src), src_key_padding_mask)
            return x + layer._ff_block(layer.norm2(x))
        x = layer.norm1(src + self._attention(src, src_key_padding_mask))
        return layer.norm2(x + layer._ff_block(x))


@contextmanager
def exportable_model(model, seq_len):
    """Patch this model instance only, restoring it even after a failed export."""
    encoder = model.encoder
    if encoder.config.model_type != "modernbert" or not hasattr(encoder, "rotary_emb"):
        raise ValueError("AISBench export requires ModernBERT with transformers 5.17 (see README-AISBench.md)")
    attention = encoder.config._attn_implementation
    layers = model.head.layers if model.head is not None else None
    try:
        encoder.config._attn_implementation = "eager"
        model.encoder = _ModernBertForExport(encoder, seq_len)
        if layers is not None:
            model.head.layers = nn.ModuleList([_HeadLayerForExport(layer) for layer in layers])
        yield model
    finally:
        model.encoder = encoder
        encoder.config._attn_implementation = attention
        if layers is not None:
            model.head.layers = layers


def export_bundle(model_path, output_dir, *, batch_size=1, seq_len=None, max_options=32):
    """Export a full decision model, tokenizer and runtime metadata, without CANN.

    ONNXRuntime verifies both outputs before publishing the bundle metadata. ATC
    compilation is a separate step that must run in the target CANN environment.
    """
    import onnx
    import onnxruntime as ort

    for name, value in (("batch_size", batch_size), ("max_options", max_options)):
        if type(value) is not int or value < (2 if name == "max_options" else 1):
            raise ValueError("Invalid %s" % name)
    if seq_len is not None and (type(seq_len) is not int or seq_len < 8):
        raise ValueError("seq_len must be an integer >= 8")
    output = Path(output_dir)
    # Avoid a failed re-export leaving new metadata paired with an old OM.
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Export directory must be empty: %s" % output)
    agent = Agent(str(model_path), device="cpu")
    seq_len = seq_len or agent.cfg.get("max_len", 512)
    if seq_len > agent.cfg.get("max_len", 512):
        raise ValueError("seq_len cannot exceed the checkpoint's max_len")
    output.mkdir(parents=True, exist_ok=True)
    shapes = (batch_size, seq_len)
    ids = torch.full(shapes, agent.tok.pad_token_id, dtype=torch.long)
    used = min(seq_len, 8)
    ids[:, :used] = agent.tok.unk_token_id if agent.tok.unk_token_id is not None else agent.tok.cls_token_id
    ids[:, 0], ids[:, used - 1] = agent.tok.cls_token_id, agent.tok.sep_token_id
    mask = torch.zeros(shapes, dtype=torch.long)
    mask[:, :used] = 1
    markers = torch.zeros((batch_size, max_options), dtype=torch.long)
    markers[:, :2] = torch.tensor([1, 2])
    marker_mask = torch.zeros_like(markers, dtype=torch.bool)
    marker_mask[:, :2] = True
    inputs = (ids, mask, markers, marker_mask, torch.arange(batch_size) % 3)
    onnx_path = output / "model.onnx"
    with torch.no_grad():
        expected = agent.model(*inputs)
        with exportable_model(agent.model, seq_len) as model:
            actual = model(*inputs)
            for ref, out in zip(expected, actual):
                torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
            torch.onnx.export(
                model, inputs, str(onnx_path), input_names=list(INPUT_NAMES),
                output_names=["logits", "act_logits"], opset_version=17,
                dynamo=False, do_constant_folding=True,
            )
    onnx.checker.check_model(str(onnx_path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"])
    outputs = session.run(None, {name: value.numpy() for name, value in zip(INPUT_NAMES, inputs)})
    for reference, actual in zip(expected, outputs):
        np.testing.assert_allclose(actual, reference.numpy(), rtol=1e-4, atol=1e-4)

    agent.tok.save_pretrained(output / "tokenizer")
    metadata = {
        "format_version": 1,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "max_options": max_options,
        "n_act": len(agent.cfg.get("act_costs", {})) + 1,
        "agent_config": agent.cfg,
        "source": str(model_path),
        "onnx_opset": 17,
    }
    (output / "aisbench_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output


def compile_bundle(output_dir, soc_version, *, precision_mode="must_keep_origin_dtype", atc="atc"):
    """Compile static ONNX with checked ATC exit status; never infer the target SoC."""
    output = Path(output_dir).resolve()
    if not soc_version:
        raise ValueError("Specify the actual --soc-version reported by your Ascend environment")
    if not (output / "model.onnx").is_file() or not (output / "aisbench_config.json").is_file():
        raise FileNotFoundError("Export the ONNX bundle before compiling")
    if (output / "model.om").exists():
        raise FileExistsError("model.om already exists; use a fresh export directory")
    executable = shutil.which(atc)
    if executable is None:
        raise FileNotFoundError("ATC not found; source your CANN toolkit set_env.sh first")
    command = [
        executable, "--framework=5", "--model=%s" % (output / "model.onnx"),
        "--output=%s" % (output / "model"), "--input_format=ND",
        "--soc_version=%s" % soc_version, "--precision_mode=%s" % precision_mode,
    ]
    subprocess.run(command, check=True)
    if not (output / "model.om").is_file():
        raise RuntimeError("ATC exited successfully but did not produce model.om")
    (output / "atc_command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
    return output / "model.om"
