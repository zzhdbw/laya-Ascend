"""NPU-friendly equivalent of torch.nn.TransformerEncoderLayer.forward.

Current torch_npu has no kernel for aten::_transformer_encoder_layer_fwd, so
the Laya decision head would otherwise fall back to CPU.  The implementation
below uses F.scaled_dot_product_attention, which is supported on Ascend NPU,
and keeps the original weights and output semantics.
"""

import torch
import torch.nn.functional as F


def _npu_mha_forward(attn, query, key_padding_mask=None, attn_mask=None, is_causal=False):
    bsz, tgt_len, embed_dim = query.shape
    src_len = query.shape[1]
    head_dim = embed_dim // attn.num_heads

    if not attn.batch_first or attn.in_proj_weight is None:
        raise RuntimeError("NPU patch expects a batch-first TransformerEncoderLayer")

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


def enable_npu_patch():
    """Install the NPU decision-head path. Safe to call more than once."""
    torch.nn.TransformerEncoderLayer.forward = _npu_transformer_encoder_layer_forward
    return True
