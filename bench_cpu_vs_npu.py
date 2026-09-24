#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fair CPU vs Ascend NPU inference benchmark for Laya.

Method:
  1. Load the same local checkpoint on the target device.
  2. Run `warmup` untimed predictions to populate caches / compile graphs.
  3. Synchronize the device, then time `runs` predictions with perf_counter.
  4. Report wall-clock latency statistics and token throughput.

Examples:
  .venv/bin/python bench_cpu_vs_npu.py --device cpu --cpu-threads 1 --output benchmarks/cpu.json
  .venv/bin/python bench_cpu_vs_npu.py --device npu --output benchmarks/npu.json
"""

import argparse
import gc
import json
import math
import os
import platform
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
MODEL_PATHS = {
    "english": ROOT / "models" / "laya",
    "multilingual": ROOT / "models" / "laya-multilingual",
    "typed-decisions": ROOT / "models" / "laya-typed-decisions",
}

ENGLISH_STATE = {
    "from": "user@acme.com",
    "subject": "Duplicate charge on invoice #4411",
    "body": (
        "Hi, we were billed twice for March. Please refund the duplicate today "
        "or we will cancel our plan."
    ),
}
HINDI_STATE = {
    "body": "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।",
}
QUESTIONS = {
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
CASES = {
    "english": {"state": ENGLISH_STATE, "label": "English state / 4 questions"},
    "multilingual": {"state": HINDI_STATE, "label": "Hindi state / 4 questions"},
    "typed-decisions": {"state": ENGLISH_STATE, "label": "English state / 4 questions"},
}


def _npu_mha_forward(attn, query, key_padding_mask=None, attn_mask=None, is_causal=False):
    """Equivalent MultiheadAttention for a batch-first TransformerEncoderLayer."""
    bsz, tgt_len, embed_dim = query.shape
    src_len = query.shape[1]
    head_dim = embed_dim // attn.num_heads
    if not attn.batch_first or attn.in_proj_weight is None:
        raise RuntimeError("benchmark expects a batch-first TransformerEncoderLayer")

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


def setup_npu(device):
    """Import torch_npu, select the device, and install the NPU decision-head path."""
    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise SystemExit("torch_npu is not installed in this Python environment") from exc
    if not torch.npu.is_available():
        raise SystemExit("No Ascend NPU is available; check `npu-smi info`")
    torch.npu.set_device(device)
    torch.nn.TransformerEncoderLayer.forward = _npu_transformer_encoder_layer_forward
    return torch.device(device)


def sync_device(device):
    if device.type == "npu":
        torch.npu.synchronize(device)


def percentile(values, p):
    """Linear-interpolated percentile, p in [0, 1]."""
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * p
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return values[int(rank)]
    return values[lo] * (hi - rank) + values[hi] * (rank - lo)


def summarize(times_ms, tokens):
    mean = statistics.mean(times_ms)
    std = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
    median = statistics.median(times_ms)
    return {
        "count": len(times_ms),
        "latency_ms": {
            "mean": round(mean, 3),
            "median": round(median, 3),
            "std": round(std, 3),
            "min": round(min(times_ms), 3),
            "p90": round(percentile(times_ms, 0.90), 3),
            "p95": round(percentile(times_ms, 0.95), 3),
            "max": round(max(times_ms), 3),
            "cv_percent": round(100.0 * std / mean, 2) if mean else None,
        },
        "input_tokens": tokens,
        "tokens_per_second": round(tokens / (median / 1000.0), 2) if median else None,
    }


def benchmark_model(name, device, device_label, warmup, runs, om_root=None, aisbench_device=0):
    from laya import Agent, AisBenchAgent

    case = CASES[name]
    path = str(Path(om_root) / name) if om_root is not None else str(MODEL_PATHS[name])
    state, label = case["state"], case["label"]

    print("[bench] loading %s on %s ..." % (name, device_label), file=sys.stderr, flush=True)
    t0 = time.perf_counter()
    agent = (AisBenchAgent(path, device=aisbench_device) if om_root is not None
             else Agent(path, device=str(device)))
    load_s = time.perf_counter() - t0
    print("[bench] loaded %s in %.1fs" % (name, load_s), file=sys.stderr, flush=True)

    with torch.no_grad():
        for _ in range(warmup):
            result = agent.predict(state, QUESTIONS)
    sync_device(device)

    times_ms = []
    with torch.no_grad():
        for _ in range(runs):
            sync_device(device)
            start = time.perf_counter()
            result = agent.predict(state, QUESTIONS)
            sync_device(device)
            times_ms.append((time.perf_counter() - start) * 1000.0)

    tokens = int(result.get("usage", {}).get("input_tokens", 0))
    summary = summarize(times_ms, tokens)
    summary["model"] = name
    summary["model_path"] = path
    summary["case"] = label
    summary["load_seconds"] = round(load_s, 3)
    summary["device_label"] = device_label
    summary["runtime_device"] = str(agent.device)
    if om_root is not None:
        summary["om_shape"] = {"batch_size": agent.batch_size, "seq_len": agent.seq_len,
                               "max_options": agent.max_options}
    summary["raw_latency_ms"] = [round(x, 3) for x in times_ms]
    print(
        "[bench] %s on %s: median %.1f ms, mean %.1f ms, p95 %.1f ms"
        % (name, agent.device, summary["latency_ms"]["median"], summary["latency_ms"]["mean"], summary["latency_ms"]["p95"]),
        file=sys.stderr,
        flush=True,
    )
    if om_root is not None:
        agent.close()
    del agent
    gc.collect()
    if device.type == "npu":
        torch.npu.empty_cache()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "npu", "aisbench", "both"], default="both")
    parser.add_argument("--models", nargs="+", choices=list(MODEL_PATHS), default=list(MODEL_PATHS))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--cpu-threads", type=int, default=1,
                        help="torch CPU threads for the CPU pass (default: 1; this VM is fastest at 1)")
    parser.add_argument("--npu-device", default=os.environ.get("LAYA_NPU_DEVICE", "npu:0"))
    parser.add_argument("--om-root", type=Path, default=ROOT / "models" / "aisbench")
    parser.add_argument("--aisbench-device", type=int, default=int(os.environ.get("LAYA_AISBENCH_DEVICE", "0")))
    parser.add_argument("--output", default=None, help="write full JSON report to this path")
    args = parser.parse_args()

    if args.warmup < 1 or args.runs < 1:
        raise SystemExit("--warmup and --runs must be positive")

    report = {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "hostname": platform.node(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cpu_count_logical": os.cpu_count(),
            "cpu_threads": args.cpu_threads,
            "warmup": args.warmup,
            "runs": args.runs,
            "models": args.models,
            "device_requested": args.device,
        },
        "results": {},
    }

    # CPU pass first, so the NPU decision-head monkeypatch cannot affect CPU numbers.
    if args.device in ("cpu", "both"):
        torch.set_num_threads(max(1, args.cpu_threads))
        report["meta"]["cpu_threads_effective"] = torch.get_num_threads()
        for name in args.models:
            key = "cpu/" + name
            report["results"][key] = benchmark_model(
                name, torch.device("cpu"), "CPU", args.warmup, args.runs
            )

    if args.device in ("npu", "both"):
        npu_device = setup_npu(args.npu_device)
        report["meta"]["npu_device"] = args.npu_device
        report["meta"]["npu_name"] = torch.npu.get_device_name(npu_device)
        report["meta"]["npu_count"] = torch.npu.device_count()
        for name in args.models:
            key = "npu/" + name
            report["results"][key] = benchmark_model(
                name, npu_device, "NPU (%s)" % report["meta"]["npu_name"], args.warmup, args.runs
            )

    if args.device == "aisbench":
        report["meta"]["aisbench_device"] = args.aisbench_device
        report["meta"]["om_root"] = str(args.om_root)
        for name in args.models:
            # InferSession.infer returns host arrays: execution and D2H are synchronous.
            report["results"]["aisbench/" + name] = benchmark_model(
                name, torch.device("cpu"), "AISBench device %d" % args.aisbench_device,
                args.warmup, args.runs, om_root=args.om_root, aisbench_device=args.aisbench_device,
            )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[bench] wrote %s" % out, file=sys.stderr, flush=True)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
