"""Shared model/device loading helpers for the Snake examples."""

import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "models" / "laya-multilingual"


def choose_device(requested="auto"):
    """Resolve auto -> NPU if available, else CPU. Returns a torch device string."""
    requested = (requested or "auto").strip()
    if requested.lower() == "auto":
        try:
            import torch_npu  # noqa: F401

            requested = "npu:0" if torch.npu.is_available() else "cpu"
        except Exception:
            requested = "cpu"

    if requested.lower().startswith("npu"):
        try:
            import torch_npu  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "NPU requested but torch_npu is not installed in this Python environment"
            ) from exc
        if not torch.npu.is_available():
            raise RuntimeError("NPU requested but torch.npu.is_available() is False")
        torch.npu.set_device(requested)
        from npu_patch import enable_npu_patch

        enable_npu_patch()
        print("[snake] device: %s" % requested, file=sys.stderr, flush=True)
    else:
        print("[snake] device: cpu", file=sys.stderr, flush=True)
    return requested


def load_agent(model=None, device="auto"):
    """Load one upstream Laya Agent for the Snake demo."""
    from laya import Agent

    model_path = Path(model or os.environ.get("LAYA_SNAKE_MODEL") or DEFAULT_MODEL).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(
            "Model directory not found: %s\n"
            "Download the checkpoints first, or pass --model /path/to/checkpoint." % model_path
        )
    device = choose_device(device)
    print("[snake] loading %s ..." % model_path, file=sys.stderr, flush=True)
    agent = Agent(str(model_path), device=device)
    print("[snake] loaded on %s" % agent.device, file=sys.stderr, flush=True)

    # Warm up the first forward pass so the first browser move is not slowed down
    # by one-time NPU compilation / cache setup.
    if os.environ.get("LAYA_SNAKE_WARMUP", "1").lower() not in ("0", "false", "no"):
        agent.predict(
            "warmup",
            {
                "warmup": {
                    "type": "noul",
                    "instructions": "Is this request a warmup?",
                }
            },
        )
        print("[snake] warmup done", file=sys.stderr, flush=True)
    return agent
