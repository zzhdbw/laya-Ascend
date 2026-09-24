#!/usr/bin/env python3
"""Export all Laya checkpoints to static ONNX and optionally compile Ascend OM."""
import argparse
from pathlib import Path

from laya.aisbench import MODEL_DIRS
from aisbench_export import compile_bundle, export_bundle

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=list(MODEL_DIRS), default=list(MODEL_DIRS))
    parser.add_argument("--model-root", type=Path, default=ROOT / "models")
    parser.add_argument("--output-root", type=Path, default=ROOT / "models" / "aisbench")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=None, help="default: checkpoint max_len; overflow is rejected at runtime")
    parser.add_argument("--max-options", type=int, default=32)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-only", action="store_true", help="compile previously exported bundles")
    parser.add_argument("--soc-version", help="actual target SoC, e.g. Ascend310P3 or Ascend910B1")
    parser.add_argument("--precision-mode", default="must_keep_origin_dtype")
    args = parser.parse_args()
    if (args.compile or args.compile_only) and not args.soc_version:
        parser.error("--soc-version is required for compilation")
    for name in args.models:
        output = args.output_root / name
        if not args.compile_only:
            print("[export] %s -> %s" % (name, output), flush=True)
            export_bundle(args.model_root / MODEL_DIRS[name], output,
                          batch_size=args.batch_size, seq_len=args.seq_len, max_options=args.max_options)
        if args.compile or args.compile_only:
            print("[compile] %s (%s)" % (name, args.soc_version), flush=True)
            compile_bundle(output, args.soc_version, precision_mode=args.precision_mode)
        print("[done] %s" % output, flush=True)


if __name__ == "__main__":
    main()
