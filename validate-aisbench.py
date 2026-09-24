#!/usr/bin/env python3
"""Compare all three OM bundles against CPU checkpoint answers on the same inputs."""
import argparse
import gc
import json
import os
from pathlib import Path

import torch

from bench_cpu_vs_npu import CASES, QUESTIONS
from laya import Agent, AisBenchAgent
from laya.aisbench import MODEL_DIRS

ROOT = Path(__file__).resolve().parent


def compare(reference, actual):
    """Return the largest numeric error and whether the categorical structure agrees."""
    if isinstance(reference, dict):
        if not isinstance(actual, dict) or set(reference) != set(actual):
            return float("inf"), False
        children = [compare(reference[key], actual[key]) for key in reference]
        return max((error for error, _ in children), default=0.0), all(ok for _, ok in children)
    if isinstance(reference, (int, float)):
        return abs(reference - actual), True
    return 0.0, reference == actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=ROOT / "models")
    parser.add_argument("--om-root", type=Path, default=ROOT / "models" / "aisbench")
    parser.add_argument("--models", nargs="+", choices=list(MODEL_DIRS), default=list(MODEL_DIRS))
    parser.add_argument("--device-id", type=int, default=int(os.environ.get("LAYA_AISBENCH_DEVICE", "0")))
    parser.add_argument("--atol", type=float, default=0.02, help="maximum answer/probability/confidence error")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0 < args.atol < 1:
        parser.error("--atol must be between 0 and 1")
    torch.set_num_threads(4)
    report = {"device_id": args.device_id, "atol": args.atol, "models": {}}
    passed = True
    for name in args.models:
        reference = Agent(str(args.model_root / MODEL_DIRS[name]), device="cpu")
        cases = {
            "mixed": (CASES[name]["state"], QUESTIONS),
            "single-option": ("hello", {"only": {"type": "choice", "instructions": "Choose", "criteria": ["only"]}}),
            "multibatch": (CASES[name]["state"], {"q%d" % i: QUESTIONS["churn_risk"] for i in range(5)}),
        }
        results = {}
        with AisBenchAgent(args.om_root / name, device=args.device_id) as actual:
            for label, (state, questions) in cases.items():
                expected = reference.predict(state, questions)
                observed = actual.predict(state, questions)
                error, same_choices = compare(expected["answers"], observed["answers"])
                same_tokens = expected["usage"] == observed["usage"]
                ok = same_choices and same_tokens and error <= args.atol
                results[label] = {"max_abs_error": round(error, 6), "same_choices": same_choices,
                                  "same_usage": same_tokens, "passed": ok}
                passed = passed and ok
        report["models"][name] = results
        print("[validate] %s: %s" % (name, json.dumps(results)), flush=True)
        del reference
        gc.collect()
    report["passed"] = passed
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not passed:
        raise SystemExit("OM/CPU comparison failed; see the report (do not assume FP16 parity)")


if __name__ == "__main__":
    main()
