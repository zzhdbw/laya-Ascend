#!/usr/bin/env python3
"""Run English, multilingual and typed decisions with AISBench (no torch_npu)."""
import argparse
import json
import os
from pathlib import Path

from laya import Router
from laya.aisbench import MODEL_DIRS

QUESTIONS = {
    "department": {
        "type": "choice", "instructions": "Which department should handle this request?",
        "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages", "sales": "new contracts"},
    },
    "urgency": {"type": "score", "instructions": "How urgent is this request?",
                "criteria": ["not urgent", "soon", "critical deadline"]},
    "refund": {"type": "noul", "instructions": "Does the user request a refund?"},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--om-root", type=Path, default=Path(__file__).resolve().parent / "models" / "aisbench")
    parser.add_argument("--device-id", type=int, default=int(os.environ.get("LAYA_AISBENCH_DEVICE", "0")))
    args = parser.parse_args()
    router = Router(models={name: str(args.om_root / name) for name in MODEL_DIRS},
                    backend="aisbench", device="npu:%d" % args.device_id, max_loaded=1)
    try:
        for state, model in (("I was charged twice. Please refund the duplicate today.", None),
                             ("मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।", None),
                             ("I was charged twice. Please refund the duplicate today.", "typed-decisions")):
            result = router.predict(state, QUESTIONS, model=model)
            print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        router.unload()


if __name__ == "__main__":
    main()
