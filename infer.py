from pathlib import Path

import laya
from laya import Router

MODELS = Path(__file__).resolve().parent / "models"

# 使用本地已下载好的三个 checkpoint，避免从 Hugging Face 重新下载
router = Router(
    models={
        "english": str(MODELS / "laya"),
        "multilingual": str(MODELS / "laya-multilingual"),
        "typed-decisions": str(MODELS / "laya-typed-decisions"),
    },
    preload=True,
)

# 1. State in any language or schema
state = {
    "from": "user@acme.com",
    "subject": "Duplicate charge on invoice #4411",
    "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."
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
            "other": "everything else"
        }
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]
    },
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the user threaten to cancel or leave?"
    },
    "refund_requested": {
        "type": "noul",
        "instructions": "Does the user explicitly request a refund?"
    }
}

# 3. English state -> automatically routed to laya (ModernBERT-large, 39.5 ms)
res_en = router.predict(state, questions)
print("Department :", res_en["answers"]["department"]["choice"])  # -> billing (confidence: 0.94)
print("Routing    :", res_en["routing"]["model"])                 # -> english

# 4. Hindi state -> automatically routed to laya-multilingual (mmBERT-base, 32.8 ms)
res_hi = router.predict({"body": "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।"}, questions)
print("Department :", res_hi["answers"]["department"]["choice"])  # -> billing (confidence: 0.86)
print("Routing    :", res_hi["routing"]["model"])                 # -> multilingual

# 5. Explicit override when you want a specific checkpoint
res_td = router.predict(state, questions, model="typed-decisions")