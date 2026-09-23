"""Laya rates each shortlisted Tetris placement independently with one noul question.

Adapted from the Laya-AXERA / laya-mlx Tetris demo.  The heuristic shortlist and
guarded execution rule are kept the same; inference now runs through the
upstream `laya` package (CPU or Ascend NPU).
"""

import time
from dataclasses import asdict, dataclass
from typing import List

from game import DANGER_HEIGHT

CN_NUM = ["零", "一", "两", "三", "四", "五", "六", "七", "八", "九", "十"]
QUESTION = "这是一个好的落点吗？"


def cn(n):
    return CN_NUM[n] if 0 <= n < len(CN_NUM) else str(n)


def describe(cand):
    lines = f"消除{cn(cand['lines'])}行" if cand["lines"] else "不消行"
    holes = f"埋下{cn(cand['holes'])}个空洞" if cand["holes"] else "不留空洞"
    if cand["height"] > DANGER_HEIGHT:
        height = "堆叠接近板顶"
    elif cand["height"] > 10:
        height = "堆叠变高"
    else:
        height = "堆叠保持低位"
    return f"这个落点{lines}，{holes}，{height}。"


@dataclass
class TetrisDecision:
    candidates: List[dict]
    proposed: int
    executed: int
    intervened: bool
    heuristic_best: int
    inference_ms: float
    decision_ms: float
    input_tokens: int

    def to_dict(self):
        return asdict(self)


class LayaTetrisPolicy:
    def __init__(self, agent, *, guarded=True):
        self.agent = agent
        self.guarded = bool(guarded)

    def rate(self, cands):
        """Rate placements independently; returns (rated, inference_ms, input_tokens)."""
        inference_ms = 0.0
        input_tokens = 0
        rated = []
        for cand in cands:
            statement = describe(cand)
            started = time.perf_counter()
            output = self.agent.predict(
                statement,
                {"q": {"type": "noul", "instructions": QUESTION}},
            )
            inference_ms += (time.perf_counter() - started) * 1000.0
            answer = output["answers"]["q"]
            input_tokens += output["usage"]["input_tokens"]
            rated.append({**cand, "statement": statement, "p_good": answer["noul"]})
        return rated, inference_ms, input_tokens

    def decide(self, game):
        started = time.perf_counter()
        candidates = game.candidates()
        if not candidates:
            raise RuntimeError("No legal placement for the current piece")
        rated, inference_ms, input_tokens = self.rate(candidates)
        proposed = max(range(len(rated)), key=lambda i: rated[i]["p_good"])
        executed = proposed
        if self.guarded and rated[proposed]["height"] > DANGER_HEIGHT:
            safe = [i for i in range(len(rated)) if rated[i]["height"] <= DANGER_HEIGHT]
            if safe:
                executed = max(safe, key=lambda i: rated[i]["p_good"])
        return TetrisDecision(
            candidates=rated,
            proposed=proposed,
            executed=executed,
            intervened=proposed != executed,
            heuristic_best=0,  # candidates arrive heuristic-sorted
            inference_ms=inference_ms,
            decision_ms=(time.perf_counter() - started) * 1000.0,
            input_tokens=input_tokens,
        )
