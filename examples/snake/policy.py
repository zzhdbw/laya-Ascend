"""Laya-driven Snake policy with an optional deterministic safety shield.

Adapted from the Laya-AXERA / laya-mlx Snake demo.  The planner features,
prompt wording and guarded execution rule are kept the same; inference now
runs through the upstream `laya` package (CPU or Ascend NPU).
"""

import math
import time
from dataclasses import asdict, dataclass

from game import DIRECTIONS


@dataclass
class Decision:
    probabilities: dict
    proposed: str
    executed: str
    safe_directions: list
    intervened: bool
    dead_end_risk: float
    food_reachable: float
    inference_ms: float
    decision_ms: float
    input_tokens: int
    safe_count: int
    planner_best: str

    def to_dict(self):
        return asdict(self)


class LayaPolicy:
    """Every move asks the resident checkpoint three questions: move, risk, food."""

    def __init__(self, agent, *, guarded=True, prompt="compact"):
        if prompt not in ("compact", "detailed"):
            raise ValueError("prompt must be compact or detailed")
        self.agent = agent
        self.guarded = bool(guarded)
        self.prompt = prompt

    def decide(self, game):
        started = time.perf_counter()
        moves = game.moves()
        safe = [m for m in moves if m.safe]
        if not safe and self.guarded:
            raise RuntimeError("Cycle safety invariant violated: no safe action")
        preferred = max(safe, key=lambda m: m.advance).direction if safe else "NONE"
        reachable, space = game.food_reachability()

        descriptions = {}
        for move in moves:
            if not move.legal:
                descriptions[move.direction] = "Collision: %s. Unsafe." % move.reason
            elif not move.safe:
                descriptions[move.direction] = "Unsafe route. Risk of trapping the snake."
            elif move.eats:
                descriptions[move.direction] = "Safe. Eat the food immediately. Best move."
            elif move.direction == preferred:
                descriptions[move.direction] = "Safe. Best progress toward food."
            else:
                descriptions[move.direction] = "Safe but less progress toward food."

        state = (
            "Snake game. %d safe directions available. "
            "Food reachable through empty cells: %s. Open cells: %d. Snake length: %d. %s"
            % (
                len(safe),
                "yes" if reachable else "no",
                space,
                len(game.body),
                "There is a safe route forward." if safe else "The snake is trapped.",
            )
        )
        questions = {
            "move": {
                "type": "choice",
                "instructions": "Select the safest move with best progress toward food. Avoid collisions.",
                "criteria": descriptions,
            },
            "risk": {
                "type": "noul",
                "instructions": "Is there a safe route forward for the snake?",
            },
            "food": {
                "type": "noul",
                "instructions": "Is food reachable through the currently empty cells?",
            },
        }

        if self.prompt == "compact":
            state = "Safe route: %s. Food reachable through empty cells: %s." % (
                "yes" if safe else "no",
                "yes" if reachable else "no",
            )
            questions["move"]["instructions"] = "Choose the best safe move toward food."
            questions["move"]["criteria"] = {
                m.direction: (
                    "Blocked. Collision."
                    if not m.legal
                    else "Unsafe. Traps the snake."
                    if not m.safe
                    else "Safe. Eat food now. Best."
                    if m.eats
                    else "Safe. Best route to food."
                    if m.direction == preferred
                    else "Safe. Slower route."
                )
                for m in moves
            }
            questions["risk"]["instructions"] = "Is a safe route available?"
            questions["food"]["instructions"] = "Is food reachable through empty cells?"

        inference_start = time.perf_counter()
        output = self.agent.predict(state, questions)
        inference_ms = (time.perf_counter() - inference_start) * 1000.0

        answers = output["answers"]
        probabilities = answers["move"]["probabilities"]
        scores = [
            *probabilities.values(),
            answers["risk"]["noul"],
            answers["food"]["noul"],
        ]
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores):
            raise ValueError("Model returned an invalid probability; no move executed")

        proposed = max(DIRECTIONS, key=probabilities.__getitem__)
        allowed = [m.direction for m in safe]
        executed = (
            max(allowed, key=probabilities.__getitem__)
            if self.guarded and proposed not in allowed
            else proposed
        )
        return Decision(
            probabilities=probabilities,
            proposed=proposed,
            executed=executed,
            safe_directions=allowed,
            intervened=proposed != executed,
            dead_end_risk=1 - answers["risk"]["noul"],
            food_reachable=answers["food"]["noul"],
            inference_ms=inference_ms,
            decision_ms=(time.perf_counter() - started) * 1000.0,
            input_tokens=output["usage"]["input_tokens"],
            safe_count=len(safe),
            planner_best=preferred,
        )
