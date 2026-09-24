#!/usr/bin/env python3
"""Terminal Tetris demo where Laya rates each candidate placement.

Examples:
  .venv/bin/python examples/tetris/play.py --device cpu --pieces 10
  .venv/bin/python examples/tetris/play.py --device npu:0 --pieces 20
"""

import argparse
import time

from game import TetrisGame
from policy import LayaTetrisPolicy
from runtime import DEFAULT_MODEL, load_agent


def render(game):
    snap = game.snapshot()
    lines = ["+" + "-" * snap["width"] + "+"]
    for row in snap["board"]:
        lines.append("|" + "".join("#" if cell else "." for cell in row) + "|")
    lines.append("+" + "-" * snap["width"] + "+")
    lines.append(
        "score=%d lines=%d pieces=%d current=%s next=%s alive=%s"
        % (snap["score"], snap["lines"], snap["pieces"], snap["current"], snap["next"], snap["alive"])
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="path to one Laya checkpoint")
    parser.add_argument("--device", default="auto", help="auto, cpu, npu:0, npu:1 ...")
    parser.add_argument("--backend", choices=["torch", "aisbench"], default="torch",
                        help="AISBench requires --model pointing to an exported OM bundle")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pieces", type=int, default=20)
    parser.add_argument("--delay", type=float, default=0.05)
    parser.add_argument("--no-guard", action="store_true", help="disable the danger-height shield")
    args = parser.parse_args()

    game = TetrisGame(seed=args.seed)
    agent = load_agent(args.model, args.device, args.backend)
    policy = LayaTetrisPolicy(agent, guarded=not args.no_guard)

    placed = 0
    while game.alive and placed < max(0, args.pieces):
        if args.delay:
            time.sleep(args.delay)
        decision = policy.decide(game)
        print("\033c", end="")
        print(render(game))
        print("candidates:")
        for i, cand in enumerate(decision.candidates):
            mark = "*" if i == decision.executed else " "
            print(
                " %s [%d] lines=%d holes=%d height=%d heuristic=%.2f p_good=%.4f  %s"
                % (
                    mark,
                    i,
                    cand["lines"],
                    cand["holes"],
                    cand["height"],
                    cand["heuristic"],
                    cand["p_good"],
                    cand["statement"],
                )
            )
        print(
            "proposed=%d executed=%d intervened=%s inference=%.1fms tokens=%d"
            % (
                decision.proposed,
                decision.executed,
                decision.intervened,
                decision.inference_ms,
                decision.input_tokens,
            )
        )
        game.apply(decision.candidates[decision.executed])
        placed += 1

    print(render(game))
    if game.alive:
        print("Stopped after %d pieces." % placed)
    else:
        print("Game over after %d pieces." % placed)


if __name__ == "__main__":
    main()
