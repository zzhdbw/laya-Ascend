#!/usr/bin/env python3
"""Terminal Snake demo where every move is decided by Laya.

Examples:
  .venv/bin/python examples/snake/play.py --device cpu --steps 20
  .venv/bin/python examples/snake/play.py --device npu:0 --steps 50
"""

import argparse
import sys
import time

from game import SnakeGame
from policy import LayaPolicy
from runtime import DEFAULT_MODEL, load_agent


def render(game):
    grid = [["." for _ in range(game.width)] for _ in range(game.height)]
    for index, (x, y) in enumerate(game.body):
        grid[y][x] = "@" if index == 0 else "o"
    if game.food is not None:
        fx, fy = game.food
        grid[fy][fx] = "*"
    border = "+" + "-" * game.width + "+"
    lines = [border]
    lines.extend("|" + "".join(row) + "|" for row in grid)
    lines.append(border)
    lines.append(
        "score=%d moves=%d length=%d alive=%s won=%s"
        % (game.score, game.ticks, len(game.body), game.alive, game.won)
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="path to one Laya checkpoint")
    parser.add_argument("--device", default="auto", help="auto, cpu, npu:0, npu:1 ...")
    parser.add_argument("--backend", choices=["torch", "aisbench"], default="torch",
                        help="AISBench requires --model pointing to an exported OM bundle")
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--delay", type=float, default=0.05, help="seconds between rendered steps")
    parser.add_argument("--prompt", choices=("compact", "detailed"), default="compact")
    parser.add_argument("--no-guard", action="store_true", help="disable the cycle safety shield")
    args = parser.parse_args()

    game = SnakeGame(width=args.width, height=args.height, seed=args.seed)
    agent = load_agent(args.model, args.device, args.backend)
    policy = LayaPolicy(agent, guarded=not args.no_guard, prompt=args.prompt)

    for _ in range(max(0, args.steps)):
        if not game.alive or game.won:
            break
        if args.delay:
            time.sleep(args.delay)
        print("\033c", end="")

        decision = policy.decide(game)
        print(render(game))
        print(
            "proposed=%s executed=%s intervened=%s inference=%.1fms risk=%.2f food=%.2f"
            % (
                decision.proposed,
                decision.executed,
                decision.intervened,
                decision.inference_ms,
                decision.dead_end_risk,
                decision.food_reachable,
            )
        )
        game.step(decision.executed)

    print(render(game))
    if game.won:
        print("Snake won!")
    elif not game.alive:
        print("Snake died: %s" % game.death_reason)
    else:
        print("Stopped after %d steps." % game.ticks)


if __name__ == "__main__":
    main()
