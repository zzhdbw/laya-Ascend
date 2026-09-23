"""Deterministic Tetris rules with drop-from-top placement enumeration.

One decision step places a whole piece: the game enumerates every reachable
(rotation, column) drop of the current piece, scores each with classic
heuristics, and exposes a shortlist of distinct candidates for the policy.
"""

import random

WIDTH, HEIGHT = 10, 20
SHORTLIST = 4
DANGER_HEIGHT = 15  # stacks above this are the "fatal" tier for the shield

# Rotation states as (x, y) cells, y growing downward.
SHAPES = {
    "I": [[(0, 0), (1, 0), (2, 0), (3, 0)], [(0, 0), (0, 1), (0, 2), (0, 3)]],
    "O": [[(0, 0), (1, 0), (0, 1), (1, 1)]],
    "T": [
        [(0, 0), (1, 0), (2, 0), (1, 1)],
        [(1, 0), (0, 1), (1, 1), (1, 2)],
        [(1, 0), (0, 1), (1, 1), (2, 1)],
        [(0, 0), (0, 1), (1, 1), (0, 2)],
    ],
    "S": [[(1, 0), (2, 0), (0, 1), (1, 1)], [(0, 0), (0, 1), (1, 1), (1, 2)]],
    "Z": [[(0, 0), (1, 0), (1, 1), (2, 1)], [(1, 0), (0, 1), (1, 1), (0, 2)]],
    "J": [
        [(0, 0), (0, 1), (1, 1), (2, 1)],
        [(0, 0), (1, 0), (0, 1), (0, 2)],
        [(0, 0), (1, 0), (2, 0), (2, 1)],
        [(1, 0), (1, 1), (0, 2), (1, 2)],
    ],
    "L": [
        [(2, 0), (0, 1), (1, 1), (2, 1)],
        [(0, 0), (0, 1), (0, 2), (1, 2)],
        [(0, 0), (1, 0), (2, 0), (0, 1)],
        [(0, 0), (1, 0), (1, 1), (1, 2)],
    ],
}
SHAPE_IDS = {name: i + 1 for i, name in enumerate(SHAPES)}


def _column_heights(board):
    heights = [0] * WIDTH
    for x in range(WIDTH):
        for y in range(HEIGHT):
            if board[y][x]:
                heights[x] = HEIGHT - y
                break
    return heights


def _count_holes(board):
    holes = 0
    for x in range(WIDTH):
        seen = False
        for y in range(HEIGHT):
            if board[y][x]:
                seen = True
            elif seen:
                holes += 1
    return holes


class TetrisGame:
    def __init__(self, seed=7):
        self.seed = seed
        self.rng = random.Random(seed)
        self.board = [[0] * WIDTH for _ in range(HEIGHT)]
        self.bag = []
        self.score = self.lines = self.pieces = 0
        self.alive = True
        self.current = self._draw()
        self.next_piece = self._draw()

    def _draw(self):
        if not self.bag:
            self.bag = list(SHAPES)
            self.rng.shuffle(self.bag)
        return self.bag.pop()

    def _drop_row(self, cells, col):
        """Resting y offset of a drop in this column; None if it cannot spawn."""
        if not self._fits(cells, col, 0):
            return None
        row = 0
        while self._fits(cells, col, row + 1):
            row += 1
        return row

    def _fits(self, cells, col, row):
        for x, y in cells:
            bx, by = col + x, row + y
            if bx < 0 or bx >= WIDTH or by < 0 or by >= HEIGHT or self.board[by][bx]:
                return False
        return True

    def _simulate(self, cells, col, row):
        board = [r[:] for r in self.board]
        for x, y in cells:
            board[row + y][col + x] = SHAPE_IDS[self.current]
        full = [y for y in range(HEIGHT) if all(board[y])]
        for y in full:
            del board[y]
            board.insert(0, [0] * WIDTH)
        return board, len(full)

    def evaluate(self, rotation, col):
        """Metrics for one drop placement of the current piece, or None if illegal."""
        rotations = SHAPES[self.current]
        if not 0 <= rotation < len(rotations):
            return None
        cells = rotations[rotation]
        if not 0 <= col <= WIDTH - 1 - max(x for x, _ in cells):
            return None
        row = self._drop_row(cells, col)
        if row is None:
            return None
        holes_before = _count_holes(self.board)
        agg_before = sum(_column_heights(self.board))
        board_after, cleared = self._simulate(cells, col, row)
        heights = _column_heights(board_after)
        holes_delta = _count_holes(board_after) - holes_before
        bump = sum(abs(a - b) for a, b in zip(heights, heights[1:]))
        score = (
            3.0 * cleared
            - 5.0 * holes_delta
            - 0.4 * (sum(heights) - agg_before)
            - 0.2 * bump
        )
        return {
            "rotation": rotation,
            "col": col,
            "row": row,
            "cells": [[col + x, row + y] for x, y in cells],
            "lines": cleared,
            "holes": max(0, holes_delta),
            "height": max(heights),
            "heuristic": round(score, 2),
        }

    def candidates(self):
        """Heuristic-ranked shortlist of distinct placements for the current piece."""
        scored = []
        for rotation, cells in enumerate(SHAPES[self.current]):
            max_x = max(x for x, _ in cells)
            for col in range(WIDTH - max_x):
                cand = self.evaluate(rotation, col)
                if cand is not None:
                    scored.append(cand)
        scored.sort(key=lambda c: c["heuristic"], reverse=True)
        shortlist, seen = [], set()
        for cand in scored:  # prefer distinct outcomes so the rating is a real choice
            signature = (cand["lines"], cand["holes"], cand["height"] > DANGER_HEIGHT)
            if signature in seen:
                continue
            seen.add(signature)
            shortlist.append(cand)
            if len(shortlist) == SHORTLIST:
                break
        for cand in scored:
            if len(shortlist) == SHORTLIST:
                break
            if cand not in shortlist:
                shortlist.append(cand)
        return shortlist

    def apply(self, cand):
        if not self.alive:
            raise RuntimeError("Cannot place on a finished game")
        cells = SHAPES[self.current][cand["rotation"]]
        self.board, cleared = self._simulate(cells, cand["col"], cand["row"])
        self.lines += cleared
        self.score += (0, 100, 300, 500, 800)[cleared]
        self.pieces += 1
        self.current, self.next_piece = self.next_piece, self._draw()
        if not self.candidates():
            self.alive = False

    def snapshot(self):
        return {
            "width": WIDTH,
            "height": HEIGHT,
            "seed": self.seed,
            "board": self.board,
            "current": self.current,
            "next": self.next_piece,
            "score": self.score,
            "lines": self.lines,
            "pieces": self.pieces,
            "alive": self.alive,
        }
