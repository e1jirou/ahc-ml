from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

SIDE = 10
CELL_COUNT = SIDE * SIDE
ACTION_COUNT = 4

FRONT = 0
BACK = 1
LEFT = 2
RIGHT = 3

Board = NDArray[np.uint8]


def empty_board() -> Board:
    return np.zeros((SIDE, SIDE), dtype=np.uint8)


def place_at_rank(board: Board, rank: int, flavor: int) -> Board:
    empty = np.flatnonzero(board.reshape(-1) == 0)
    if rank < 1 or rank > len(empty):
        raise ValueError(f"rank {rank} is outside 1..={len(empty)}")
    result = board.copy()
    result.reshape(-1)[empty[rank - 1]] = flavor
    return result


def inserted(board: Board, cell: int, flavor: int) -> Board:
    if cell < 0 or cell >= CELL_COUNT or board.reshape(-1)[cell] != 0:
        raise ValueError(f"cell {cell} is not empty")
    result = board.copy()
    result.reshape(-1)[cell] = flavor
    return result


def tilt(board: Board, action: int) -> Board:
    result = empty_board()
    if action in (FRONT, BACK):
        for column in range(SIDE):
            values = board[:, column]
            values = values[values != 0]
            if action == FRONT:
                result[: len(values), column] = values
            else:
                result[SIDE - len(values) :, column] = values
    elif action in (LEFT, RIGHT):
        for row in range(SIDE):
            values = board[row, :]
            values = values[values != 0]
            if action == LEFT:
                result[row, : len(values)] = values
            else:
                result[row, SIDE - len(values) :] = values
    else:
        raise ValueError(f"invalid action: {action}")
    return result


def afterstates(board: Board) -> NDArray[np.uint8]:
    return np.stack([tilt(board, action) for action in range(ACTION_COUNT)])


def connectivity_numerator(board: Board) -> int:
    visited = np.zeros((SIDE, SIDE), dtype=np.bool_)
    numerator = 0
    for start_row in range(SIDE):
        for start_column in range(SIDE):
            flavor = int(board[start_row, start_column])
            if flavor == 0 or visited[start_row, start_column]:
                continue
            visited[start_row, start_column] = True
            stack = [(start_row, start_column)]
            size = 0
            while stack:
                row, column = stack.pop()
                size += 1
                for next_row, next_column in (
                    (row - 1, column),
                    (row + 1, column),
                    (row, column - 1),
                    (row, column + 1),
                ):
                    if (
                        0 <= next_row < SIDE
                        and 0 <= next_column < SIDE
                        and not visited[next_row, next_column]
                        and board[next_row, next_column] == flavor
                    ):
                        visited[next_row, next_column] = True
                        stack.append((next_row, next_column))
            numerator += size * size
    return numerator


def flavor_totals(flavors: NDArray[np.uint8]) -> NDArray[np.int64]:
    if flavors.shape != (CELL_COUNT,) or np.any((flavors < 1) | (flavors > 3)):
        raise ValueError("flavors must contain 100 values in 1..=3")
    return np.bincount(flavors, minlength=4)[1:].astype(np.int64)


def denominator(flavors: NDArray[np.uint8]) -> int:
    totals = flavor_totals(flavors)
    return int(np.dot(totals, totals))


def potential(board: Board, score_denominator: int) -> float:
    return connectivity_numerator(board) / score_denominator


def official_score(board: Board, score_denominator: int) -> int:
    return int(np.floor(1_000_000 * potential(board, score_denominator) + 0.5))


@dataclass(slots=True)
class EpisodeState:
    flavors: NDArray[np.uint8]
    board: Board
    placed: int = 0

    @classmethod
    def new(cls, flavors: NDArray[np.uint8]) -> EpisodeState:
        flavor_totals(flavors)
        return cls(flavors.copy(), empty_board())

    @property
    def denominator(self) -> int:
        return denominator(self.flavors)

    def place_cell(self, cell: int) -> None:
        self.board = inserted(self.board, cell, int(self.flavors[self.placed]))
        self.placed += 1

    def place_rank(self, rank: int) -> None:
        self.board = place_at_rank(self.board, rank, int(self.flavors[self.placed]))
        self.placed += 1

    def apply(self, action: int) -> None:
        self.board = tilt(self.board, action)
