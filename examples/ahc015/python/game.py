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


def place_at_ranks(
    boards: NDArray[np.uint8],
    ranks: NDArray[np.integer],
    flavors: NDArray[np.integer],
) -> NDArray[np.uint8]:
    """Place one item in every board without a Python loop over episodes."""
    boards_array = np.asarray(boards, dtype=np.uint8)
    ranks_array = np.asarray(ranks).reshape(-1)
    flavors_array = np.asarray(flavors, dtype=np.uint8).reshape(-1)
    if boards_array.ndim != 3 or boards_array.shape[1:] != (SIDE, SIDE):
        raise ValueError("boards must have shape [episodes, 10, 10]")
    if len(boards_array) != len(ranks_array) or len(boards_array) != len(flavors_array):
        raise ValueError("batched placement fields have different lengths")

    flat = boards_array.reshape(len(boards_array), CELL_COUNT)
    empty = flat == 0
    valid = (ranks_array >= 1) & (ranks_array <= empty.sum(axis=1))
    if not np.all(valid):
        raise ValueError("rank is outside the remaining empty cells")
    selected = np.argmax(np.cumsum(empty, axis=1) == ranks_array[:, None], axis=1)
    result = flat.copy()
    result[np.arange(len(result)), selected] = flavors_array
    return result.reshape(-1, SIDE, SIDE)


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


def tilt_batch(boards: NDArray[np.uint8], action: int) -> NDArray[np.uint8]:
    """Apply one stable tilt to a batch of boards using NumPy indexing."""
    boards_array = np.asarray(boards, dtype=np.uint8)
    if boards_array.ndim != 3 or boards_array.shape[1:] != (SIDE, SIDE):
        raise ValueError("boards must have shape [episodes, 10, 10]")
    mask = boards_array != 0
    episode_indices, row_indices, column_indices = np.nonzero(mask)
    result = np.zeros_like(boards_array)
    if action == FRONT:
        destinations = np.cumsum(mask, axis=1) - 1
        destination_rows = destinations[episode_indices, row_indices, column_indices]
        result[episode_indices, destination_rows, column_indices] = boards_array[mask]
    elif action == BACK:
        counts = mask.sum(axis=1)
        destinations = SIDE - counts[:, None, :] + np.cumsum(mask, axis=1) - 1
        destination_rows = destinations[episode_indices, row_indices, column_indices]
        result[episode_indices, destination_rows, column_indices] = boards_array[mask]
    elif action == LEFT:
        destinations = np.cumsum(mask, axis=2) - 1
        destination_columns = destinations[episode_indices, row_indices, column_indices]
        result[episode_indices, row_indices, destination_columns] = boards_array[mask]
    elif action == RIGHT:
        counts = mask.sum(axis=2)
        destinations = SIDE - counts[:, :, None] + np.cumsum(mask, axis=2) - 1
        destination_columns = destinations[episode_indices, row_indices, column_indices]
        result[episode_indices, row_indices, destination_columns] = boards_array[mask]
    else:
        raise ValueError(f"invalid action: {action}")
    return result


def afterstates_batch(boards: NDArray[np.uint8]) -> NDArray[np.uint8]:
    return np.stack([tilt_batch(boards, action) for action in range(ACTION_COUNT)], axis=1)


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
