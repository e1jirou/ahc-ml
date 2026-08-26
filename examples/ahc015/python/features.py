from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from .game import (
    BACK,
    CELL_COUNT,
    FRONT,
    LEFT,
    RIGHT,
    SIDE,
    Board,
    denominator,
    flavor_totals,
    potential,
)

BOARD_CHANNELS = 12
FUTURE_CHANNELS = 3
FUTURE_LENGTH = CELL_COUNT
POTENTIAL_CHANNEL = 11


def dynamic_flavor_mapping(flavors: NDArray[np.uint8], placed: int) -> NDArray[np.uint8]:
    totals = flavor_totals(flavors)
    first = np.full(3, CELL_COUNT + 1, dtype=np.int64)
    for flavor in range(1, 4):
        positions = np.flatnonzero(flavors == flavor)
        if len(positions):
            first[flavor - 1] = positions[0]

    def key(original: int) -> tuple[int, int, int]:
        positions = np.flatnonzero(flavors[placed:] == original + 1)
        next_position = placed + int(positions[0]) if len(positions) else CELL_COUNT + 1
        return next_position, int(totals[original]), int(first[original])

    order = sorted(range(3), key=key)
    mapping = np.zeros(4, dtype=np.uint8)
    for canonical, original in enumerate(order, start=1):
        mapping[original + 1] = canonical
    return mapping


def rotate_action_to_front(board: Board, action: int) -> Board:
    if action == FRONT:
        return board.copy()
    if action == BACK:
        return np.rot90(board, 2).copy()
    if action == LEFT:
        return np.rot90(board, -1).copy()
    if action == RIGHT:
        return np.rot90(board, 1).copy()
    raise ValueError(f"invalid action: {action}")


def normalized_board(
    board: Board,
    action: int,
    flavors: NDArray[np.uint8],
    placed: int,
) -> Board:
    mapping = dynamic_flavor_mapping(flavors, placed)
    normalized = mapping[rotate_action_to_front(board, action)]
    mirrored = normalized[:, ::-1]
    if tuple(mirrored.reshape(-1)) < tuple(normalized.reshape(-1)):
        return mirrored.copy()
    return normalized.copy()


def component_size_planes(board: Board) -> NDArray[np.float32]:
    result = np.zeros((3, SIDE, SIDE), dtype=np.float32)
    visited = np.zeros((SIDE, SIDE), dtype=np.bool_)
    for start_row in range(SIDE):
        for start_column in range(SIDE):
            flavor = int(board[start_row, start_column])
            if flavor == 0 or visited[start_row, start_column]:
                continue
            visited[start_row, start_column] = True
            stack = [(start_row, start_column)]
            component: list[tuple[int, int]] = []
            while stack:
                row, column = stack.pop()
                component.append((row, column))
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
            value = len(component) / CELL_COUNT
            for row, column in component:
                result[flavor - 1, row, column] = value
    return result


def encode_afterstates(
    boards: Sequence[Board] | NDArray[np.uint8],
    actions: Sequence[int] | NDArray[np.int64],
    placed: Sequence[int] | NDArray[np.int64] | int,
    flavor_sequences: Sequence[NDArray[np.uint8]] | NDArray[np.uint8],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    boards_array = np.asarray(boards, dtype=np.uint8).reshape(-1, SIDE, SIDE)
    actions_array = np.asarray(actions, dtype=np.int64).reshape(-1)
    if np.isscalar(placed):
        placed_array = np.full(len(boards_array), int(placed), dtype=np.int64)
    else:
        placed_array = np.asarray(placed, dtype=np.int64).reshape(-1)
    flavors_array = np.asarray(flavor_sequences, dtype=np.uint8)
    if flavors_array.ndim == 1:
        flavors_array = np.broadcast_to(flavors_array, (len(boards_array), CELL_COUNT))
    if not (len(boards_array) == len(actions_array) == len(placed_array) == len(flavors_array)):
        raise ValueError("afterstate batch fields have different lengths")

    board_features = np.zeros((len(boards_array), BOARD_CHANNELS, SIDE, SIDE), dtype=np.float32)
    future_features = np.zeros(
        (len(boards_array), FUTURE_CHANNELS, FUTURE_LENGTH), dtype=np.float32
    )
    for sample, (board, action, turn, flavors) in enumerate(
        zip(boards_array, actions_array, placed_array, flavors_array, strict=True)
    ):
        turn = int(turn)
        mapping = dynamic_flavor_mapping(flavors, turn)
        normalized = normalized_board(board, int(action), flavors, turn)
        for flavor in range(1, 4):
            board_features[sample, flavor - 1] = normalized == flavor
        board_features[sample, 3] = normalized == 0
        board_features[sample, 4:7] = component_size_planes(normalized)
        board_features[sample, 7].fill(turn / CELL_COUNT)

        remaining = np.bincount(flavors[turn:], minlength=4)[1:]
        for original in range(3):
            canonical = int(mapping[original + 1]) - 1
            board_features[sample, 8 + canonical].fill(remaining[original] / CELL_COUNT)
        board_features[sample, POTENTIAL_CHANNEL].fill(potential(board, denominator(flavors)))

        for position in range(turn, CELL_COUNT):
            original = flavors[position]
            canonical = int(mapping[int(original)]) - 1
            future_features[sample, canonical, position] = 1.0
    return board_features, future_features
