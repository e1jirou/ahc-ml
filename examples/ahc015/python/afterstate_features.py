from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from .features import BOARD_CHANNELS, dynamic_flavor_mapping
from .game import BACK, CELL_COUNT, FRONT, LEFT, RIGHT, SIDE, Board


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


def normalized_afterstate(
    board: Board,
    action: int,
    flavors: NDArray[np.uint8],
    placed: int,
) -> Board:
    mapping = dynamic_flavor_mapping(flavors, placed)
    normalized = mapping[rotate_action_to_front(board, action)]
    mirrored = normalized[:, ::-1]
    if mirrored.tobytes() < normalized.tobytes():
        return mirrored.copy()
    return normalized.copy()


def encode_afterstates(
    boards: Sequence[Board] | NDArray[np.uint8],
    actions: Sequence[int] | NDArray[np.int64],
    placed: Sequence[int] | NDArray[np.int64] | int,
    flavor_sequences: Sequence[NDArray[np.uint8]] | NDArray[np.uint8],
) -> NDArray[np.float32]:
    """Encode action-normalized afterstates using only four occupancy planes."""
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

    features = np.zeros((len(boards_array), BOARD_CHANNELS, SIDE, SIDE), dtype=np.float32)
    for sample, (board, action, turn, flavors) in enumerate(
        zip(boards_array, actions_array, placed_array, flavors_array, strict=True)
    ):
        normalized = normalized_afterstate(board, int(action), flavors, int(turn))
        for flavor in range(1, 4):
            features[sample, flavor - 1] = normalized == flavor
        features[sample, 3] = normalized == 0
    return features
