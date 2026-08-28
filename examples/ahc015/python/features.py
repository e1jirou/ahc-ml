from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from .game import BACK, CELL_COUNT, FRONT, LEFT, RIGHT, SIDE, Board, flavor_totals

BOARD_CHANNELS = 4

_ACTION_VECTORS = {FRONT: (-1, 0), BACK: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}
_VECTOR_ACTIONS = {vector: action for action, vector in _ACTION_VECTORS.items()}


def dynamic_flavor_mapping(flavors: NDArray[np.uint8], placed: int) -> NDArray[np.uint8]:
    """Map flavor labels deterministically while keeping the next flavor first."""
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


def _transform_board(board: Board, rotation: int, mirrored: bool) -> Board:
    transformed = np.rot90(board, rotation)
    if mirrored:
        transformed = transformed[:, ::-1]
    return transformed.copy()


def _transform_vector(vector: tuple[int, int], rotation: int, mirrored: bool) -> tuple[int, int]:
    row, column = vector
    for _ in range(rotation):
        row, column = -column, row
    if mirrored:
        column = -column
    return row, column


def _normalized_to_original_actions(rotation: int, mirrored: bool) -> NDArray[np.int64]:
    original_to_normalized = np.empty(4, dtype=np.int64)
    for original, vector in _ACTION_VECTORS.items():
        original_to_normalized[original] = _VECTOR_ACTIONS[
            _transform_vector(vector, rotation, mirrored)
        ]
    return np.argsort(original_to_normalized)


def normalized_board(
    board: Board,
    flavors: NDArray[np.uint8],
    placed: int,
) -> tuple[Board, NDArray[np.int64]]:
    """Canonicalize flavor labels and all eight square orientations.

    The returned permutation maps actions in the normalized orientation back
    to actions in the original board orientation.
    """
    mapped = dynamic_flavor_mapping(flavors, placed)[board]
    best_board: Board | None = None
    best_transform = (0, False)
    for mirrored in (False, True):
        for rotation in range(4):
            candidate = _transform_board(mapped, rotation, mirrored)
            if best_board is None or candidate.tobytes() < best_board.tobytes():
                best_board = candidate
                best_transform = (rotation, mirrored)
    assert best_board is not None
    return best_board, _normalized_to_original_actions(*best_transform)


def encode_states(
    boards: Sequence[Board] | NDArray[np.uint8],
    placed: Sequence[int] | NDArray[np.int64] | int,
    flavor_sequences: Sequence[NDArray[np.uint8]] | NDArray[np.uint8],
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Encode one pre-tilt board per turn using only occupancy planes."""
    boards_array = np.asarray(boards, dtype=np.uint8).reshape(-1, SIDE, SIDE)
    if np.isscalar(placed):
        placed_array = np.full(len(boards_array), int(placed), dtype=np.int64)
    else:
        placed_array = np.asarray(placed, dtype=np.int64).reshape(-1)
    flavors_array = np.asarray(flavor_sequences, dtype=np.uint8)
    if flavors_array.ndim == 1:
        flavors_array = np.broadcast_to(flavors_array, (len(boards_array), CELL_COUNT))
    if not (len(boards_array) == len(placed_array) == len(flavors_array)):
        raise ValueError("state batch fields have different lengths")

    features = np.zeros((len(boards_array), BOARD_CHANNELS, SIDE, SIDE), dtype=np.float32)
    normalized_to_original = np.empty((len(boards_array), 4), dtype=np.int64)
    for sample, (board, turn, flavors) in enumerate(
        zip(boards_array, placed_array, flavors_array, strict=True)
    ):
        normalized, permutation = normalized_board(board, flavors, int(turn))
        for flavor in range(1, 4):
            features[sample, flavor - 1] = normalized == flavor
        features[sample, 3] = normalized == 0
        normalized_to_original[sample] = permutation
    return features, normalized_to_original
