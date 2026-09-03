from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from .features import dynamic_flavor_mapping
from .game import (
    BACK,
    CELL_COUNT,
    FRONT,
    LEFT,
    RIGHT,
    SIDE,
    Board,
    flavor_totals,
)
from .model import ResidualDepthwiseBlock, dimensions_from_state_dict

LEGACY_BOARD_CHANNELS = 15
LEGACY_FUTURE_CHANNELS = 3
LEGACY_FUTURE_LENGTH = CELL_COUNT


class LegacyFutureValueNet(nn.Module):
    """The 144-channel pre-compact model used by the saved FiLM checkpoints."""

    def __init__(self, channels: int, residual_blocks: int) -> None:
        super().__init__()
        self.board_stem = nn.Conv2d(
            LEGACY_BOARD_CHANNELS, channels, kernel_size=3, padding=1
        )
        self.blocks = nn.ModuleList(
            ResidualDepthwiseBlock(channels) for _ in range(residual_blocks)
        )
        self.future_fc1 = nn.Linear(
            LEGACY_FUTURE_CHANNELS * LEGACY_FUTURE_LENGTH, channels
        )
        self.future_fc2 = nn.Linear(channels, channels)
        self.film = nn.Linear(channels, channels * 2)
        self.fusion_fc = nn.Linear(channels * 2, channels * 2)
        self.output = nn.Linear(channels * 2, 1)

    def forward(
        self,
        board_inputs: torch.Tensor,
        future_inputs: torch.Tensor,
    ) -> torch.Tensor:
        future = torch.relu(self.future_fc1(future_inputs.flatten(start_dim=1)))
        future = torch.relu(self.future_fc2(future))
        gamma, beta = self.film(future).chunk(2, dim=1)

        board = torch.relu(self.board_stem(board_inputs))
        board = board * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        for block in self.blocks:
            board = block(board)
        board = board.mean(dim=(2, 3))

        fused = torch.relu(self.fusion_fc(torch.cat((board, future), dim=1)))
        return self.output(fused).squeeze(1)


def legacy_model_from_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> LegacyFutureValueNet:
    channels, residual_blocks = dimensions_from_state_dict(state_dict)
    model = LegacyFutureValueNet(channels, residual_blocks)
    model.load_state_dict(state_dict)
    return model


def _rotate_action_to_front(board: Board, action: int) -> Board:
    if action == FRONT:
        return board.copy()
    if action == BACK:
        return np.rot90(board, 2).copy()
    if action == LEFT:
        return np.rot90(board, -1).copy()
    if action == RIGHT:
        return np.rot90(board, 1).copy()
    raise ValueError(f"invalid action: {action}")


def _component_planes_and_numerator(
    board: Board,
) -> tuple[NDArray[np.float32], int]:
    planes = np.zeros((3, SIDE, SIDE), dtype=np.float32)
    visited = np.zeros((SIDE, SIDE), dtype=np.bool_)
    numerator = 0
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
            size = len(component)
            numerator += size * size
            value = size / CELL_COUNT
            for row, column in component:
                planes[flavor - 1, row, column] = value
    return planes, numerator


def encode_legacy_afterstates(
    boards: Sequence[Board] | NDArray[np.uint8],
    actions: Sequence[int] | NDArray[np.int64],
    placed: int,
    flavor_sequences: NDArray[np.uint8],
    score_denominators: NDArray[np.integer],
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Reproduce the exact 15-plane board and left-aligned future encoding."""
    boards_array = np.asarray(boards, dtype=np.uint8).reshape(-1, SIDE, SIDE)
    actions_array = np.asarray(actions, dtype=np.int64).reshape(-1)
    flavors_array = np.asarray(flavor_sequences, dtype=np.uint8)
    denominators_array = np.asarray(score_denominators).reshape(-1)
    if not (
        len(boards_array)
        == len(actions_array)
        == len(flavors_array)
        == len(denominators_array)
    ):
        raise ValueError("legacy afterstate batch fields have different lengths")

    board_features = np.zeros(
        (len(boards_array), LEGACY_BOARD_CHANNELS, SIDE, SIDE), dtype=np.float32
    )
    future_features = np.zeros(
        (len(boards_array), LEGACY_FUTURE_CHANNELS, LEGACY_FUTURE_LENGTH),
        dtype=np.float32,
    )
    potentials = np.empty(len(boards_array), dtype=np.float32)
    for sample, (board, action, flavors, score_denominator) in enumerate(
        zip(
            boards_array,
            actions_array,
            flavors_array,
            denominators_array,
            strict=True,
        )
    ):
        mapping = dynamic_flavor_mapping(flavors, placed)
        normalized = mapping[_rotate_action_to_front(board, int(action))]
        mirrored = normalized[:, ::-1]
        if tuple(mirrored.reshape(-1)) < tuple(normalized.reshape(-1)):
            normalized = mirrored.copy()

        for flavor in range(1, 4):
            board_features[sample, flavor - 1] = normalized == flavor
        board_features[sample, 3] = normalized == 0
        component_planes, numerator = _component_planes_and_numerator(normalized)
        board_features[sample, 4:7] = component_planes
        board_features[sample, 7].fill(placed / CELL_COUNT)

        totals = flavor_totals(flavors)
        remaining = np.bincount(flavors[placed:], minlength=4)[1:]
        for original in range(3):
            canonical = int(mapping[original + 1]) - 1
            board_features[sample, 8 + canonical].fill(totals[original] / CELL_COUNT)
            board_features[sample, 11 + canonical].fill(
                remaining[original] / CELL_COUNT
            )
        potentials[sample] = numerator / int(score_denominator)
        board_features[sample, 14].fill(potentials[sample])

        canonical_future = mapping[flavors[placed:]].astype(np.intp) - 1
        positions = np.arange(len(canonical_future))
        future_features[sample, canonical_future, positions] = 1.0
    return board_features, future_features, potentials


def ablate_legacy_futures(
    future_features: NDArray[np.float32],
    mode: str,
    *,
    episode_permutation: NDArray[np.int64] | None = None,
    order_priorities: NDArray[np.float64] | None = None,
    remaining_length: int,
) -> NDArray[np.float32]:
    if mode == "correct":
        return future_features
    if mode == "zero":
        return np.zeros_like(future_features)
    episodes = len(future_features) // 4
    grouped = future_features.reshape(
        episodes, 4, LEGACY_FUTURE_CHANNELS, LEGACY_FUTURE_LENGTH
    )
    if mode == "episode_shuffle":
        if episode_permutation is None or len(episode_permutation) != episodes:
            raise ValueError("episode_shuffle requires one permutation entry per episode")
        return grouped[episode_permutation].reshape(future_features.shape)
    if mode == "order_shuffle":
        if order_priorities is None or order_priorities.shape != (episodes, CELL_COUNT):
            raise ValueError("order_shuffle requires [episodes, 100] priorities")
        order = np.argsort(order_priorities[:, -remaining_length:], axis=1)
        base = grouped[:, 0].copy()
        base[:, :, :remaining_length] = np.take_along_axis(
            base[:, :, :remaining_length], order[:, None, :], axis=2
        )
        return np.repeat(base[:, None], 4, axis=1).reshape(future_features.shape)
    raise ValueError("mode must be correct, zero, episode_shuffle, or order_shuffle")
