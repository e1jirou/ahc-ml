from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from .features import dynamic_flavor_mapping
from .game import ACTION_COUNT, CELL_COUNT, SIDE

FEATURE_COUNT = 16


def handcrafted_prior_features(
    candidates: NDArray[np.uint8],
    flavors: NDArray[np.uint8],
    placed: int,
) -> NDArray[np.float32]:
    """Rotation-invariant structural features for each candidate afterstate."""
    boards = np.asarray(candidates, dtype=np.uint8).reshape(-1, ACTION_COUNT, SIDE, SIDE)
    sequences = np.asarray(flavors, dtype=np.uint8).reshape(-1, CELL_COUNT)
    if len(boards) != len(sequences):
        raise ValueError("candidate and flavor batch sizes differ")
    output = np.zeros((len(boards), ACTION_COUNT, FEATURE_COUNT), dtype=np.float32)
    for sample, sequence in enumerate(sequences):
        mapping = dynamic_flavor_mapping(sequence, placed)
        for action in range(ACTION_COUNT):
            board = mapping[boards[sample, action]]
            same_edges = 0
            for flavor in range(1, 4):
                mask = board == flavor
                visited = np.zeros((SIDE, SIDE), dtype=np.bool_)
                component_square = 0
                largest = 0
                components = 0
                for row, column in zip(*np.nonzero(mask), strict=True):
                    if visited[row, column]:
                        continue
                    components += 1
                    visited[row, column] = True
                    stack = [(int(row), int(column))]
                    size = 0
                    while stack:
                        current_row, current_column = stack.pop()
                        size += 1
                        for next_row, next_column in (
                            (current_row - 1, current_column),
                            (current_row + 1, current_column),
                            (current_row, current_column - 1),
                            (current_row, current_column + 1),
                        ):
                            if (
                                0 <= next_row < SIDE
                                and 0 <= next_column < SIDE
                                and mask[next_row, next_column]
                                and not visited[next_row, next_column]
                            ):
                                visited[next_row, next_column] = True
                                stack.append((next_row, next_column))
                    component_square += size * size
                    largest = max(largest, size)

                empty_contacts = 0
                exposure_edges = 0
                for row, column in zip(*np.nonzero(mask), strict=True):
                    for next_row, next_column in (
                        (row - 1, column),
                        (row + 1, column),
                        (row, column - 1),
                        (row, column + 1),
                    ):
                        if not (0 <= next_row < SIDE and 0 <= next_column < SIDE):
                            exposure_edges += 1
                        elif board[next_row, next_column] == 0:
                            empty_contacts += 1
                            exposure_edges += 1
                        elif board[next_row, next_column] == flavor:
                            same_edges += 1
                base = (flavor - 1) * 5
                output[sample, action, base : base + 5] = (
                    component_square / 10_000.0,
                    largest / 100.0,
                    components / 100.0,
                    empty_contacts / 180.0,
                    exposure_edges / 220.0,
                )
            output[sample, action, 15] = same_edges / 360.0
    return output


class TinyMctsPrior(nn.Module):
    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.hidden = nn.Linear(FEATURE_COUNT, hidden)
        self.output = nn.Linear(hidden, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.output(torch.relu(self.hidden(features))).squeeze(-1)
