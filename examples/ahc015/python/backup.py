from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

from .features import encode_afterstates
from .game import ACTION_COUNT, CELL_COUNT, SIDE, afterstates, denominator, inserted, potential
from .replay import ReplayBatch


def _batched_predict(
    model: torch.nn.Module,
    features: NDArray[np.float32],
    device: torch.device,
    batch_size: int,
) -> NDArray[np.float32]:
    predictions = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            inputs = torch.from_numpy(features[start : start + batch_size]).to(device)
            predictions.append(model(inputs).cpu().numpy())
    return np.concatenate(predictions).astype(np.float32, copy=False)


def bellman_residual_targets(
    batch: ReplayBatch,
    online_model: torch.nn.Module,
    target_model: torch.nn.Module,
    device: torch.device,
    rng: np.random.Generator,
    *,
    placement_samples: int,
    enumerate_threshold: int,
    inference_batch_size: int,
) -> NDArray[np.float32]:
    sample_count = len(batch.placed)
    current_phi = np.empty(sample_count, dtype=np.float32)
    absolute_targets = np.empty(sample_count, dtype=np.float32)

    next_boards: list[NDArray[np.uint8]] = []
    next_actions: list[int] = []
    next_placed: list[int] = []
    next_flavors: list[NDArray[np.uint8]] = []
    next_phi: list[float] = []
    parent_groups: list[tuple[int, int, int]] = []

    for sample in range(sample_count):
        board = batch.boards[sample].reshape(SIDE, SIDE)
        flavors = batch.flavors[sample]
        placed = int(batch.placed[sample])
        score_denominator = denominator(flavors)
        current_phi[sample] = potential(board, score_denominator)
        empty = np.flatnonzero(batch.boards[sample] == 0)
        if placed == CELL_COUNT - 1:
            final_board = inserted(board, int(empty[0]), int(flavors[placed]))
            absolute_targets[sample] = potential(final_board, score_denominator)
            continue

        if len(empty) <= enumerate_threshold:
            placements = empty
        else:
            placements = rng.choice(
                empty,
                size=min(placement_samples, len(empty)),
                replace=False,
            )
        group_start = len(next_boards)
        for cell in placements:
            placed_board = inserted(board, int(cell), int(flavors[placed]))
            candidates = afterstates(placed_board)
            for action in range(ACTION_COUNT):
                next_boards.append(candidates[action])
                next_actions.append(action)
                next_placed.append(placed + 1)
                next_flavors.append(flavors)
                next_phi.append(potential(candidates[action], score_denominator))
        parent_groups.append((sample, group_start, len(placements)))

    if next_boards:
        features = encode_afterstates(
            np.asarray(next_boards, dtype=np.uint8),
            np.asarray(next_actions, dtype=np.uint8),
            np.asarray(next_placed, dtype=np.uint8),
            np.asarray(next_flavors, dtype=np.uint8),
        )
        online = _batched_predict(online_model, features, device, inference_batch_size)
        target = _batched_predict(target_model, features, device, inference_batch_size)
        phi = np.asarray(next_phi, dtype=np.float32)
        for sample, group_start, placements in parent_groups:
            values = []
            for placement in range(placements):
                start = group_start + placement * ACTION_COUNT
                stop = start + ACTION_COUNT
                best = int(np.argmax(phi[start:stop] + online[start:stop]))
                values.append(phi[start + best] + target[start + best])
            absolute_targets[sample] = np.mean(values)

    np.clip(absolute_targets, 0.0, 1.0, out=absolute_targets)
    return absolute_targets - current_phi
