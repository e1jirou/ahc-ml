from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

from .afterstate_features import encode_afterstates
from .game import (
    ACTION_COUNT,
    CELL_COUNT,
    SIDE,
    afterstates_batch,
    denominator,
    place_at_ranks,
    potential,
)
from .simulation import EvaluationResult


def evaluate_afterstate_policy(
    model: torch.nn.Module | None,
    device: torch.device,
    flavors: NDArray[np.uint8],
    ranks: NDArray[np.uint8],
    *,
    inference_batch_size: int,
) -> EvaluationResult:
    episodes = len(flavors)
    boards = np.zeros((episodes, SIDE, SIDE), dtype=np.uint8)
    score_denominators = np.asarray([denominator(row) for row in flavors])
    for turn in range(CELL_COUNT):
        boards = place_at_ranks(boards, ranks[:, turn], flavors[:, turn])
        if turn + 1 == CELL_COUNT:
            break
        candidates = afterstates_batch(boards)
        candidate_potentials = np.asarray(
            [
                [
                    potential(candidates[episode, action], score_denominators[episode])
                    for action in range(ACTION_COUNT)
                ]
                for episode in range(episodes)
            ],
            dtype=np.float32,
        )
        if model is None:
            residuals = np.zeros_like(candidate_potentials)
        else:
            flat_features = encode_afterstates(
                candidates.reshape(episodes * ACTION_COUNT, SIDE, SIDE),
                np.tile(np.arange(ACTION_COUNT, dtype=np.int64), episodes),
                turn + 1,
                np.repeat(flavors, ACTION_COUNT, axis=0),
            )
            residuals = np.empty(episodes * ACTION_COUNT, dtype=np.float32)
            model.eval()
            with torch.inference_mode():
                for start in range(0, len(flat_features), inference_batch_size):
                    stop = start + inference_batch_size
                    inputs = torch.from_numpy(flat_features[start:stop]).to(device)
                    residuals[start:stop] = model(inputs).cpu().numpy()
            residuals = residuals.reshape(episodes, ACTION_COUNT)
        selected = np.argmax(candidate_potentials + residuals, axis=1)
        boards = candidates[np.arange(episodes), selected]

    potentials = np.asarray(
        [potential(boards[index], score_denominators[index]) for index in range(episodes)],
        dtype=np.float64,
    )
    scores = np.floor(1_000_000 * potentials + 0.5).astype(np.int64)
    return EvaluationResult(scores, potentials)
