from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from .features import encode_states
from .game import (
    ACTION_COUNT,
    CELL_COUNT,
    SIDE,
    afterstates_batch,
    denominator,
    place_at_ranks,
    potential,
)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    scores: NDArray[np.int64]
    potentials: NDArray[np.float64]


def generate_cases(
    episodes: int,
    seed: int,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    rng = np.random.default_rng(seed)
    flavors = rng.integers(1, 4, size=(episodes, CELL_COUNT), dtype=np.uint8)
    ranks = np.empty((episodes, CELL_COUNT), dtype=np.uint8)
    for turn in range(CELL_COUNT):
        ranks[:, turn] = rng.integers(1, CELL_COUNT - turn + 1, size=episodes)
    return flavors, ranks


def _predict_residuals(
    model: torch.nn.Module | None,
    board_features: NDArray[np.float32],
    device: torch.device,
    inference_batch_size: int,
) -> NDArray[np.float32]:
    if model is None:
        return np.zeros((len(board_features), ACTION_COUNT), dtype=np.float32)
    predictions = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(board_features), inference_batch_size):
            stop = start + inference_batch_size
            boards = torch.from_numpy(board_features[start:stop]).to(device)
            predictions.append(model(boards).cpu().numpy())
    return np.concatenate(predictions).astype(np.float32, copy=False)


def evaluate_policy(
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
        if model is None:
            residuals = np.zeros((episodes, ACTION_COUNT), dtype=np.float32)
            candidate_potentials = np.asarray(
                [
                    potential(candidates[episode, action], score_denominators[episode])
                    for episode in range(episodes)
                    for action in range(ACTION_COUNT)
                ],
                dtype=np.float32,
            ).reshape(episodes, ACTION_COUNT)
        else:
            board_features, normalized_to_original = encode_states(boards, turn + 1, flavors)
            original_potentials = np.asarray(
                [
                    [
                        potential(candidates[episode, action], score_denominators[episode])
                        for action in range(ACTION_COUNT)
                    ]
                    for episode in range(episodes)
                ],
                dtype=np.float32,
            )
            normalized_residuals = _predict_residuals(
                model,
                board_features,
                device,
                inference_batch_size,
            )
            original_residuals = np.empty_like(normalized_residuals)
            np.put_along_axis(
                original_residuals,
                normalized_to_original,
                normalized_residuals,
                axis=1,
            )
            selected = np.argmax(original_residuals + original_potentials, axis=1)
            boards = candidates[np.arange(episodes), selected]
            continue
        values = residuals + candidate_potentials
        boards = candidates[np.arange(episodes), np.argmax(values, axis=1)]

    potentials = np.asarray(
        [potential(boards[index], score_denominators[index]) for index in range(episodes)],
        dtype=np.float64,
    )
    scores = np.floor(1_000_000 * potentials + 0.5).astype(np.int64)
    return EvaluationResult(scores, potentials)


def evaluate_random_policy(
    flavors: NDArray[np.uint8],
    ranks: NDArray[np.uint8],
    *,
    seed: int,
) -> EvaluationResult:
    rng = np.random.default_rng(seed)
    episodes = len(flavors)
    boards = np.zeros((episodes, SIDE, SIDE), dtype=np.uint8)
    score_denominators = np.asarray([denominator(row) for row in flavors])
    for turn in range(CELL_COUNT):
        boards = place_at_ranks(boards, ranks[:, turn], flavors[:, turn])
        if turn + 1 < CELL_COUNT:
            actions = rng.integers(0, ACTION_COUNT, size=episodes)
            candidates = afterstates_batch(boards)
            boards = candidates[np.arange(episodes), actions]
    potentials = np.asarray(
        [potential(boards[index], score_denominators[index]) for index in range(episodes)],
        dtype=np.float64,
    )
    scores = np.floor(1_000_000 * potentials + 0.5).astype(np.int64)
    return EvaluationResult(scores, potentials)
