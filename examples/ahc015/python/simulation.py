from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from .features import encode_afterstates
from .game import (
    ACTION_COUNT,
    CELL_COUNT,
    SIDE,
    afterstates,
    denominator,
    place_at_rank,
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
    future_features: NDArray[np.float32],
    device: torch.device,
    inference_batch_size: int,
) -> NDArray[np.float32]:
    if model is None:
        return np.zeros(len(board_features), dtype=np.float32)
    predictions = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(board_features), inference_batch_size):
            stop = start + inference_batch_size
            boards = torch.from_numpy(board_features[start:stop]).to(device)
            futures = torch.from_numpy(future_features[start:stop]).to(device)
            predictions.append(model(boards, futures).cpu().numpy())
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
    for turn in range(CELL_COUNT):
        for episode in range(episodes):
            boards[episode] = place_at_rank(
                boards[episode], int(ranks[episode, turn]), int(flavors[episode, turn])
            )
        if turn + 1 == CELL_COUNT:
            break
        candidates = np.stack([afterstates(board) for board in boards])
        flat = candidates.reshape(episodes * ACTION_COUNT, SIDE, SIDE)
        if model is None:
            residuals = np.zeros((episodes, ACTION_COUNT), dtype=np.float32)
        else:
            actions = np.tile(np.arange(ACTION_COUNT, dtype=np.uint8), episodes)
            placed = np.full(episodes * ACTION_COUNT, turn + 1, dtype=np.uint8)
            repeated_flavors = np.repeat(flavors, ACTION_COUNT, axis=0)
            board_features, future_features = encode_afterstates(
                flat, actions, placed, repeated_flavors
            )
            residuals = _predict_residuals(
                model,
                board_features,
                future_features,
                device,
                inference_batch_size,
            ).reshape(episodes, ACTION_COUNT)
        values = residuals
        for episode in range(episodes):
            score_denominator = denominator(flavors[episode])
            for action in range(ACTION_COUNT):
                values[episode, action] += potential(candidates[episode, action], score_denominator)
        selected = np.argmax(values, axis=1)
        boards = candidates[np.arange(episodes), selected]

    potentials = np.asarray(
        [potential(boards[index], denominator(flavors[index])) for index in range(episodes)],
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
    for turn in range(CELL_COUNT):
        for episode in range(episodes):
            boards[episode] = place_at_rank(
                boards[episode], int(ranks[episode, turn]), int(flavors[episode, turn])
            )
        if turn + 1 < CELL_COUNT:
            actions = rng.integers(0, ACTION_COUNT, size=episodes)
            candidates = np.stack([afterstates(board) for board in boards])
            boards = candidates[np.arange(episodes), actions]
    potentials = np.asarray(
        [potential(boards[index], denominator(flavors[index])) for index in range(episodes)],
        dtype=np.float64,
    )
    scores = np.floor(1_000_000 * potentials + 0.5).astype(np.int64)
    return EvaluationResult(scores, potentials)
