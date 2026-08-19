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
from .replay import ReplayBatch


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
    features: NDArray[np.float32],
    device: torch.device,
    inference_batch_size: int,
) -> NDArray[np.float32]:
    if model is None:
        return np.zeros(len(features), dtype=np.float32)
    predictions = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), inference_batch_size):
            inputs = torch.from_numpy(features[start : start + inference_batch_size]).to(device)
            predictions.append(model(inputs).cpu().numpy())
    return np.concatenate(predictions).astype(np.float32, copy=False)


def collect_rollouts(
    model: torch.nn.Module,
    device: torch.device,
    episodes: int,
    rng: np.random.Generator,
    *,
    epsilon: float,
    inference_batch_size: int,
) -> tuple[ReplayBatch, EvaluationResult]:
    flavors = rng.integers(1, 4, size=(episodes, CELL_COUNT), dtype=np.uint8)
    ranks = np.empty((episodes, CELL_COUNT), dtype=np.uint8)
    for turn in range(CELL_COUNT):
        ranks[:, turn] = rng.integers(1, CELL_COUNT - turn + 1, size=episodes)
    boards = np.zeros((episodes, SIDE, SIDE), dtype=np.uint8)

    stored_boards: list[NDArray[np.uint8]] = []
    stored_actions: list[NDArray[np.uint8]] = []
    stored_placed: list[NDArray[np.uint8]] = []
    stored_flavors: list[NDArray[np.uint8]] = []
    selected_offsets: list[NDArray[np.int64]] = []
    selected_potentials: list[NDArray[np.float32]] = []

    for turn in range(CELL_COUNT - 1):
        placed = turn + 1
        for episode in range(episodes):
            boards[episode] = place_at_rank(
                boards[episode], int(ranks[episode, turn]), int(flavors[episode, turn])
            )
        candidates = np.stack([afterstates(board) for board in boards])
        flat_candidates = candidates.reshape(episodes * ACTION_COUNT, SIDE, SIDE)
        actions = np.tile(np.arange(ACTION_COUNT, dtype=np.uint8), episodes)
        turns = np.full(episodes * ACTION_COUNT, placed, dtype=np.uint8)
        repeated_flavors = np.repeat(flavors, ACTION_COUNT, axis=0)
        features = encode_afterstates(flat_candidates, actions, turns, repeated_flavors)
        residuals = _predict_residuals(model, features, device, inference_batch_size).reshape(
            episodes, ACTION_COUNT
        )
        candidate_potentials = np.empty((episodes, ACTION_COUNT), dtype=np.float32)
        for episode in range(episodes):
            score_denominator = denominator(flavors[episode])
            for action in range(ACTION_COUNT):
                candidate_potentials[episode, action] = potential(
                    candidates[episode, action], score_denominator
                )
        values = candidate_potentials + residuals
        selected = np.argmax(values, axis=1)
        explore = rng.random(episodes) < epsilon
        selected[explore] = rng.integers(0, ACTION_COUNT, size=int(explore.sum()))
        boards = candidates[np.arange(episodes), selected]

        stored_boards.append(flat_candidates.reshape(-1, CELL_COUNT).copy())
        stored_actions.append(actions)
        stored_placed.append(turns)
        stored_flavors.append(repeated_flavors.copy())
        selected_offsets.append(selected.astype(np.int64) + np.arange(episodes) * ACTION_COUNT)
        selected_potentials.append(candidate_potentials[np.arange(episodes), selected])

    for episode in range(episodes):
        boards[episode] = place_at_rank(
            boards[episode], int(ranks[episode, -1]), int(flavors[episode, -1])
        )
    final_potentials = np.asarray(
        [potential(boards[index], denominator(flavors[index])) for index in range(episodes)],
        dtype=np.float32,
    )

    mc_targets: list[NDArray[np.float32]] = []
    for offsets, selected_phi in zip(selected_offsets, selected_potentials, strict=True):
        targets = np.full(episodes * ACTION_COUNT, np.nan, dtype=np.float32)
        targets[offsets] = final_potentials - selected_phi
        mc_targets.append(targets)

    replay = ReplayBatch(
        np.concatenate(stored_boards),
        np.concatenate(stored_actions),
        np.concatenate(stored_placed),
        np.concatenate(stored_flavors),
        np.concatenate(mc_targets),
    )
    scores = np.floor(1_000_000 * final_potentials + 0.5).astype(np.int64)
    return replay, EvaluationResult(scores, final_potentials.astype(np.float64))


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
            features = encode_afterstates(flat, actions, placed, repeated_flavors)
            residuals = _predict_residuals(model, features, device, inference_batch_size).reshape(
                episodes, ACTION_COUNT
            )
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
