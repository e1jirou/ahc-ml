from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

from .afterstate_features import encode_afterstates, encode_future_sequences
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
    policy_phi_coefficient: float = 1.0,
    future_mode: str = "none",
    future_ablation: str = "correct",
    future_ablation_seed: int = 0,
) -> EvaluationResult:
    if future_ablation not in {"correct", "off", "episode_shuffle", "order_shuffle"}:
        raise ValueError(
            "future_ablation must be correct, off, episode_shuffle, or order_shuffle"
        )
    if future_ablation != "correct" and future_mode != "full_late":
        raise ValueError("future ablation requires full_late mode")
    episodes = len(flavors)
    ablation_rng = np.random.default_rng(future_ablation_seed)
    episode_permutation = (
        ablation_rng.permutation(episodes)
        if future_ablation == "episode_shuffle"
        else None
    )
    order_priorities = (
        ablation_rng.random((episodes, CELL_COUNT))
        if future_ablation == "order_shuffle"
        else None
    )
    boards = np.zeros((episodes, SIDE, SIDE), dtype=np.uint8)
    score_denominators = np.asarray([denominator(row) for row in flavors])
    for turn in range(CELL_COUNT):
        boards = place_at_ranks(boards, ranks[:, turn], flavors[:, turn])
        if turn + 1 == CELL_COUNT:
            break
        candidates = afterstates_batch(boards)
        candidate_potentials = None
        if model is None or policy_phi_coefficient != 0.0:
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
            assert candidate_potentials is not None
            residuals = np.zeros_like(candidate_potentials)
        else:
            flat_features = encode_afterstates(
                candidates.reshape(episodes * ACTION_COUNT, SIDE, SIDE),
                np.tile(np.arange(ACTION_COUNT, dtype=np.int64), episodes),
                turn + 1,
                np.repeat(flavors, ACTION_COUNT, axis=0),
            )
            future_features = (
                encode_future_sequences(flavors, turn + 1)
                if future_mode != "none" and future_ablation != "off"
                else None
            )
            if future_features is not None and episode_permutation is not None:
                future_features = future_features[episode_permutation]
            if future_features is not None and order_priorities is not None:
                start_position = turn + 1
                remaining = future_features[:, :, start_position:]
                order = np.argsort(order_priorities[:, start_position:], axis=1)
                shuffled = np.take_along_axis(remaining, order[:, None, :], axis=2)
                future_features = future_features.copy()
                future_features[:, :, start_position:] = shuffled
            flat_future_features = (
                np.repeat(future_features, ACTION_COUNT, axis=0)
                if future_features is not None
                else None
            )
            residuals = np.empty(episodes * ACTION_COUNT, dtype=np.float32)
            model.eval()
            with torch.inference_mode():
                for start in range(0, len(flat_features), inference_batch_size):
                    stop = start + inference_batch_size
                    inputs = torch.from_numpy(flat_features[start:stop]).to(device)
                    future_inputs = None
                    if flat_future_features is not None:
                        future_inputs = torch.from_numpy(flat_future_features[start:stop]).to(
                            device=device, dtype=torch.float32
                        )
                    residuals[start:stop] = (
                        model(
                            inputs,
                            future_inputs,
                            use_future_correction=future_ablation != "off",
                        )
                        .cpu()
                        .numpy()
                    )
            residuals = residuals.reshape(episodes, ACTION_COUNT)
        if model is None:
            policy_values = candidate_potentials
        elif policy_phi_coefficient == 0.0:
            policy_values = residuals
        else:
            assert candidate_potentials is not None
            policy_values = policy_phi_coefficient * candidate_potentials + residuals
        selected = np.argmax(policy_values, axis=1)
        boards = candidates[np.arange(episodes), selected]

    potentials = np.asarray(
        [potential(boards[index], score_denominators[index]) for index in range(episodes)],
        dtype=np.float64,
    )
    scores = np.floor(1_000_000 * potentials + 0.5).astype(np.int64)
    return EvaluationResult(scores, potentials)
