from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from .afterstate_features import (
    FUTURE_CHANNELS,
    FUTURE_LENGTH,
    encode_afterstates,
    encode_future_sequences,
)
from .game import (
    ACTION_COUNT,
    CELL_COUNT,
    SIDE,
    afterstates_batch,
    denominator,
    place_at_ranks,
    potential,
)
from .ppo import generalized_advantages
from .simulation import EvaluationResult

BOARD_CHANNELS = 4


@dataclass(frozen=True, slots=True)
class AfterstatePpoRollout:
    board_features: NDArray[np.uint8]
    future_features: NDArray[np.uint8] | None
    candidate_potentials: NDArray[np.float32] | None
    actions: NDArray[np.int64]
    old_log_probs: NDArray[np.float32]
    old_values: NDArray[np.float32]
    advantages: NDArray[np.float32]
    returns: NDArray[np.float32]

    def __len__(self) -> int:
        return len(self.actions)


@dataclass(frozen=True, slots=True)
class AfterstatePpoRolloutStorage:
    """Turn-major afterstate rollout arrays, optionally backed by shared memory."""

    board_features: NDArray[np.uint8]
    future_features: NDArray[np.uint8] | None
    candidate_potentials: NDArray[np.float32] | None
    actions: NDArray[np.int64]
    old_log_probs: NDArray[np.float32]
    old_values: NDArray[np.float32]
    advantages: NDArray[np.float32]
    returns: NDArray[np.float32]

    @classmethod
    def empty(
        cls,
        episodes: int,
        *,
        future_mode: str,
        policy_phi_coefficient: float,
    ) -> AfterstatePpoRolloutStorage:
        prefix = (CELL_COUNT - 1, episodes)
        return cls(
            board_features=np.empty(
                (*prefix, ACTION_COUNT, BOARD_CHANNELS, SIDE, SIDE), dtype=np.uint8
            ),
            future_features=(
                np.empty((*prefix, FUTURE_CHANNELS, FUTURE_LENGTH), dtype=np.uint8)
                if future_mode != "none"
                else None
            ),
            candidate_potentials=(
                np.empty((*prefix, ACTION_COUNT), dtype=np.float32)
                if policy_phi_coefficient != 0.0
                else None
            ),
            actions=np.empty(prefix, dtype=np.int64),
            old_log_probs=np.empty(prefix, dtype=np.float32),
            old_values=np.empty(prefix, dtype=np.float32),
            advantages=np.empty(prefix, dtype=np.float32),
            returns=np.empty(prefix, dtype=np.float32),
        )

    def as_rollout(self) -> AfterstatePpoRollout:
        transitions = self.actions.size
        return AfterstatePpoRollout(
            board_features=self.board_features.reshape(
                transitions, ACTION_COUNT, BOARD_CHANNELS, SIDE, SIDE
            ),
            future_features=(
                self.future_features.reshape(transitions, FUTURE_CHANNELS, FUTURE_LENGTH)
                if self.future_features is not None
                else None
            ),
            candidate_potentials=(
                self.candidate_potentials.reshape(transitions, ACTION_COUNT)
                if self.candidate_potentials is not None
                else None
            ),
            actions=self.actions.reshape(transitions),
            old_log_probs=self.old_log_probs.reshape(transitions),
            old_values=self.old_values.reshape(transitions),
            advantages=self.advantages.reshape(transitions),
            returns=self.returns.reshape(transitions),
        )


def _sample_actions(
    probabilities: NDArray[np.float32], rng: np.random.Generator
) -> NDArray[np.int64]:
    uniforms = rng.random(len(probabilities))
    cumulative = np.cumsum(probabilities, axis=1)
    return np.minimum((uniforms[:, None] > cumulative).sum(axis=1), ACTION_COUNT - 1)


def collect_afterstate_ppo_rollout(
    model: torch.nn.Module,
    device: torch.device,
    episodes: int,
    rng: np.random.Generator,
    *,
    gamma: float,
    gae_lambda: float,
    logit_scale: float,
    inference_batch_size: int,
    policy_phi_coefficient: float = 1.0,
    reward_mode: str = "potential_shaping",
    future_mode: str = "none",
    storage: AfterstatePpoRolloutStorage | None = None,
) -> tuple[AfterstatePpoRollout, EvaluationResult, dict[str, float]]:
    if reward_mode not in {"potential_shaping", "terminal"}:
        raise ValueError("reward_mode must be potential_shaping or terminal")
    flavors = rng.integers(1, 4, size=(episodes, CELL_COUNT), dtype=np.uint8)
    ranks = np.empty((episodes, CELL_COUNT), dtype=np.uint8)
    for turn in range(CELL_COUNT):
        ranks[:, turn] = rng.integers(1, CELL_COUNT - turn + 1, size=episodes)
    boards = np.zeros((episodes, SIDE, SIDE), dtype=np.uint8)
    score_denominators = np.asarray([denominator(row) for row in flavors])

    steps = CELL_COUNT - 1
    rollout_storage = storage or AfterstatePpoRolloutStorage.empty(
        episodes,
        future_mode=future_mode,
        policy_phi_coefficient=policy_phi_coefficient,
    )
    if rollout_storage.actions.shape != (steps, episodes):
        raise ValueError("afterstate rollout storage has an incompatible episode count")
    if (rollout_storage.future_features is not None) != (future_mode != "none"):
        raise ValueError("afterstate rollout storage has incompatible future features")
    if (rollout_storage.candidate_potentials is not None) != (
        policy_phi_coefficient != 0.0
    ):
        raise ValueError("afterstate rollout storage has incompatible candidate potentials")
    state_potential_storage = (
        np.empty((steps, episodes), dtype=np.float32)
        if reward_mode == "potential_shaping"
        else None
    )
    entropy_steps: list[float] = []

    model.eval()
    for turn in range(steps):
        boards = place_at_ranks(boards, ranks[:, turn], flavors[:, turn])
        if state_potential_storage is not None:
            state_potential_storage[turn] = np.asarray(
                [potential(boards[i], score_denominators[i]) for i in range(episodes)],
                dtype=np.float32,
            )
        candidates = afterstates_batch(boards)
        flat_features = encode_afterstates(
            candidates.reshape(episodes * ACTION_COUNT, SIDE, SIDE),
            np.tile(np.arange(ACTION_COUNT, dtype=np.int64), episodes),
            turn + 1,
            np.repeat(flavors, ACTION_COUNT, axis=0),
        )
        board_features = flat_features.reshape(episodes, ACTION_COUNT, BOARD_CHANNELS, SIDE, SIDE)
        future_features = (
            encode_future_sequences(flavors, turn + 1)
            if rollout_storage.future_features is not None
            else None
        )
        candidate_potentials = None
        if policy_phi_coefficient != 0.0:
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
        probabilities = np.empty((episodes, ACTION_COUNT), dtype=np.float32)
        values_array = np.empty(episodes, dtype=np.float32)
        episode_batch_size = max(1, inference_batch_size // ACTION_COUNT)
        # DataParallel worker threads do not reliably inherit inference-mode state.
        # no_grad provides the same memory saving here and is safe on CUDA replicas.
        with torch.no_grad():
            for start in range(0, episodes, episode_batch_size):
                stop = min(start + episode_batch_size, episodes)
                board_tensor = torch.from_numpy(board_features[start:stop]).to(
                    device=device, dtype=torch.float32
                )
                potential_tensor = (
                    torch.from_numpy(candidate_potentials[start:stop]).to(device)
                    if candidate_potentials is not None
                    else None
                )
                logits, values = model(
                    board_tensor,
                    potential_tensor,
                    logit_scale,
                    policy_phi_coefficient,
                    future_inputs=(
                        torch.from_numpy(future_features[start:stop]).to(
                            device=device, dtype=torch.float32
                        )
                        if future_features is not None
                        else None
                    ),
                )
                probabilities[start:stop] = torch.softmax(logits, dim=1).cpu().numpy()
                values_array[start:stop] = values.cpu().numpy()
        chosen = _sample_actions(probabilities, rng)
        chosen_probabilities = probabilities[np.arange(episodes), chosen]
        boards = candidates[np.arange(episodes), chosen]

        rollout_storage.board_features[turn] = board_features
        if rollout_storage.future_features is not None:
            assert future_features is not None
            rollout_storage.future_features[turn] = future_features
        if rollout_storage.candidate_potentials is not None:
            assert candidate_potentials is not None
            rollout_storage.candidate_potentials[turn] = candidate_potentials
        rollout_storage.actions[turn] = chosen
        rollout_storage.old_log_probs[turn] = np.log(
            np.maximum(chosen_probabilities, 1e-12)
        )
        rollout_storage.old_values[turn] = values_array
        entropy_steps.append(float((-probabilities * np.log(probabilities + 1e-12)).sum(1).mean()))

    boards = place_at_ranks(boards, ranks[:, -1], flavors[:, -1])
    final_potentials = np.asarray(
        [potential(boards[i], score_denominators[i]) for i in range(episodes)],
        dtype=np.float32,
    )
    if state_potential_storage is None:
        rewards = np.zeros((episodes, steps), dtype=np.float32)
        rewards[:, -1] = final_potentials
    else:
        state_potentials = state_potential_storage.T
        next_potentials = np.concatenate(
            (state_potentials[:, 1:], final_potentials[:, None]), axis=1
        )
        rewards = next_potentials - state_potentials
    values = rollout_storage.old_values.T
    advantages, returns = generalized_advantages(
        rewards, values, gamma=gamma, gae_lambda=gae_lambda
    )
    rollout_storage.advantages[:] = advantages.T
    rollout_storage.returns[:] = returns.T
    rollout = rollout_storage.as_rollout()
    scores = np.floor(1_000_000 * final_potentials + 0.5).astype(np.int64)
    result = EvaluationResult(scores, final_potentials.astype(np.float64))
    return (
        rollout,
        result,
        {
            "rollout/entropy": float(np.mean(entropy_steps)),
            "rollout/mean_reward": float(rewards.sum(axis=1).mean()),
            "rollout/mean_score": float(scores.mean()),
            "rollout/value_mean": float(values.mean()),
            "rollout/feature_buffer_gib": (
                rollout_storage.board_features.nbytes
                + (
                    rollout_storage.future_features.nbytes
                    if rollout_storage.future_features is not None
                    else 0
                )
            )
            / (1024**3),
            "rollout/policy_phi_coefficient": policy_phi_coefficient,
            "rollout/state_phi_evaluations": float(
                steps * episodes if state_potential_storage is not None else 0
            ),
            "rollout/candidate_phi_evaluations": float(
                steps * episodes * ACTION_COUNT
                if rollout_storage.candidate_potentials is not None
                else 0
            ),
            "rollout/final_phi_evaluations": float(episodes),
        },
    )


def update_afterstate_ppo(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rollout: AfterstatePpoRollout,
    device: torch.device,
    rng: np.random.Generator,
    *,
    epochs: int,
    batch_size: int,
    clip_ratio: float,
    value_clip: float,
    value_coefficient: float,
    entropy_coefficient: float,
    gradient_clip_norm: float,
    logit_scale: float,
    target_kl: float,
    micro_batch_size: int | None = None,
    policy_phi_coefficient: float = 1.0,
    teacher_actor: torch.nn.Module | None = None,
    distillation_coefficient: float = 0.0,
    advantage_mean: float | None = None,
    advantage_std: float | None = None,
) -> dict[str, float]:
    if micro_batch_size is None:
        micro_batch_size = batch_size
    if micro_batch_size <= 0 or micro_batch_size > batch_size:
        raise ValueError("micro_batch_size must be in [1, batch_size]")
    if distillation_coefficient < 0:
        raise ValueError("distillation_coefficient must be non-negative")
    if distillation_coefficient > 0 and teacher_actor is None:
        raise ValueError("teacher_actor is required when distillation is enabled")
    advantages = rollout.advantages.copy()
    normalization_mean = advantages.mean() if advantage_mean is None else advantage_mean
    normalization_std = advantages.std() if advantage_std is None else advantage_std
    advantages = (advantages - normalization_mean) / (normalization_std + 1e-8)
    metric_rows: dict[str, list[float]] = {
        name: []
        for name in (
            "loss",
            "policy_loss",
            "value_loss",
            "entropy",
            "kl",
            "clip_fraction",
            "gradient_norm",
            "distillation_kl",
        )
    }
    updates = 0
    stopped_early = False
    model.train()
    for _ in range(epochs):
        indices = rng.permutation(len(rollout))
        epoch_kls: list[float] = []
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            batch_metrics = {
                name: torch.zeros((), device=device)
                for name in (
                    "loss",
                    "policy_loss",
                    "value_loss",
                    "entropy",
                    "kl",
                    "clip_fraction",
                    "distillation_kl",
                )
            }
            for micro_start in range(0, len(batch_indices), micro_batch_size):
                micro_indices = batch_indices[micro_start : micro_start + micro_batch_size]
                weight = len(micro_indices) / len(batch_indices)
                boards = torch.from_numpy(rollout.board_features[micro_indices]).to(
                    device=device, dtype=torch.float32
                )
                potentials = (
                    torch.from_numpy(rollout.candidate_potentials[micro_indices]).to(device)
                    if rollout.candidate_potentials is not None
                    else None
                )
                futures = (
                    torch.from_numpy(rollout.future_features[micro_indices]).to(
                        device=device, dtype=torch.float32
                    )
                    if rollout.future_features is not None
                    else None
                )
                actions = torch.from_numpy(rollout.actions[micro_indices]).to(device)
                old_log_probs = torch.from_numpy(rollout.old_log_probs[micro_indices]).to(device)
                old_values = torch.from_numpy(rollout.old_values[micro_indices]).to(device)
                micro_advantages = torch.from_numpy(advantages[micro_indices]).to(device)
                returns = torch.from_numpy(rollout.returns[micro_indices]).to(device)

                logits, values = model(
                    boards,
                    potentials,
                    logit_scale,
                    policy_phi_coefficient,
                    future_inputs=futures,
                )
                distribution = torch.distributions.Categorical(logits=logits)
                log_probs = distribution.log_prob(actions)
                log_ratio = log_probs - old_log_probs
                ratio = log_ratio.exp()
                policy_loss = -torch.minimum(
                    ratio * micro_advantages,
                    ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * micro_advantages,
                ).mean()
                clipped_values = old_values + (values - old_values).clamp(-value_clip, value_clip)
                value_loss = (
                    0.5
                    * torch.maximum(
                        (values - returns).square(), (clipped_values - returns).square()
                    ).mean()
                )
                entropy = distribution.entropy().mean()
                distillation_kl = torch.zeros((), device=device)
                if teacher_actor is not None and distillation_coefficient > 0:
                    batch_length, action_count = boards.shape[:2]
                    with torch.no_grad():
                        teacher_values = teacher_actor(boards.flatten(0, 1)).reshape(
                            batch_length, action_count
                        )
                        teacher_log_probs = torch.log_softmax(logit_scale * teacher_values, dim=1)
                        teacher_probs = teacher_log_probs.exp()
                    student_log_probs = torch.log_softmax(logits, dim=1)
                    distillation_kl = (
                        (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=1).mean()
                    )
                loss = (
                    policy_loss
                    + value_coefficient * value_loss
                    - entropy_coefficient * entropy
                    + distillation_coefficient * distillation_kl
                )
                (loss * weight).backward()
                with torch.no_grad():
                    approximate_kl = ((ratio - 1) - log_ratio).mean()
                    clip_fraction = ((ratio - 1).abs() > clip_ratio).float().mean()
                for name, value in (
                    ("loss", loss),
                    ("policy_loss", policy_loss),
                    ("value_loss", value_loss),
                    ("entropy", entropy),
                    ("kl", approximate_kl),
                    ("clip_fraction", clip_fraction),
                    ("distillation_kl", distillation_kl),
                ):
                    batch_metrics[name] += value.detach() * weight

            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            for name in batch_metrics:
                metric_rows[name].append(float(batch_metrics[name].cpu()))
            metric_rows["gradient_norm"].append(float(gradient_norm.detach().cpu()))
            epoch_kls.append(metric_rows["kl"][-1])
            updates += 1
        if epoch_kls and np.mean(epoch_kls) > target_kl:
            stopped_early = True
            break

    return {
        "training/loss": float(np.mean(metric_rows["loss"])),
        "training/policy_loss": float(np.mean(metric_rows["policy_loss"])),
        "training/value_loss": float(np.mean(metric_rows["value_loss"])),
        "training/entropy": float(np.mean(metric_rows["entropy"])),
        "training/approximate_kl": float(np.mean(metric_rows["kl"])),
        "training/clip_fraction": float(np.mean(metric_rows["clip_fraction"])),
        "training/gradient_norm": float(np.mean(metric_rows["gradient_norm"])),
        "training/distillation_kl": float(np.mean(metric_rows["distillation_kl"])),
        "training/distillation_coefficient": distillation_coefficient,
        "training/updates_this_iteration": float(updates),
        "training/early_stop": float(stopped_early),
        "training/explained_variance": float(
            1.0 - np.var(rollout.returns - rollout.old_values) / (np.var(rollout.returns) + 1e-8)
        ),
    }
