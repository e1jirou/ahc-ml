from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from .features import (
    BOARD_CHANNELS,
    encode_states,
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
from .simulation import EvaluationResult


@dataclass(frozen=True, slots=True)
class PpoRollout:
    # Occupancy planes are binary, so uint8 is lossless and compact.
    board_features: NDArray[np.uint8]
    candidate_potentials: NDArray[np.float32]
    actions: NDArray[np.int64]
    old_log_probs: NDArray[np.float32]
    old_values: NDArray[np.float32]
    advantages: NDArray[np.float32]
    returns: NDArray[np.float32]

    def __len__(self) -> int:
        return len(self.actions)


@dataclass(frozen=True, slots=True)
class PpoRolloutStorage:
    """Turn-major rollout arrays, optionally backed by shared memory."""

    board_features: NDArray[np.uint8]
    candidate_potentials: NDArray[np.float32]
    actions: NDArray[np.int64]
    old_log_probs: NDArray[np.float32]
    old_values: NDArray[np.float32]
    advantages: NDArray[np.float32]
    returns: NDArray[np.float32]

    @classmethod
    def empty(cls, episodes: int) -> PpoRolloutStorage:
        steps = CELL_COUNT - 1
        return cls(
            board_features=np.empty((steps, episodes, BOARD_CHANNELS, SIDE, SIDE), dtype=np.uint8),
            candidate_potentials=np.empty((steps, episodes, ACTION_COUNT), dtype=np.float32),
            actions=np.empty((steps, episodes), dtype=np.int64),
            old_log_probs=np.empty((steps, episodes), dtype=np.float32),
            old_values=np.empty((steps, episodes), dtype=np.float32),
            advantages=np.empty((steps, episodes), dtype=np.float32),
            returns=np.empty((steps, episodes), dtype=np.float32),
        )

    def episode_slice(self, start: int, stop: int) -> PpoRolloutStorage:
        return PpoRolloutStorage(
            **{field: getattr(self, field)[:, start:stop] for field in self.__dataclass_fields__}
        )

    def as_rollout(self) -> PpoRollout:
        transitions = self.actions.size
        return PpoRollout(
            board_features=self.board_features.reshape(transitions, BOARD_CHANNELS, SIDE, SIDE),
            candidate_potentials=self.candidate_potentials.reshape(transitions, ACTION_COUNT),
            actions=self.actions.reshape(transitions),
            old_log_probs=self.old_log_probs.reshape(transitions),
            old_values=self.old_values.reshape(transitions),
            advantages=self.advantages.reshape(transitions),
            returns=self.returns.reshape(transitions),
        )


def generalized_advantages(
    rewards: NDArray[np.float32],
    values: NDArray[np.float32],
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    if rewards.shape != values.shape or rewards.ndim != 2:
        raise ValueError("rewards and values must have the same [episodes, steps] shape")
    advantages = np.zeros_like(rewards)
    accumulator = np.zeros(rewards.shape[0], dtype=np.float32)
    for step in range(rewards.shape[1] - 1, -1, -1):
        next_value = values[:, step + 1] if step + 1 < rewards.shape[1] else 0.0
        delta = rewards[:, step] + gamma * next_value - values[:, step]
        accumulator = delta + gamma * gae_lambda * accumulator
        advantages[:, step] = accumulator
    return advantages, advantages + values


def _sample_actions(
    probabilities: NDArray[np.float32], rng: np.random.Generator
) -> NDArray[np.int64]:
    uniforms = rng.random(len(probabilities))
    cumulative = np.cumsum(probabilities, axis=1)
    return np.minimum((uniforms[:, None] > cumulative).sum(axis=1), ACTION_COUNT - 1)


def collect_ppo_rollout(
    model: torch.nn.Module,
    device: torch.device,
    episodes: int,
    rng: np.random.Generator,
    *,
    gamma: float,
    gae_lambda: float,
    logit_scale: float,
    inference_batch_size: int,
    storage: PpoRolloutStorage | None = None,
) -> tuple[PpoRollout, EvaluationResult, dict[str, float]]:
    flavors = rng.integers(1, 4, size=(episodes, CELL_COUNT), dtype=np.uint8)
    ranks = np.empty((episodes, CELL_COUNT), dtype=np.uint8)
    for turn in range(CELL_COUNT):
        ranks[:, turn] = rng.integers(1, CELL_COUNT - turn + 1, size=episodes)
    boards = np.zeros((episodes, SIDE, SIDE), dtype=np.uint8)
    score_denominators = np.asarray([denominator(row) for row in flavors])

    rollout_storage = storage or PpoRolloutStorage.empty(episodes)
    if rollout_storage.actions.shape != (CELL_COUNT - 1, episodes):
        raise ValueError("rollout storage has an incompatible episode count")
    state_potential_storage = np.empty((CELL_COUNT - 1, episodes), dtype=np.float32)
    entropy_steps: list[float] = []

    model.eval()
    for turn in range(CELL_COUNT - 1):
        boards = place_at_ranks(boards, ranks[:, turn], flavors[:, turn])
        state_potentials = np.asarray(
            [potential(boards[i], score_denominators[i]) for i in range(episodes)],
            dtype=np.float32,
        )
        board_features, normalized_to_original = encode_states(boards, turn + 1, flavors)
        candidates = afterstates_batch(boards)
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
        candidate_potentials = np.take_along_axis(
            original_potentials, normalized_to_original, axis=1
        )
        probabilities = np.empty((episodes, ACTION_COUNT), dtype=np.float32)
        values_array = np.empty(episodes, dtype=np.float32)
        episode_batch_size = inference_batch_size
        with torch.inference_mode():
            for start in range(0, episodes, episode_batch_size):
                stop = min(start + episode_batch_size, episodes)
                board_tensor = torch.from_numpy(board_features[start:stop]).to(device)
                potential_tensor = torch.from_numpy(candidate_potentials[start:stop]).to(device)
                logits, values = model(board_tensor, potential_tensor, logit_scale)
                probabilities[start:stop] = torch.softmax(logits, dim=1).cpu().numpy()
                values_array[start:stop] = values.cpu().numpy()
        chosen = _sample_actions(probabilities, rng)
        chosen_probabilities = probabilities[np.arange(episodes), chosen]
        original_actions = normalized_to_original[np.arange(episodes), chosen]
        boards = candidates[np.arange(episodes), original_actions]

        rollout_storage.board_features[turn] = board_features
        rollout_storage.candidate_potentials[turn] = candidate_potentials
        rollout_storage.actions[turn] = chosen
        rollout_storage.old_log_probs[turn] = np.log(np.maximum(chosen_probabilities, 1e-12))
        rollout_storage.old_values[turn] = values_array
        state_potential_storage[turn] = state_potentials
        entropy_steps.append(float((-probabilities * np.log(probabilities + 1e-12)).sum(1).mean()))

    boards = place_at_ranks(boards, ranks[:, -1], flavors[:, -1])
    final_potentials = np.asarray(
        [potential(boards[i], score_denominators[i]) for i in range(episodes)],
        dtype=np.float32,
    )
    state_potentials_by_episode = state_potential_storage.T
    next_potentials = np.concatenate(
        (state_potentials_by_episode[:, 1:], final_potentials[:, None]), axis=1
    )
    rewards = next_potentials - state_potentials_by_episode
    values = rollout_storage.old_values.T
    advantages, returns = generalized_advantages(
        rewards, values, gamma=gamma, gae_lambda=gae_lambda
    )
    rollout_storage.advantages[:] = advantages.T
    rollout_storage.returns[:] = returns.T
    rollout = rollout_storage.as_rollout()
    scores = np.floor(1_000_000 * final_potentials + 0.5).astype(np.int64)
    result = EvaluationResult(scores, final_potentials.astype(np.float64))
    rollout_metrics = {
        "rollout/entropy": float(np.mean(entropy_steps)),
        "rollout/mean_reward": float(rewards.sum(axis=1).mean()),
        "rollout/mean_score": float(scores.mean()),
        "rollout/value_mean": float(values.mean()),
        "rollout/feature_buffer_gib": (rollout_storage.board_features.nbytes) / (1024**3),
    }
    return rollout, result, rollout_metrics


def ppo_update(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rollout: PpoRollout,
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
) -> dict[str, float]:
    if micro_batch_size is None:
        micro_batch_size = batch_size
    if micro_batch_size <= 0 or micro_batch_size > batch_size:
        raise ValueError("micro_batch_size must be in [1, batch_size]")
    advantages = rollout.advantages.copy()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    losses: list[float] = []
    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    kls: list[float] = []
    clip_fractions: list[float] = []
    gradient_norms: list[float] = []
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
                )
            }
            for micro_start in range(0, len(batch_indices), micro_batch_size):
                micro_indices = batch_indices[micro_start : micro_start + micro_batch_size]
                weight = len(micro_indices) / len(batch_indices)
                potentials = torch.from_numpy(rollout.candidate_potentials[micro_indices]).to(
                    device
                )
                board_features = torch.from_numpy(rollout.board_features[micro_indices]).to(
                    device=device, dtype=torch.float32
                )
                actions = torch.from_numpy(rollout.actions[micro_indices]).to(device)
                old_log_probs = torch.from_numpy(rollout.old_log_probs[micro_indices]).to(device)
                old_values = torch.from_numpy(rollout.old_values[micro_indices]).to(device)
                micro_advantages = torch.from_numpy(advantages[micro_indices]).to(device)
                returns = torch.from_numpy(rollout.returns[micro_indices]).to(device)

                logits, values = model(board_features, potentials, logit_scale)
                distribution = torch.distributions.Categorical(logits=logits)
                log_probs = distribution.log_prob(actions)
                log_ratio = log_probs - old_log_probs
                ratio = log_ratio.exp()
                unclipped = ratio * micro_advantages
                clipped = ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * micro_advantages
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                clipped_values = old_values + (values - old_values).clamp(-value_clip, value_clip)
                raw_value_loss = (values - returns).square()
                clipped_value_loss = (clipped_values - returns).square()
                value_loss = 0.5 * torch.maximum(raw_value_loss, clipped_value_loss).mean()
                entropy = distribution.entropy().mean()
                loss = policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy
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
                ):
                    batch_metrics[name] += value.detach() * weight

            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()

            losses.append(float(batch_metrics["loss"].cpu()))
            policy_losses.append(float(batch_metrics["policy_loss"].cpu()))
            value_losses.append(float(batch_metrics["value_loss"].cpu()))
            entropies.append(float(batch_metrics["entropy"].cpu()))
            kls.append(float(batch_metrics["kl"].cpu()))
            epoch_kls.append(kls[-1])
            clip_fractions.append(float(batch_metrics["clip_fraction"].cpu()))
            gradient_norms.append(float(gradient_norm.detach().cpu()))
            updates += 1
        if epoch_kls and np.mean(epoch_kls) > target_kl:
            stopped_early = True
            break

    return {
        "training/loss": float(np.mean(losses)),
        "training/policy_loss": float(np.mean(policy_losses)),
        "training/value_loss": float(np.mean(value_losses)),
        "training/entropy": float(np.mean(entropies)),
        "training/approximate_kl": float(np.mean(kls)),
        "training/clip_fraction": float(np.mean(clip_fractions)),
        "training/gradient_norm": float(np.mean(gradient_norms)),
        "training/updates_this_iteration": float(updates),
        "training/early_stop": float(stopped_early),
        "training/explained_variance": float(
            1.0 - np.var(rollout.returns - rollout.old_values) / (np.var(rollout.returns) + 1e-8)
        ),
    }
