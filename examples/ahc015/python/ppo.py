from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from .features import (
    BOARD_CHANNELS,
    FUTURE_CHANNELS,
    FUTURE_LENGTH,
    POTENTIAL_CHANNEL,
    encode_afterstates,
)
from .game import ACTION_COUNT, CELL_COUNT, SIDE, afterstates, denominator, place_at_rank, potential
from .model import Ahc015PpoNet
from .simulation import EvaluationResult


@dataclass(frozen=True, slots=True)
class PpoRollout:
    # All board planes except the potential plane are exact multiples of
    # 1 / CELL_COUNT. The potential plane is reconstructed from the lossless
    # float32 candidate_potentials array before each update.  Keeping the large
    # on-policy buffer in this compact form makes 4096-episode rollouts fit in
    # memory without changing the model inputs.
    board_features: NDArray[np.uint8]
    future_features: NDArray[np.uint8]
    candidate_potentials: NDArray[np.float32]
    actions: NDArray[np.int64]
    old_log_probs: NDArray[np.float32]
    old_values: NDArray[np.float32]
    advantages: NDArray[np.float32]
    returns: NDArray[np.float32]

    def __len__(self) -> int:
        return len(self.actions)


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
    model: Ahc015PpoNet,
    device: torch.device,
    episodes: int,
    rng: np.random.Generator,
    *,
    gamma: float,
    gae_lambda: float,
    logit_scale: float,
    inference_batch_size: int,
) -> tuple[PpoRollout, EvaluationResult, dict[str, float]]:
    flavors = rng.integers(1, 4, size=(episodes, CELL_COUNT), dtype=np.uint8)
    ranks = np.empty((episodes, CELL_COUNT), dtype=np.uint8)
    for turn in range(CELL_COUNT):
        ranks[:, turn] = rng.integers(1, CELL_COUNT - turn + 1, size=episodes)
    boards = np.zeros((episodes, SIDE, SIDE), dtype=np.uint8)
    score_denominators = np.asarray([denominator(row) for row in flavors])

    board_storage = np.empty(
        (CELL_COUNT - 1, episodes, ACTION_COUNT, BOARD_CHANNELS, SIDE, SIDE),
        dtype=np.uint8,
    )
    future_storage = np.empty(
        (CELL_COUNT - 1, episodes, ACTION_COUNT, FUTURE_CHANNELS, FUTURE_LENGTH),
        dtype=np.uint8,
    )
    potential_steps: list[NDArray[np.float32]] = []
    action_steps: list[NDArray[np.int64]] = []
    log_prob_steps: list[NDArray[np.float32]] = []
    value_steps: list[NDArray[np.float32]] = []
    state_potential_steps: list[NDArray[np.float32]] = []
    entropy_steps: list[float] = []

    model.eval()
    for turn in range(CELL_COUNT - 1):
        for episode in range(episodes):
            boards[episode] = place_at_rank(
                boards[episode], int(ranks[episode, turn]), int(flavors[episode, turn])
            )
        state_potentials = np.asarray(
            [potential(boards[i], score_denominators[i]) for i in range(episodes)],
            dtype=np.float32,
        )
        candidates = np.stack([afterstates(board) for board in boards])
        flat_candidates = candidates.reshape(episodes * ACTION_COUNT, SIDE, SIDE)
        actions_for_features = np.tile(np.arange(ACTION_COUNT, dtype=np.uint8), episodes)
        placed = np.full(episodes * ACTION_COUNT, turn + 1, dtype=np.uint8)
        repeated_flavors = np.repeat(flavors, ACTION_COUNT, axis=0)
        board_features, future_features = encode_afterstates(
            flat_candidates, actions_for_features, placed, repeated_flavors
        )
        board_features = board_features.reshape(episodes, ACTION_COUNT, BOARD_CHANNELS, SIDE, SIDE)
        future_features = future_features.reshape(
            episodes, ACTION_COUNT, FUTURE_CHANNELS, FUTURE_LENGTH
        )
        candidate_potentials = np.empty((episodes, ACTION_COUNT), dtype=np.float32)
        for episode in range(episodes):
            for action in range(ACTION_COUNT):
                candidate_potentials[episode, action] = potential(
                    candidates[episode, action], score_denominators[episode]
                )
        probabilities = np.empty((episodes, ACTION_COUNT), dtype=np.float32)
        values_array = np.empty(episodes, dtype=np.float32)
        episode_batch_size = max(1, inference_batch_size // ACTION_COUNT)
        with torch.inference_mode():
            for start in range(0, episodes, episode_batch_size):
                stop = min(start + episode_batch_size, episodes)
                board_tensor = torch.from_numpy(board_features[start:stop]).to(device)
                future_tensor = torch.from_numpy(future_features[start:stop]).to(device)
                potential_tensor = torch.from_numpy(candidate_potentials[start:stop]).to(device)
                logits, values = model(board_tensor, future_tensor, potential_tensor, logit_scale)
                probabilities[start:stop] = torch.softmax(logits, dim=1).cpu().numpy()
                values_array[start:stop] = values.cpu().numpy()
        chosen = _sample_actions(probabilities, rng)
        chosen_probabilities = probabilities[np.arange(episodes), chosen]
        boards = candidates[np.arange(episodes), chosen]

        # The potential plane is not quantized: candidate_potentials already
        # stores the same value once per candidate, instead of 100 times.
        board_features *= CELL_COUNT
        np.rint(board_features, out=board_features)
        board_storage[turn] = board_features
        board_storage[turn, :, :, POTENTIAL_CHANNEL] = 0
        future_storage[turn] = future_features
        potential_steps.append(candidate_potentials)
        action_steps.append(chosen)
        log_prob_steps.append(np.log(np.maximum(chosen_probabilities, 1e-12)))
        value_steps.append(values_array)
        state_potential_steps.append(state_potentials)
        entropy_steps.append(float((-probabilities * np.log(probabilities + 1e-12)).sum(1).mean()))

    for episode in range(episodes):
        boards[episode] = place_at_rank(
            boards[episode], int(ranks[episode, -1]), int(flavors[episode, -1])
        )
    final_potentials = np.asarray(
        [potential(boards[i], score_denominators[i]) for i in range(episodes)],
        dtype=np.float32,
    )
    state_potentials = np.stack(state_potential_steps, axis=1)
    next_potentials = np.concatenate((state_potentials[:, 1:], final_potentials[:, None]), axis=1)
    rewards = next_potentials - state_potentials
    values = np.stack(value_steps, axis=1)
    advantages, returns = generalized_advantages(
        rewards, values, gamma=gamma, gae_lambda=gae_lambda
    )

    # PPO shuffles all transitions before every epoch, so a turn-major buffer is
    # equivalent to episode-major ordering and lets rollout collection write
    # each large feature block contiguously.
    def transition_major(steps: list[NDArray[np.generic]]) -> NDArray[np.generic]:
        return np.stack(steps, axis=0).reshape(episodes * (CELL_COUNT - 1), *steps[0].shape[1:])

    rollout = PpoRollout(
        board_features=board_storage.reshape(
            episodes * (CELL_COUNT - 1), ACTION_COUNT, BOARD_CHANNELS, SIDE, SIDE
        ),
        future_features=future_storage.reshape(
            episodes * (CELL_COUNT - 1), ACTION_COUNT, FUTURE_CHANNELS, FUTURE_LENGTH
        ),
        candidate_potentials=transition_major(potential_steps).astype(np.float32, copy=False),
        actions=transition_major(action_steps).astype(np.int64, copy=False),
        old_log_probs=transition_major(log_prob_steps).astype(np.float32, copy=False),
        old_values=values.T.reshape(-1),
        advantages=advantages.T.reshape(-1),
        returns=returns.T.reshape(-1),
    )
    scores = np.floor(1_000_000 * final_potentials + 0.5).astype(np.int64)
    result = EvaluationResult(scores, final_potentials.astype(np.float64))
    rollout_metrics = {
        "rollout/entropy": float(np.mean(entropy_steps)),
        "rollout/mean_reward": float(rewards.sum(axis=1).mean()),
        "rollout/mean_score": float(scores.mean()),
        "rollout/value_mean": float(values.mean()),
        "rollout/feature_buffer_gib": (board_storage.nbytes + future_storage.nbytes) / (1024**3),
    }
    return rollout, result, rollout_metrics


def ppo_update(
    model: Ahc015PpoNet,
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
) -> dict[str, float]:
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
            potentials = torch.from_numpy(rollout.candidate_potentials[batch_indices]).to(device)
            board_features = torch.from_numpy(rollout.board_features[batch_indices]).to(
                device=device, dtype=torch.float32
            )
            board_features.div_(CELL_COUNT)
            board_features[:, :, POTENTIAL_CHANNEL] = potentials[:, :, None, None]
            future_features = torch.from_numpy(rollout.future_features[batch_indices]).to(
                device=device, dtype=torch.float32
            )
            actions = torch.from_numpy(rollout.actions[batch_indices]).to(device)
            old_log_probs = torch.from_numpy(rollout.old_log_probs[batch_indices]).to(device)
            old_values = torch.from_numpy(rollout.old_values[batch_indices]).to(device)
            batch_advantages = torch.from_numpy(advantages[batch_indices]).to(device)
            returns = torch.from_numpy(rollout.returns[batch_indices]).to(device)

            logits, values = model(board_features, future_features, potentials, logit_scale)
            distribution = torch.distributions.Categorical(logits=logits)
            log_probs = distribution.log_prob(actions)
            log_ratio = log_probs - old_log_probs
            ratio = log_ratio.exp()
            unclipped = ratio * batch_advantages
            clipped = ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * batch_advantages
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            clipped_values = old_values + (values - old_values).clamp(-value_clip, value_clip)
            raw_value_loss = (values - returns).square()
            clipped_value_loss = (clipped_values - returns).square()
            value_loss = 0.5 * torch.maximum(raw_value_loss, clipped_value_loss).mean()
            entropy = distribution.entropy().mean()
            loss = policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()

            with torch.no_grad():
                approximate_kl = ((ratio - 1) - log_ratio).mean()
                clip_fraction = ((ratio - 1).abs() > clip_ratio).float().mean()
            losses.append(float(loss.detach().cpu()))
            policy_losses.append(float(policy_loss.detach().cpu()))
            value_losses.append(float(value_loss.detach().cpu()))
            entropies.append(float(entropy.detach().cpu()))
            kls.append(float(approximate_kl.cpu()))
            epoch_kls.append(kls[-1])
            clip_fractions.append(float(clip_fraction.cpu()))
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
