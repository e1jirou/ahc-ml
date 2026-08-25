from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from ahc_ml.checkpoint import load_checkpoint
from ahc_ml.device import select_device

from .features import FEATURE_CHANNELS, encode_afterstates
from .game import (
    ACTION_COUNT,
    CELL_COUNT,
    SIDE,
    afterstates,
    denominator,
    place_at_rank,
    potential,
    tilt,
)
from .model import Ahc015PpoNet
from .ppo import PpoRollout, ppo_update


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def representative_boards(
    episodes: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    placed = CELL_COUNT // 2
    flavors = rng.integers(1, 4, size=(episodes, CELL_COUNT), dtype=np.uint8)
    boards = np.zeros((episodes, SIDE, SIDE), dtype=np.uint8)
    for episode in range(episodes):
        cells = rng.choice(CELL_COUNT, size=placed, replace=False)
        boards[episode].reshape(-1)[cells] = flavors[episode, :placed]
        boards[episode] = tilt(boards[episode], int(rng.integers(ACTION_COUNT)))
    ranks = rng.integers(1, CELL_COUNT - placed + 1, size=episodes)
    return boards, flavors, ranks


def benchmark_rollout_turn(
    model: Ahc015PpoNet,
    device: torch.device,
    episodes: int,
    inference_batch_size: int,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray]:
    turn = CELL_COUNT // 2
    boards, flavors, ranks = representative_boards(episodes, rng)
    score_denominators = np.asarray([denominator(row) for row in flavors])
    storage = np.empty(
        (CELL_COUNT - 1, episodes, ACTION_COUNT, FEATURE_CHANNELS, SIDE, SIDE),
        dtype=np.uint8,
    )

    synchronize(device)
    started = time.perf_counter()
    for episode in range(episodes):
        boards[episode] = place_at_rank(
            boards[episode], int(ranks[episode]), int(flavors[episode, turn])
        )
    candidates = np.stack([afterstates(board) for board in boards])
    features = encode_afterstates(
        candidates.reshape(episodes * ACTION_COUNT, SIDE, SIDE),
        np.tile(np.arange(ACTION_COUNT, dtype=np.uint8), episodes),
        np.full(episodes * ACTION_COUNT, turn + 1, dtype=np.uint8),
        np.repeat(flavors, ACTION_COUNT, axis=0),
    ).reshape(episodes, ACTION_COUNT, FEATURE_CHANNELS, SIDE, SIDE)
    candidate_potentials = np.empty((episodes, ACTION_COUNT), dtype=np.float32)
    for episode in range(episodes):
        for action in range(ACTION_COUNT):
            candidate_potentials[episode, action] = potential(
                candidates[episode, action], score_denominators[episode]
            )
    episode_batch_size = max(1, inference_batch_size // ACTION_COUNT)
    model.eval()
    with torch.inference_mode():
        for start in range(0, episodes, episode_batch_size):
            stop = min(start + episode_batch_size, episodes)
            inputs = torch.from_numpy(features[start:stop]).to(device)
            potentials = torch.from_numpy(candidate_potentials[start:stop]).to(device)
            logits, values = model(inputs, potentials, 12.0)
            _ = logits.cpu().numpy(), values.cpu().numpy()
    features *= CELL_COUNT
    np.rint(features, out=features)
    storage[0] = features
    storage[0, :, :, 14] = 0
    synchronize(device)
    return time.perf_counter() - started, storage


def synthetic_rollout(transitions: int, rng: np.random.Generator) -> PpoRollout:
    features = rng.integers(
        0,
        CELL_COUNT + 1,
        size=(transitions, ACTION_COUNT, FEATURE_CHANNELS, SIDE, SIDE),
        dtype=np.uint8,
    )
    features[:, :, 14] = 0
    potentials = rng.random((transitions, ACTION_COUNT), dtype=np.float32)
    return PpoRollout(
        features=features,
        candidate_potentials=potentials,
        actions=rng.integers(ACTION_COUNT, size=transitions, dtype=np.int64),
        old_log_probs=np.zeros(transitions, dtype=np.float32),
        old_values=np.zeros(transitions, dtype=np.float32),
        advantages=rng.normal(size=transitions).astype(np.float32),
        returns=rng.normal(size=transitions).astype(np.float32),
    )


def benchmark_updates(
    model: Ahc015PpoNet,
    device: torch.device,
    batch_size: int,
    batches: int,
    rng: np.random.Generator,
) -> float:
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    warmup = synthetic_rollout(batch_size, rng)
    ppo_update(
        model,
        optimizer,
        warmup,
        device,
        rng,
        epochs=1,
        batch_size=batch_size,
        clip_ratio=0.2,
        value_clip=0.2,
        value_coefficient=0.5,
        entropy_coefficient=0.01,
        gradient_clip_norm=1.0,
        logit_scale=12.0,
        target_kl=math.inf,
    )
    rollout = synthetic_rollout(batch_size * batches, rng)
    synchronize(device)
    started = time.perf_counter()
    ppo_update(
        model,
        optimizer,
        rollout,
        device,
        rng,
        epochs=1,
        batch_size=batch_size,
        clip_ratio=0.2,
        value_clip=0.2,
        value_coefficient=0.5,
        entropy_coefficient=0.01,
        gradient_clip_norm=1.0,
        logit_scale=12.0,
        target_kl=math.inf,
    )
    synchronize(device)
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description="Microbenchmark AHC015 PPO training")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/ahc015/ppo-20260824-230503/best-training.pt"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--episodes", type=int, default=4096)
    parser.add_argument("--inference-batch-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--update-batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=15026)
    args = parser.parse_args()

    device, device_info = select_device(args.device)
    rng = np.random.default_rng(args.seed)
    model = Ahc015PpoNet().to(device)
    load_checkpoint(args.checkpoint, model=model, map_location=device)

    rollout_turn_seconds, storage = benchmark_rollout_turn(
        model, device, args.episodes, args.inference_batch_size, rng
    )
    started = time.perf_counter()
    storage.fill(0)
    buffer_write_seconds = time.perf_counter() - started
    update_seconds = benchmark_updates(model, device, args.batch_size, args.update_batches, rng)

    transitions = args.episodes * (CELL_COUNT - 1)
    update_count = math.ceil(transitions / args.batch_size)
    estimated_rollout = rollout_turn_seconds * (CELL_COUNT - 1)
    estimated_optimization = update_seconds / args.update_batches * update_count
    report = {
        "device": device_info.to_dict(),
        "episodes": args.episodes,
        "rollout_turn_seconds": rollout_turn_seconds,
        "estimated_rollout_seconds": estimated_rollout,
        "buffer_gib": storage.nbytes / (1024**3),
        "buffer_full_write_seconds": buffer_write_seconds,
        "measured_update_batches": args.update_batches,
        "update_seconds": update_seconds,
        "estimated_updates": update_count,
        "estimated_optimization_seconds": estimated_optimization,
        "estimated_iteration_seconds": estimated_rollout + estimated_optimization,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
