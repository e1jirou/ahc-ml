from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import queue
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ahc_ml.checkpoint import load_checkpoint
from ahc_ml.device import select_device

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
    tilt,
)
from .model import (
    TEACHER_CHANNELS,
    TEACHER_RESIDUAL_BLOCKS,
    Ahc015PpoNet,
    dimensions_from_state_dict,
)
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
) -> tuple[float, tuple[np.ndarray, ...]]:
    turn = CELL_COUNT // 2
    boards, flavors, ranks = representative_boards(episodes, rng)
    score_denominators = np.asarray([denominator(row) for row in flavors])
    board_storage = np.empty(
        (CELL_COUNT - 1, episodes, BOARD_CHANNELS, SIDE, SIDE),
        dtype=np.uint8,
    )

    synchronize(device)
    started = time.perf_counter()
    boards = place_at_ranks(boards, ranks, flavors[:, turn])
    candidates = afterstates_batch(boards)
    board_features, normalized_to_original = encode_states(boards, turn + 1, flavors)
    original_potentials = np.asarray(
        [
            [potential(candidate, int(denominator_value)) for candidate in episode_candidates]
            for episode_candidates, denominator_value in zip(
                candidates, score_denominators, strict=True
            )
        ],
        dtype=np.float32,
    )
    candidate_potentials = np.take_along_axis(original_potentials, normalized_to_original, axis=1)
    episode_batch_size = inference_batch_size
    model.eval()
    with torch.inference_mode():
        for start in range(0, episodes, episode_batch_size):
            stop = min(start + episode_batch_size, episodes)
            boards_input = torch.from_numpy(board_features[start:stop]).to(device)
            potentials = torch.from_numpy(candidate_potentials[start:stop]).to(device)
            logits, values = model(boards_input, potentials, 12.0)
            _ = logits.cpu().numpy(), values.cpu().numpy()
    board_storage[0] = board_features
    synchronize(device)
    return time.perf_counter() - started, (board_storage,)


def _parallel_rollout_worker(
    worker: int,
    checkpoint_path: str | None,
    channels: int,
    residual_blocks: int,
    episodes: int,
    inference_batch_size: int,
    seed: int,
    start_event: Any,
    result_queue: Any,
) -> None:
    try:
        # Each process owns one GPU and one CPU-side episode shard. Restricting
        # PyTorch's CPU pool avoids oversubscribing Kaggle's relatively small
        # CPU allocation; NumPy feature construction remains inside this shard.
        torch.set_num_threads(1)
        device = torch.device(f"cuda:{worker}")
        torch.cuda.set_device(device)
        rng = np.random.default_rng(seed)
        model = Ahc015PpoNet(channels, residual_blocks).to(device)
        if checkpoint_path is not None:
            load_checkpoint(Path(checkpoint_path), model=model, map_location=device)

        # Initialize CUDA/cuBLAS before the synchronized measurement.
        with torch.inference_mode():
            warmup_boards = torch.zeros((1, BOARD_CHANNELS, SIDE, SIDE), device=device)
            warmup_potentials = torch.zeros((1, ACTION_COUNT), device=device)
            model(warmup_boards, warmup_potentials, 12.0)
        synchronize(device)
        result_queue.put(("ready", worker, None))
        if not start_event.wait(timeout=120):
            raise TimeoutError("parallel rollout start timed out")

        seconds, _ = benchmark_rollout_turn(
            model,
            device,
            episodes,
            inference_batch_size,
            rng,
        )
        result_queue.put(("result", worker, seconds))
    except BaseException as error:
        result_queue.put(("error", worker, repr(error)))


def benchmark_parallel_rollout_turn(
    checkpoint: Path | None,
    channels: int,
    residual_blocks: int,
    episodes: int,
    inference_batch_size: int,
    seed: int,
    workers: int,
) -> float:
    if workers < 2:
        raise ValueError("parallel rollout requires at least two workers")
    if torch.cuda.device_count() < workers:
        raise RuntimeError(f"parallel rollout requires {workers} CUDA devices")

    shard_sizes = [episodes // workers] * workers
    for worker in range(episodes % workers):
        shard_sizes[worker] += 1
    context = mp.get_context("spawn")
    start_event = context.Event()
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_parallel_rollout_worker,
            args=(
                worker,
                str(checkpoint) if checkpoint is not None else None,
                channels,
                residual_blocks,
                shard_sizes[worker],
                inference_batch_size,
                seed + worker,
                start_event,
                result_queue,
            ),
        )
        for worker in range(workers)
    ]
    for process in processes:
        process.start()

    try:
        ready = set()
        while len(ready) < workers:
            kind, worker, payload = result_queue.get(timeout=180)
            if kind == "error":
                raise RuntimeError(f"rollout worker {worker} failed: {payload}")
            if kind == "ready":
                ready.add(worker)
        start_event.set()
        durations: dict[int, float] = {}
        while len(durations) < workers:
            kind, worker, payload = result_queue.get(timeout=300)
            if kind == "error":
                raise RuntimeError(f"rollout worker {worker} failed: {payload}")
            if kind == "result":
                durations[worker] = float(payload)
    except queue.Empty as error:
        raise TimeoutError("parallel rollout worker timed out") from error
    finally:
        start_event.set()
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join()
    return max(durations.values())


def synthetic_rollout(transitions: int, rng: np.random.Generator) -> PpoRollout:
    board_features = rng.integers(
        0,
        2,
        size=(transitions, BOARD_CHANNELS, SIDE, SIDE),
        dtype=np.uint8,
    )
    potentials = rng.random((transitions, ACTION_COUNT), dtype=np.float32)
    return PpoRollout(
        board_features=board_features,
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
    micro_batch_size: int,
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
        micro_batch_size=micro_batch_size,
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
        micro_batch_size=micro_batch_size,
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
        help="optional training checkpoint; its model dimensions are detected automatically",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--episodes", type=int, default=4096)
    parser.add_argument("--inference-batch-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--micro-batch-size", type=int, default=128)
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument(
        "--rollout-processes",
        type=int,
        choices=(1, 2),
        default=1,
        help="independent CPU/GPU rollout pipelines (benchmark only)",
    )
    parser.add_argument("--update-batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=15026)
    args = parser.parse_args()

    device, device_info = select_device(args.device)
    rng = np.random.default_rng(args.seed)
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        channels, residual_blocks = dimensions_from_state_dict(
            checkpoint["model_state_dict"], prefix="actor."
        )
    else:
        channels, residual_blocks = TEACHER_CHANNELS, TEACHER_RESIDUAL_BLOCKS
    model = Ahc015PpoNet(channels, residual_blocks).to(device)
    if args.checkpoint is not None:
        load_checkpoint(args.checkpoint, model=model, map_location=device)
    execution_model: torch.nn.Module = model
    if args.data_parallel:
        if device.type != "cuda" or torch.cuda.device_count() < 2:
            raise RuntimeError("data parallel benchmark requires at least two CUDA devices")
        execution_model = torch.nn.DataParallel(model)

    if args.rollout_processes == 1:
        rollout_turn_seconds, storage = benchmark_rollout_turn(
            execution_model, device, args.episodes, args.inference_batch_size, rng
        )
        started = time.perf_counter()
        for array in storage:
            array.fill(0)
        buffer_write_seconds = time.perf_counter() - started
        buffer_bytes = sum(array.nbytes for array in storage)
    else:
        rollout_turn_seconds = benchmark_parallel_rollout_turn(
            args.checkpoint,
            channels,
            residual_blocks,
            args.episodes,
            args.inference_batch_size,
            args.seed,
            args.rollout_processes,
        )
        # Both workers together retain the same number of episode features as
        # the single-process collector; no large arrays cross process bounds.
        buffer_bytes = (CELL_COUNT - 1) * args.episodes * BOARD_CHANNELS * SIDE * SIDE
        buffer_write_seconds = math.nan
    update_seconds = benchmark_updates(
        execution_model,
        device,
        args.batch_size,
        args.micro_batch_size,
        args.update_batches,
        rng,
    )

    transitions = args.episodes * (CELL_COUNT - 1)
    update_count = math.ceil(transitions / args.batch_size)
    estimated_rollout = rollout_turn_seconds * (CELL_COUNT - 1)
    estimated_optimization = update_seconds / args.update_batches * update_count
    report = {
        "device": device_info.to_dict(),
        "episodes": args.episodes,
        "rollout_turn_seconds": rollout_turn_seconds,
        "estimated_rollout_seconds": estimated_rollout,
        "buffer_gib": buffer_bytes / (1024**3),
        "buffer_full_write_seconds": buffer_write_seconds,
        "measured_update_batches": args.update_batches,
        "micro_batch_size": args.micro_batch_size,
        "data_parallel": args.data_parallel,
        "rollout_processes": args.rollout_processes,
        "update_seconds": update_seconds,
        "estimated_updates": update_count,
        "estimated_optimization_seconds": estimated_optimization,
        "estimated_iteration_seconds": estimated_rollout + estimated_optimization,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
