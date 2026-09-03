from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import time
from pathlib import Path

import torch

from .afterstate_model import AfterstatePpoNet
from .afterstate_parallel_runtime import ParallelAfterstateRuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark future-free afterstate PPO DDP microbatch sizes"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--micro-batch-sizes", type=int, nargs="+", default=(256, 512, 1024)
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=15043)
    return parser.parse_args()


def reset_training_state(
    model: AfterstatePpoNet,
    model_state: dict[str, torch.Tensor],
    optimizer_state: dict[str, object],
) -> torch.optim.Optimizer:
    model.load_state_dict(model_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    optimizer.load_state_dict(copy.deepcopy(optimizer_state))
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = 3e-4
    return optimizer


def main() -> None:
    args = parse_args()
    if torch.cuda.device_count() != 2:
        raise RuntimeError("benchmark requires exactly two CUDA GPUs")
    if args.episodes <= 0 or args.repeats <= 0:
        raise ValueError("episodes and repeats must be positive")
    if args.episodes % 2:
        raise ValueError("episodes must be divisible by two")
    for micro_batch_size in args.micro_batch_sizes:
        if not 0 < micro_batch_size <= args.batch_size:
            raise ValueError("microbatch sizes must be in [1, batch_size]")
        if micro_batch_size % 2:
            raise ValueError("microbatch sizes must be divisible by two DDP workers")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_config = checkpoint["config"]
    model_config = checkpoint_config["model"]
    if model_config["input_mode"] != "afterstate" or model_config["future_mode"] != "none":
        raise ValueError("checkpoint must be a future-free afterstate model")
    channels = int(model_config["channels"])
    residual_blocks = int(model_config["residual_blocks"])
    model = AfterstatePpoNet(channels, residual_blocks, future_mode="none")
    initial_model_state = copy.deepcopy(checkpoint["model_state_dict"])
    initial_optimizer_state = copy.deepcopy(checkpoint["optimizer_state_dict"])
    model.load_state_dict(initial_model_state)

    runtime = ParallelAfterstateRuntime(
        workers=2,
        episodes=args.episodes,
        channels=channels,
        residual_blocks=residual_blocks,
        seed=args.seed,
    )
    try:
        runtime.collect(
            model,
            gamma=1.0,
            gae_lambda=0.95,
            logit_scale=12.0,
            inference_batch_size=4096,
            policy_phi_coefficient=0.0,
            reward_mode="potential_shaping",
            future_mode="none",
        )
        durations: dict[int, list[float]] = {
            micro_batch_size: [] for micro_batch_size in args.micro_batch_sizes
        }
        # Reverse the order on the second repetition to reduce ordering bias.
        for repetition in range(args.repeats):
            order = list(args.micro_batch_sizes)
            if repetition % 2:
                order.reverse()
            for micro_batch_size in order:
                optimizer = reset_training_state(
                    model, initial_model_state, initial_optimizer_state
                )
                started = time.perf_counter()
                metrics = runtime.update(
                    model,
                    optimizer,
                    epochs=1,
                    batch_size=args.batch_size,
                    micro_batch_size=micro_batch_size,
                    learning_rate=3e-4,
                    weight_decay=1e-4,
                    clip_ratio=0.2,
                    value_clip=0.2,
                    value_coefficient=0.5,
                    entropy_coefficient=0.01,
                    gradient_clip_norm=1.0,
                    logit_scale=12.0,
                    target_kl=math.inf,
                    data_parallel=True,
                    teacher_actor=None,
                    distillation_coefficient=0.0,
                )
                seconds = time.perf_counter() - started
                durations[micro_batch_size].append(seconds)
                print(
                    f"microbatch={micro_batch_size} repetition={repetition + 1} "
                    f"seconds={seconds:.3f} "
                    f"updates={metrics['training/updates_this_iteration']:.0f}",
                    flush=True,
                )
    finally:
        runtime.close()

    transitions = args.episodes * 99
    medians = {
        micro_batch_size: statistics.median(samples)
        for micro_batch_size, samples in durations.items()
    }
    fastest = min(medians, key=medians.__getitem__)
    baseline = medians[512] if 512 in medians else medians[fastest]
    report = {
        "batch_size": args.batch_size,
        "checkpoint": str(args.checkpoint),
        "episodes": args.episodes,
        "fastest_micro_batch_size": fastest,
        "model": {
            "channels": channels,
            "future_mode": "none",
            "residual_blocks": residual_blocks,
        },
        "results": {
            str(micro_batch_size): {
                "median_seconds": medians[micro_batch_size],
                "samples_seconds": samples,
                "speedup_vs_512": baseline / medians[micro_batch_size],
                "transitions_per_second": transitions / medians[micro_batch_size],
            }
            for micro_batch_size, samples in durations.items()
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
