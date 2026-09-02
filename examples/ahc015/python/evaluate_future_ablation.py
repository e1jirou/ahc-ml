from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from ahc_ml.checkpoint import load_checkpoint

from .afterstate_model import AfterstateValueNet
from .afterstate_simulation import evaluate_afterstate_policy
from .model import dimensions_from_state_dict
from .simulation import generate_cases

ABLATIONS = ("correct", "off", "episode_shuffle", "order_shuffle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate full-future late-fusion ablations")
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--future-checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--inference-batch-size", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=5)
    return parser.parse_args()


def _evaluate_task(
    name: str,
    checkpoint_path: str,
    episodes: int,
    seed: int,
    inference_batch_size: int,
    torch_threads: int,
) -> tuple[str, np.ndarray]:
    torch.set_num_threads(torch_threads)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    channels, residual_blocks = dimensions_from_state_dict(checkpoint["model_state_dict"])
    future_mode = checkpoint.get("config", {}).get("model", {}).get("future_mode", "none")
    model = AfterstateValueNet(channels, residual_blocks, future_mode).eval()
    load_checkpoint(checkpoint_path, model=model, map_location="cpu")
    flavors, ranks = generate_cases(episodes, seed)
    metrics = checkpoint.get("metrics", {})
    ppo = checkpoint.get("config", {}).get("ppo", {})
    policy_phi_coefficient = float(
        metrics.get(
            "training/policy_phi_coefficient",
            ppo.get("policy_phi_coefficient_start", 1.0),
        )
    )
    scores = evaluate_afterstate_policy(
        model,
        torch.device("cpu"),
        flavors,
        ranks,
        inference_batch_size=inference_batch_size,
        policy_phi_coefficient=policy_phi_coefficient,
        future_mode=future_mode,
        future_ablation=name if future_mode == "full_late" else "correct",
        future_ablation_seed=seed + 1,
    ).scores
    return name, scores


def _paired_report(scores: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    difference = scores - reference
    return {
        "mean_score": float(scores.mean()),
        "difference": float(difference.mean()),
        "difference_se": float(difference.std(ddof=1) / math.sqrt(len(difference))),
        "win_rate": float(np.mean(difference > 0)),
        "tie_rate": float(np.mean(difference == 0)),
    }


def main() -> None:
    args = parse_args()
    if args.episodes <= 1:
        raise ValueError("--episodes must be greater than 1")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    task_count = len(ABLATIONS) + 1
    workers = min(args.workers, task_count)
    torch_threads = max(1, (os.cpu_count() or 1) // workers)
    tasks = [
        (
            "baseline",
            str(args.baseline_checkpoint),
            args.episodes,
            args.seed,
            args.inference_batch_size,
            torch_threads,
        )
    ]
    tasks.extend(
        (
            name,
            str(args.future_checkpoint),
            args.episodes,
            args.seed,
            args.inference_batch_size,
            torch_threads,
        )
        for name in ABLATIONS
    )
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        results = dict(executor.map(_run_task, tasks))

    baseline = results["baseline"]
    correct = results["correct"]
    report = {
        "episodes": args.episodes,
        "seed": args.seed,
        "torch_threads_per_worker": torch_threads,
        "versus_baseline": {
            name: _paired_report(scores, baseline)
            for name, scores in results.items()
            if name != "baseline"
        },
        "versus_correct_future": {
            name: _paired_report(scores, correct)
            for name, scores in results.items()
            if name not in {"baseline", "correct"}
        },
        "baseline_mean_score": float(baseline.mean()),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def _run_task(arguments: tuple[str, str, int, int, int, int]) -> tuple[str, np.ndarray]:
    return _evaluate_task(*arguments)


if __name__ == "__main__":
    main()
