from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from ahc_ml.device import select_device

from .game import (
    ACTION_COUNT,
    CELL_COUNT,
    SIDE,
    afterstates_batch,
    denominator,
    place_at_ranks,
    potential,
)
from .legacy_future import (
    ablate_legacy_futures,
    encode_legacy_afterstates,
    legacy_model_from_state_dict,
)
from .simulation import generate_cases

ABLATIONS = ("correct", "zero", "episode_shuffle", "order_shuffle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate whether a legacy FiLM policy uses its future sequence"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--inference-batch-size", type=int, default=4096)
    return parser.parse_args()


def evaluate_ablation(
    model: torch.nn.Module,
    device: torch.device,
    flavors: np.ndarray,
    ranks: np.ndarray,
    *,
    mode: str,
    seed: int,
    inference_batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    episodes = len(flavors)
    rng = np.random.default_rng(seed)
    episode_permutation = rng.permutation(episodes)
    order_priorities = rng.random((episodes, CELL_COUNT))
    score_denominators = np.asarray([denominator(row) for row in flavors])
    boards = np.zeros((episodes, SIDE, SIDE), dtype=np.uint8)
    selected_actions = np.empty((episodes, CELL_COUNT - 1), dtype=np.int8)
    model.eval()
    for turn in range(CELL_COUNT):
        boards = place_at_ranks(boards, ranks[:, turn], flavors[:, turn])
        if turn + 1 == CELL_COUNT:
            break
        candidates = afterstates_batch(boards)
        board_features, future_features, flat_potentials = encode_legacy_afterstates(
            candidates.reshape(episodes * ACTION_COUNT, SIDE, SIDE),
            np.tile(np.arange(ACTION_COUNT, dtype=np.int64), episodes),
            turn + 1,
            np.repeat(flavors, ACTION_COUNT, axis=0),
            np.repeat(score_denominators, ACTION_COUNT),
        )
        future_features = ablate_legacy_futures(
            future_features,
            mode,
            episode_permutation=episode_permutation,
            order_priorities=order_priorities,
            remaining_length=CELL_COUNT - turn - 1,
        )
        residuals = np.empty(episodes * ACTION_COUNT, dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(residuals), inference_batch_size):
                stop = min(start + inference_batch_size, len(residuals))
                board_tensor = torch.from_numpy(board_features[start:stop]).to(device)
                future_tensor = torch.from_numpy(future_features[start:stop]).to(device)
                residuals[start:stop] = (
                    model(board_tensor, future_tensor).cpu().numpy()
                )
        values = residuals.reshape(episodes, ACTION_COUNT) + flat_potentials.reshape(
            episodes, ACTION_COUNT
        )
        selected = np.argmax(values, axis=1)
        selected_actions[:, turn] = selected
        boards = candidates[np.arange(episodes), selected]

    final_potentials = np.asarray(
        [potential(boards[index], score_denominators[index]) for index in range(episodes)],
        dtype=np.float64,
    )
    scores = np.floor(1_000_000 * final_potentials + 0.5).astype(np.int64)
    return scores, selected_actions


def paired_report(
    scores: np.ndarray,
    actions: np.ndarray,
    reference_scores: np.ndarray,
    reference_actions: np.ndarray,
) -> dict[str, float]:
    difference = scores - reference_scores
    return {
        "mean_score": float(scores.mean()),
        "difference_vs_correct": float(difference.mean()),
        "difference_se": float(difference.std(ddof=1) / math.sqrt(len(difference))),
        "win_rate": float(np.mean(difference > 0)),
        "tie_rate": float(np.mean(difference == 0)),
        "trajectory_action_match_rate": float(np.mean(actions == reference_actions)),
    }


def main() -> None:
    args = parse_args()
    if args.episodes <= 1:
        raise ValueError("--episodes must be greater than 1")
    if args.inference_batch_size <= 0:
        raise ValueError("--inference-batch-size must be positive")
    device, device_info = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = legacy_model_from_state_dict(checkpoint["model_state_dict"]).to(device)
    flavors, ranks = generate_cases(args.episodes, args.seed)
    print(f"device: {device_info.selected} ({device_info.name})", flush=True)
    print(
        f"legacy checkpoint: {args.checkpoint}; episodes={args.episodes}; seed={args.seed}",
        flush=True,
    )

    results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    elapsed_by_mode: dict[str, float] = {}
    for mode in ABLATIONS:
        started = time.monotonic()
        results[mode] = evaluate_ablation(
            model,
            device,
            flavors,
            ranks,
            mode=mode,
            seed=args.seed + 1,
            inference_batch_size=args.inference_batch_size,
        )
        elapsed_by_mode[mode] = time.monotonic() - started
        print(f"completed {mode}: {elapsed_by_mode[mode]:.1f} seconds", flush=True)

    correct_scores, correct_actions = results["correct"]
    report = {
        "checkpoint": str(args.checkpoint),
        "episodes": args.episodes,
        "seed": args.seed,
        "device": device_info.selected,
        "results": {
            mode: paired_report(
                scores,
                actions,
                correct_scores,
                correct_actions,
            )
            for mode, (scores, actions) in results.items()
        },
        "elapsed_seconds": elapsed_by_mode,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
