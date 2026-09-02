from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from ahc_ml.checkpoint import load_checkpoint
from ahc_ml.device import select_device

from .afterstate_model import AfterstateValueNet
from .afterstate_simulation import evaluate_afterstate_policy
from .game import EpisodeState, official_score
from .model import Ahc015ValueNet, dimensions_from_state_dict
from .simulation import evaluate_policy, evaluate_random_policy, generate_cases

ACTION_FROM_CHAR = {"F": 0, "B": 1, "L": 2, "R": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AHC015 policies on paired cases")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=515015)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--inference-batch-size", type=int, default=4096)
    parser.add_argument(
        "--rust-executable",
        type=Path,
        help="also evaluate the quantized model embedded in this Rust executable",
    )
    parser.add_argument(
        "--rust-model",
        type=Path,
        help="model.bin or model.q8.bin passed to the Rust executable with --model",
    )
    parser.add_argument("--rust-workers", type=int, default=min(8, os.cpu_count() or 1))
    return parser.parse_args()


def summarize(name: str, scores: np.ndarray) -> dict[str, float | str]:
    return {
        "policy": name,
        "mean_score": float(scores.mean()),
        "standard_error": float(scores.std(ddof=1) / np.sqrt(len(scores))),
    }


def evaluate_rust_case(
    executable: Path,
    model_path: Path | None,
    flavors: np.ndarray,
    ranks: np.ndarray,
) -> int:
    input_lines = [" ".join(map(str, flavors.tolist()))]
    input_lines.extend(map(str, ranks.tolist()))
    command = [str(executable)]
    if model_path is not None:
        command.extend(("--model", str(model_path)))
    completed = subprocess.run(
        command,
        input=("\n".join(input_lines) + "\n").encode(),
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.decode(errors="replace"))
    action_chars = completed.stdout.decode().splitlines()
    invalid_action = any(char not in ACTION_FROM_CHAR for char in action_chars)
    if len(action_chars) != len(ranks) or invalid_action:
        raise RuntimeError("Rust solver did not output exactly 100 valid actions")
    state = EpisodeState.new(flavors)
    for rank, action_char in zip(ranks, action_chars, strict=True):
        state.place_rank(int(rank))
        state.apply(ACTION_FROM_CHAR[action_char])
    return official_score(state.board, state.denominator)


def evaluate_rust_policy(
    executable: Path,
    model_path: Path | None,
    flavors: np.ndarray,
    ranks: np.ndarray,
    workers: int,
) -> np.ndarray:
    if workers <= 0:
        raise ValueError("--rust-workers must be positive")
    executable = executable.resolve()
    if not executable.is_file():
        raise FileNotFoundError(executable)
    if model_path is not None:
        model_path = model_path.resolve()
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        scores = executor.map(
            lambda case: evaluate_rust_case(executable, model_path, *case),
            zip(flavors, ranks, strict=True),
        )
        return np.fromiter(scores, dtype=np.int64, count=len(flavors))


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.rust_model is not None and args.rust_executable is None:
        raise ValueError("--rust-model requires --rust-executable")
    device, _ = select_device(args.device)
    flavors, ranks = generate_cases(args.episodes, args.seed)
    checkpoint = None
    phi_greedy_baseline = True
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        phi_greedy_baseline = bool(
            checkpoint.get("config", {})
            .get("evaluation", {})
            .get("phi_greedy_baseline", True)
        )
    random_result = evaluate_random_policy(flavors, ranks, seed=args.seed + 1)
    reports = [summarize("random", random_result.scores)]
    greedy = None
    if phi_greedy_baseline:
        greedy = evaluate_policy(
            None,
            device,
            flavors,
            ranks,
            inference_batch_size=args.inference_batch_size,
        )
        reports.append(summarize("phi-greedy", greedy.scores))
    if args.checkpoint is not None:
        assert checkpoint is not None
        channels, residual_blocks = dimensions_from_state_dict(checkpoint["model_state_dict"])
        input_mode = checkpoint.get("config", {}).get("model", {}).get("input_mode")
        if input_mode is None:
            output_size = checkpoint["model_state_dict"]["output.weight"].shape[0]
            input_mode = "afterstate" if output_size == 1 else "pretilt"
        if input_mode == "afterstate":
            future_mode = (
                checkpoint.get("config", {}).get("model", {}).get("future_mode", "none")
            )
            model = AfterstateValueNet(channels, residual_blocks, future_mode).to(device)
            learned_evaluator = evaluate_afterstate_policy
        else:
            model = Ahc015ValueNet(channels, residual_blocks).to(device)
            learned_evaluator = evaluate_policy
        load_checkpoint(args.checkpoint, model=model, map_location=device)
        learned_kwargs = {"inference_batch_size": args.inference_batch_size}
        if input_mode == "afterstate":
            checkpoint_metrics = checkpoint.get("metrics", {})
            checkpoint_ppo = checkpoint.get("config", {}).get("ppo", {})
            learned_kwargs["policy_phi_coefficient"] = float(
                checkpoint_metrics.get(
                    "training/policy_phi_coefficient",
                    checkpoint_ppo.get("policy_phi_coefficient_start", 1.0),
                )
            )
            learned_kwargs["future_mode"] = future_mode
        learned = learned_evaluator(
            model,
            device,
            flavors,
            ranks,
            **learned_kwargs,
        )
        report = summarize("learned", learned.scores)
        if greedy is not None:
            difference = learned.scores - greedy.scores
            report.update(
                {
                    "paired_mean_gain": float(difference.mean()),
                    "paired_gain_standard_error": float(
                        difference.std(ddof=1) / np.sqrt(len(difference))
                    ),
                    "win_rate": float(np.mean(difference > 0)),
                }
            )
        reports.append(report)
    else:
        learned = None
    if args.rust_executable is not None:
        rust_scores = evaluate_rust_policy(
            args.rust_executable,
            args.rust_model,
            flavors,
            ranks,
            args.rust_workers,
        )
        report = summarize("rust-quantized", rust_scores)
        if greedy is not None:
            difference = rust_scores - greedy.scores
            report.update(
                {
                    "paired_mean_gain": float(difference.mean()),
                    "paired_gain_standard_error": float(
                        difference.std(ddof=1) / np.sqrt(len(difference))
                    ),
                    "win_rate": float(np.mean(difference > 0)),
                }
            )
        if learned is not None:
            quantization_difference = rust_scores - learned.scores
            report.update(
                {
                    "float_score_match_rate": float(np.mean(quantization_difference == 0)),
                    "paired_mean_gain_vs_float": float(quantization_difference.mean()),
                    "paired_gain_vs_float_standard_error": float(
                        quantization_difference.std(ddof=1) / np.sqrt(len(quantization_difference))
                    ),
                }
            )
        reports.append(report)
    json.dump(reports, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
