from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import random
import shlex
import statistics
import subprocess
import time
from pathlib import Path

SIDE = 10
CELL_COUNT = SIDE * SIDE
ACTIONS = {"F": 0, "B": 1, "L": 2, "R": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate two AHC015 Rust solver settings without third-party packages"
    )
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--candidate-executable", type=Path)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=515015)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--baseline-args", default="")
    parser.add_argument("--candidate-args")
    parser.add_argument("--timeout", type=float, default=10.0)
    return parser.parse_args()


def generate_case(seed: int) -> tuple[list[int], list[int]]:
    random_generator = random.Random(seed)
    flavors = [random_generator.randrange(1, 4) for _ in range(CELL_COUNT)]
    ranks = [random_generator.randrange(1, CELL_COUNT - turn + 1) for turn in range(CELL_COUNT)]
    return flavors, ranks


def tilt(board: list[int], action: int) -> list[int]:
    result = [0] * CELL_COUNT
    if action in (0, 1):
        for column in range(SIDE):
            values = [board[row * SIDE + column] for row in range(SIDE)]
            values = [value for value in values if value]
            offset = 0 if action == 0 else SIDE - len(values)
            for index, value in enumerate(values):
                result[(offset + index) * SIDE + column] = value
    else:
        for row in range(SIDE):
            values = [board[row * SIDE + column] for column in range(SIDE)]
            values = [value for value in values if value]
            offset = 0 if action == 2 else SIDE - len(values)
            for index, value in enumerate(values):
                result[row * SIDE + offset + index] = value
    return result


def place_at_rank(board: list[int], rank: int, flavor: int) -> None:
    for index, value in enumerate(board):
        if value == 0:
            rank -= 1
            if rank == 0:
                board[index] = flavor
                return
    raise ValueError("rank does not identify an empty cell")


def connectivity_numerator(board: list[int]) -> int:
    visited = [False] * CELL_COUNT
    result = 0
    for start, flavor in enumerate(board):
        if flavor == 0 or visited[start]:
            continue
        visited[start] = True
        stack = [start]
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            row, column = divmod(current, SIDE)
            for next_row, next_column in (
                (row - 1, column),
                (row + 1, column),
                (row, column - 1),
                (row, column + 1),
            ):
                if not (0 <= next_row < SIDE and 0 <= next_column < SIDE):
                    continue
                neighbor = next_row * SIDE + next_column
                if not visited[neighbor] and board[neighbor] == flavor:
                    visited[neighbor] = True
                    stack.append(neighbor)
        result += size * size
    return result


def run_solver(
    executable: Path,
    extra_args: list[str],
    flavors: list[int],
    ranks: list[int],
    timeout: float,
) -> tuple[int, float]:
    input_text = " ".join(map(str, flavors)) + "\n" + "\n".join(map(str, ranks)) + "\n"
    started = time.perf_counter()
    completed = subprocess.run(
        [str(executable), *extra_args],
        input=input_text,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr)
    actions = completed.stdout.splitlines()
    if len(actions) != CELL_COUNT or any(action not in ACTIONS for action in actions):
        raise RuntimeError("solver did not emit exactly 100 valid actions")

    board = [0] * CELL_COUNT
    for flavor, rank, action in zip(flavors, ranks, actions, strict=True):
        place_at_rank(board, rank, flavor)
        board = tilt(board, ACTIONS[action])
    totals = [flavors.count(flavor) for flavor in range(1, 4)]
    denominator = sum(total * total for total in totals)
    score = math.floor(1_000_000 * connectivity_numerator(board) / denominator + 0.5)
    return score, elapsed


def summary(scores: list[int], times: list[float]) -> dict[str, float]:
    return {
        "mean_score": statistics.fmean(scores),
        "score_standard_error": statistics.stdev(scores) / math.sqrt(len(scores)),
        "mean_seconds": statistics.fmean(times),
        "p95_seconds": sorted(times)[math.ceil(0.95 * len(times)) - 1],
        "maximum_seconds": max(times),
    }


def main() -> None:
    args = parse_args()
    if args.episodes < 2 or args.workers <= 0 or args.timeout <= 0:
        raise ValueError("episodes must be at least 2 and workers/timeout must be positive")
    executable = args.executable.resolve()
    candidate_executable = (
        executable if args.candidate_executable is None else args.candidate_executable.resolve()
    )
    baseline_args = shlex.split(args.baseline_args)
    candidate_args = None if args.candidate_args is None else shlex.split(args.candidate_args)
    candidate_enabled = args.candidate_executable is not None or candidate_args is not None
    case_seeds = [args.seed + index for index in range(args.episodes)]

    def evaluate(case_seed: int) -> tuple[tuple[int, float], tuple[int, float] | None]:
        flavors, ranks = generate_case(case_seed)
        baseline = run_solver(executable, baseline_args, flavors, ranks, args.timeout)
        candidate = None
        if candidate_enabled:
            candidate = run_solver(
                candidate_executable,
                candidate_args or [],
                flavors,
                ranks,
                args.timeout,
            )
        return baseline, candidate

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(evaluate, case_seeds))

    baseline_scores = [result[0][0] for result in results]
    baseline_times = [result[0][1] for result in results]
    report: dict[str, object] = {
        "episodes": args.episodes,
        "seed": args.seed,
        "workers": args.workers,
        "baseline_executable": str(executable),
        "baseline_args": baseline_args,
        "baseline": summary(baseline_scores, baseline_times),
    }
    if candidate_enabled:
        candidate_scores = [result[1][0] for result in results if result[1] is not None]
        candidate_times = [result[1][1] for result in results if result[1] is not None]
        differences = [
            candidate - baseline
            for candidate, baseline in zip(candidate_scores, baseline_scores, strict=True)
        ]
        report.update(
            {
                "candidate_executable": str(candidate_executable),
                "candidate_args": candidate_args,
                "candidate": summary(candidate_scores, candidate_times),
                "paired": {
                    "mean_gain": statistics.fmean(differences),
                    "gain_standard_error": statistics.stdev(differences)
                    / math.sqrt(len(differences)),
                    "win_rate": sum(difference > 0 for difference in differences)
                    / len(differences),
                    "tie_rate": sum(difference == 0 for difference in differences)
                    / len(differences),
                },
            }
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
