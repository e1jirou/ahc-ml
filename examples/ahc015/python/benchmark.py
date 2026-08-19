from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from .simulation import generate_cases


def case_bytes(seed: int) -> bytes:
    flavors, ranks = generate_cases(1, seed)
    lines = [" ".join(map(str, flavors[0]))]
    lines.extend(str(int(rank)) for rank in ranks[0])
    return ("\n".join(lines) + "\n").encode()


def run_once(command: list[str], input_data: bytes) -> float:
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        input=input_data,
        capture_output=True,
        check=False,
    )
    elapsed = time.perf_counter() - start
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.decode(errors="replace"))
    actions = completed.stdout.decode().splitlines()
    if len(actions) != 100 or any(action not in {"F", "B", "L", "R"} for action in actions):
        raise RuntimeError("solver output is not 100 valid actions")
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the final AHC015 Rust executable")
    parser.add_argument(
        "--executable",
        type=Path,
        default=Path("target/release/ahc015-inference"),
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=15015)
    args = parser.parse_args()
    if args.warmups < 0 or args.runs <= 0:
        raise ValueError("warmups must be non-negative and runs must be positive")
    command = [str(args.executable)]
    if args.model is not None:
        command.extend(("--model", str(args.model)))
    input_data = case_bytes(args.seed)
    for _ in range(args.warmups):
        run_once(command, input_data)
    measurements = np.asarray(
        [run_once(command, input_data) for _ in range(args.runs)], dtype=np.float64
    )
    report = {
        "command": command,
        "seed": args.seed,
        "runs": args.runs,
        "mean_seconds": float(measurements.mean()),
        "median_seconds": float(np.median(measurements)),
        "minimum_seconds": float(measurements.min()),
        "maximum_seconds": float(measurements.max()),
        "measurements_seconds": measurements.tolist(),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
