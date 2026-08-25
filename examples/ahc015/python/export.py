from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ahc_ml.checkpoint import load_checkpoint
from ahc_ml.export import export_quantized_state_dict, export_state_dict

from .model import FILM_PARAMETER_NAMES, PARAMETER_COUNT, Ahc015ValueNet, parameter_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the AHC015 PPO actor for Rust")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/ahc015/model.bin"))
    parser.add_argument(
        "--quantized-output",
        type=Path,
        default=Path("outputs/ahc015/model.q8.bin"),
    )
    parser.add_argument(
        "--rust-output",
        type=Path,
        default=Path("examples/ahc015/rust/src/generated_model.rs"),
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    model = Ahc015ValueNet()
    if parameter_count(model) != PARAMETER_COUNT:
        raise RuntimeError("AHC015 model parameter count changed unexpectedly")
    if args.checkpoint is not None:
        load_checkpoint(
            args.checkpoint,
            model=model,
            allowed_missing_keys=FILM_PARAMETER_NAMES,
        )

    metadata = {
        "architecture": "ahc015-ppo-actor-144x9-film-v2",
        "training_algorithm": "ppo",
        "channels": 144,
        "residual_blocks": 9,
        "parameter_count": PARAMETER_COUNT,
        "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
        "zero_residual_baseline": args.checkpoint is None,
    }
    float_manifest = export_state_dict(model.state_dict(), args.output, metadata=metadata)
    quantized_manifest = export_quantized_state_dict(
        model.state_dict(),
        args.quantized_output,
        metadata=metadata,
        rust_source=args.rust_output,
    )
    print(f"float tensors: {float_manifest['tensor_count']}")
    print(f"quantized bytes: {quantized_manifest['compressed_size']}")
    print(f"rust source: {args.rust_output}")


if __name__ == "__main__":
    main()
