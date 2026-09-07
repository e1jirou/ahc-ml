from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ahc_ml.checkpoint import load_checkpoint
from ahc_ml.export import export_quantized_state_dict, export_state_dict

from .afterstate_model import AfterstateValueNet, parameter_count
from .model import dimensions_from_state_dict

SUBMISSION_CHANNELS = 128
SUBMISSION_RESIDUAL_BLOCKS = 10


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
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state_dict = checkpoint["model_state_dict"]
        prefix = "actor." if "actor.board_stem.weight" in state_dict else ""
        channels, residual_blocks = dimensions_from_state_dict(state_dict, prefix=prefix)
    else:
        channels, residual_blocks = SUBMISSION_CHANNELS, SUBMISSION_RESIDUAL_BLOCKS
    if (channels, residual_blocks) != (SUBMISSION_CHANNELS, SUBMISSION_RESIDUAL_BLOCKS):
        raise ValueError("Rust export only supports the 128-channel, 10-block afterstate model")
    model = AfterstateValueNet(channels, residual_blocks, "none")
    if args.checkpoint is not None:
        if prefix:
            model.load_state_dict(
                {
                    name.removeprefix(prefix): value
                    for name, value in state_dict.items()
                    if name.startswith(prefix)
                }
            )
        else:
            load_checkpoint(args.checkpoint, model=model)

    parameters = parameter_count(model)
    metadata = {
        "architecture": f"ahc015-afterstate-{channels}x{residual_blocks}-occupancy-v1",
        "training_algorithm": "ppo",
        "channels": channels,
        "residual_blocks": residual_blocks,
        "parameter_count": parameters,
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
