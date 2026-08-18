from __future__ import annotations

import argparse
from pathlib import Path

from ahc_ml.checkpoint import load_checkpoint
from ahc_ml.export import export_quantized_state_dict, export_state_dict
from mnist_config import load_config
from model import MnistCnn


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an MNIST-family checkpoint for Rust")
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--config", default="examples/mnist/config.toml")
    parser.add_argument(
        "--quantized-output",
        help="compressed int8 output (default: model.q8.bin next to output)",
    )
    parser.add_argument(
        "--rust-output",
        help="Rust Base93 constant (default: model_data.rs next to output)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    model = MnistCnn(config.model.channels, config.model.hidden_size)
    load_checkpoint(args.checkpoint, model=model)
    output = Path(args.output)
    metadata = {
        "architecture": "mnist-cnn-v1",
        "dataset": config.data.dataset,
        "channels": list(config.model.channels),
        "hidden_size": config.model.hidden_size,
        "input_shape": [1, 28, 28],
        "output_size": 10,
    }
    state_dict = model.eval().state_dict()
    manifest = export_state_dict(state_dict, output, metadata=metadata)
    quantized_output = (
        Path(args.quantized_output) if args.quantized_output else output.parent / "model.q8.bin"
    )
    rust_output = Path(args.rust_output) if args.rust_output else output.parent / "model_data.rs"
    quantized_manifest = export_quantized_state_dict(
        state_dict,
        quantized_output,
        metadata=metadata,
        rust_source=rust_output,
    )
    print(f"exported {manifest['tensor_count']} tensors to {output}")
    print(
        f"exported quantized model to {quantized_output}: "
        f"{quantized_manifest['compressed_size']} bytes"
    )
    print(f"exported Rust model constant to {rust_output}")


if __name__ == "__main__":
    main()
