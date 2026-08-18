from __future__ import annotations

import argparse

from ahc_ml.checkpoint import load_checkpoint
from ahc_ml.device import select_device
from engine import evaluate
from mnist_config import load_config
from mnist_data import create_data_loaders
from model import MnistCnn
from torch import nn


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an MNIST-family checkpoint")
    parser.add_argument("checkpoint")
    parser.add_argument("--config", default="examples/mnist/config.toml")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    args = parser.parse_args()

    config = load_config(args.config)
    device, device_info = select_device(args.device)
    _, test_loader = create_data_loaders(config.data, device=device)
    model = MnistCnn(config.model.channels, config.model.hidden_size).to(device)
    checkpoint = load_checkpoint(args.checkpoint, model=model, map_location=device)
    metrics = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    print(f"device: {device_info.selected} ({device_info.name})")
    print(f"checkpoint epoch: {checkpoint['epoch']}")
    print(f"test loss: {metrics.loss:.6f}")
    print(f"test accuracy: {metrics.accuracy:.4%}")
    print(f"samples/s: {metrics.samples_per_second:.0f}")


if __name__ == "__main__":
    main()
