from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import torch
from ahc_ml.checkpoint import load_checkpoint, save_checkpoint
from ahc_ml.device import select_device
from ahc_ml.export import export_quantized_state_dict, export_state_dict
from ahc_ml.seed import seed_everything
from ahc_ml.tracking import WandbTracker
from ahc_ml.visualization import render_model_graph
from engine import evaluate, train_epoch
from mnist_config import MnistConfig, load_config
from mnist_data import create_data_loaders
from model import MnistCnn
from torch import nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MNIST-family reference model")
    parser.add_argument("--config", default="examples/mnist/config.toml")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"))
    return parser.parse_args()


def apply_overrides(config: MnistConfig, args: argparse.Namespace) -> MnistConfig:
    run = replace(config.run, device=args.device) if args.device else config.run
    data = replace(
        config.data,
        batch_size=args.batch_size or config.data.batch_size,
        num_workers=(args.num_workers if args.num_workers is not None else config.data.num_workers),
    )
    training = replace(config.training, epochs=args.epochs or config.training.epochs)
    wandb = replace(config.wandb, mode=args.wandb_mode) if args.wandb_mode else config.wandb
    return replace(config, run=run, data=data, training=training, wandb=wandb)


def build_optimizer(model: nn.Module, config: MnistConfig) -> torch.optim.Optimizer:
    name = config.optimizer.name.lower()
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.optimizer.learning_rate,
            weight_decay=config.optimizer.weight_decay,
        )
    if name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.optimizer.learning_rate,
            weight_decay=config.optimizer.weight_decay,
        )
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.optimizer.learning_rate,
            weight_decay=config.optimizer.weight_decay,
            momentum=0.9,
        )
    raise ValueError(f"unsupported optimizer: {config.optimizer.name}")


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    seed_everything(config.run.seed, deterministic=config.run.deterministic)
    device, device_info = select_device(config.run.device)

    run_name = datetime.now().strftime(f"{config.data.dataset}-%Y%m%d-%H%M%S")
    output_dir = Path(config.run.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    config_dict = config.to_dict()
    config_dict["device_info"] = device_info.to_dict()
    (output_dir / "config.json").write_text(
        json.dumps(config_dict, indent=2, sort_keys=True) + "\n"
    )

    tracker = WandbTracker(
        project=config.wandb.project,
        entity=config.wandb.entity,
        mode=config.wandb.mode,
        name=run_name,
        config=config_dict,
        directory=output_dir,
    )

    print(f"device: {device_info.selected} ({device_info.name})")
    train_loader, test_loader = create_data_loaders(config.data, device=device)
    model = MnistCnn(config.model.channels, config.model.hidden_size)
    graph_svg, graph_png = render_model_graph(
        model,
        input_size=(1, 1, 28, 28),
        output_stem=output_dir / "model-graph",
    )
    tracker.log_image(
        graph_png,
        key="model/architecture",
        caption=f"{config.data.dataset} CNN architecture",
    )
    print(f"model graph: {graph_svg}")
    model = model.to(device)
    optimizer = build_optimizer(model, config)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    best_path = output_dir / "best.pt"

    try:
        for epoch in range(1, config.training.epochs + 1):
            train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
            test_metrics = evaluate(model, test_loader, criterion, device)
            metrics = {
                "epoch": epoch,
                **train_metrics.with_prefix("train"),
                **test_metrics.with_prefix("test"),
            }
            tracker.log(metrics, step=epoch)
            print(
                f"epoch {epoch:02d}/{config.training.epochs:02d} "
                f"train_loss={train_metrics.loss:.4f} "
                f"test_loss={test_metrics.loss:.4f} "
                f"test_accuracy={test_metrics.accuracy:.4%} "
                f"train_samples/s={train_metrics.samples_per_second:.0f}"
            )

            save_checkpoint(
                output_dir / "last.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=config_dict,
                metrics=metrics,
            )
            if test_metrics.accuracy > best_accuracy:
                best_accuracy = test_metrics.accuracy
                save_checkpoint(
                    best_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    config=config_dict,
                    metrics=metrics,
                )

        load_checkpoint(best_path, model=model, map_location=device)
        model = model.to("cpu").eval()
        model_path = output_dir / "model.bin"
        metadata = {
            "architecture": "mnist-cnn-v1",
            "dataset": config.data.dataset,
            "channels": list(config.model.channels),
            "hidden_size": config.model.hidden_size,
            "input_shape": [1, 28, 28],
            "output_size": 10,
        }
        export_state_dict(model.state_dict(), model_path, metadata=metadata)
        quantized_path = output_dir / "model.q8.bin"
        rust_source = output_dir / "model_data.rs"
        quantized_manifest = export_quantized_state_dict(
            model.state_dict(),
            quantized_path,
            metadata=metadata,
            rust_source=rust_source,
        )
        tracker.log_artifact(best_path, name=f"{run_name}-checkpoint", artifact_type="model")
        tracker.log_artifact(model_path, name=f"{run_name}-rust-weights", artifact_type="model")
        tracker.log_artifact(
            quantized_path,
            name=f"{run_name}-rust-weights-q8",
            artifact_type="model",
        )
        print(f"best accuracy: {best_accuracy:.4%}")
        print(
            f"quantized model: {quantized_manifest['compressed_size']} bytes "
            f"({quantized_manifest['compression_ratio']:.1%} of quantized data)"
        )
        print(f"outputs: {output_dir}")
    finally:
        tracker.finish()


if __name__ == "__main__":
    main()
