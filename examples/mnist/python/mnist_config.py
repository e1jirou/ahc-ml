from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunConfig:
    seed: int
    device: str
    deterministic: bool
    output_dir: str


@dataclass(frozen=True)
class DataConfig:
    dataset: str
    root: str
    batch_size: int
    num_workers: int


@dataclass(frozen=True)
class ModelConfig:
    channels: tuple[int, int]
    hidden_size: int


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    learning_rate: float
    weight_decay: float


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int


@dataclass(frozen=True)
class WandbConfig:
    project: str
    entity: str
    mode: str


@dataclass(frozen=True)
class MnistConfig:
    run: RunConfig
    data: DataConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    training: TrainingConfig
    wandb: WandbConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> MnistConfig:
    with Path(path).open("rb") as file:
        values = tomllib.load(file)

    channels = tuple(int(value) for value in values["model"]["channels"])
    if len(channels) != 2 or any(value <= 0 for value in channels):
        raise ValueError("model.channels must contain two positive integers")

    config = MnistConfig(
        run=RunConfig(**values["run"]),
        data=DataConfig(**values["data"]),
        model=ModelConfig(channels=channels, hidden_size=values["model"]["hidden_size"]),
        optimizer=OptimizerConfig(**values["optimizer"]),
        training=TrainingConfig(**values["training"]),
        wandb=WandbConfig(**values["wandb"]),
    )
    if config.data.batch_size <= 0:
        raise ValueError("data.batch_size must be positive")
    if config.data.num_workers < 0:
        raise ValueError("data.num_workers must be non-negative")
    if config.data.dataset not in {"mnist", "fashion_mnist"}:
        raise ValueError("data.dataset must be 'mnist' or 'fashion_mnist'")
    if config.training.epochs <= 0:
        raise ValueError("training.epochs must be positive")
    return config
