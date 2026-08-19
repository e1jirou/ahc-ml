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
    experiment_log: str


@dataclass(frozen=True)
class TrainingConfig:
    iterations: int
    rollout_episodes: int
    updates_per_iteration: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    target_update_interval: int
    checkpoint_interval: int
    max_hours: float


@dataclass(frozen=True)
class ExplorationConfig:
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_iterations: int


@dataclass(frozen=True)
class ReplayConfig:
    capacity: int
    minimum_size: int


@dataclass(frozen=True)
class BackupConfig:
    placement_samples: int
    enumerate_threshold: int
    inference_batch_size: int
    mc_beta_start: float
    mc_beta_end: float
    mc_beta_decay_iterations: int
    minimum_placed_start: int
    curriculum_iterations: int


@dataclass(frozen=True)
class EvaluationConfig:
    interval: int
    episodes: int
    seed: int


@dataclass(frozen=True)
class WandbConfig:
    project: str
    entity: str
    mode: str


@dataclass(frozen=True)
class Ahc015Config:
    run: RunConfig
    training: TrainingConfig
    exploration: ExplorationConfig
    replay: ReplayConfig
    backup: BackupConfig
    evaluation: EvaluationConfig
    wandb: WandbConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def load_config(path: str | Path) -> Ahc015Config:
    with Path(path).open("rb") as file:
        values = tomllib.load(file)
    config = Ahc015Config(
        run=RunConfig(**values["run"]),
        training=TrainingConfig(**values["training"]),
        exploration=ExplorationConfig(**values["exploration"]),
        replay=ReplayConfig(**values["replay"]),
        backup=BackupConfig(**values["backup"]),
        evaluation=EvaluationConfig(**values["evaluation"]),
        wandb=WandbConfig(**values["wandb"]),
    )
    for name, value in (
        ("training.iterations", config.training.iterations),
        ("training.rollout_episodes", config.training.rollout_episodes),
        ("training.updates_per_iteration", config.training.updates_per_iteration),
        ("training.batch_size", config.training.batch_size),
        ("training.learning_rate", config.training.learning_rate),
        ("training.target_update_interval", config.training.target_update_interval),
        ("training.max_hours", config.training.max_hours),
        ("replay.capacity", config.replay.capacity),
        ("replay.minimum_size", config.replay.minimum_size),
        ("backup.placement_samples", config.backup.placement_samples),
        ("backup.inference_batch_size", config.backup.inference_batch_size),
        ("evaluation.interval", config.evaluation.interval),
        ("evaluation.episodes", config.evaluation.episodes),
    ):
        _positive(name, value)
    if config.replay.minimum_size > config.replay.capacity:
        raise ValueError("replay.minimum_size must not exceed replay.capacity")
    if not 0 <= config.exploration.epsilon_end <= config.exploration.epsilon_start <= 1:
        raise ValueError("exploration epsilons must satisfy 0 <= end <= start <= 1")
    if not 0 <= config.backup.mc_beta_end <= config.backup.mc_beta_start <= 1:
        raise ValueError("backup MC beta must satisfy 0 <= end <= start <= 1")
    if not 1 <= config.backup.minimum_placed_start <= 99:
        raise ValueError("backup.minimum_placed_start must be in 1..=99")
    if config.wandb.mode not in {"online", "offline", "disabled"}:
        raise ValueError("wandb.mode must be online, offline, or disabled")
    return config
