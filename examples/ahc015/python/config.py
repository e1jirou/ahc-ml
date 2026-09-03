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
    name_prefix: str = "small"


@dataclass(frozen=True)
class TrainingConfig:
    iterations: int
    rollout_episodes: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    checkpoint_interval: int
    max_hours: float
    inference_batch_size: int
    micro_batch_size: int = 128
    data_parallel: bool = False
    rollout_processes: int = 1


@dataclass(frozen=True)
class ModelConfig:
    channels: int
    residual_blocks: int
    input_mode: str = "pretilt"
    future_mode: str = "none"


@dataclass(frozen=True)
class PpoConfig:
    gamma: float
    gae_lambda: float
    clip_ratio: float
    value_clip: float
    value_coefficient: float
    entropy_coefficient: float
    logit_scale: float
    target_kl: float
    policy_phi_coefficient_start: float = 1.0
    policy_phi_coefficient_end: float = 1.0
    policy_phi_anneal_hours: float = 0.0
    reward_mode: str = "potential_shaping"


@dataclass(frozen=True)
class DistillationConfig:
    coefficient_start: float = 0.0
    coefficient_end: float = 0.0
    anneal_hours: float = 0.0


@dataclass(frozen=True)
class EvaluationConfig:
    interval: int
    episodes: int
    seed: int
    phi_greedy_baseline: bool = True


@dataclass(frozen=True)
class WandbConfig:
    project: str
    entity: str
    mode: str


@dataclass(frozen=True)
class Ahc015Config:
    run: RunConfig
    model: ModelConfig
    training: TrainingConfig
    ppo: PpoConfig
    distillation: DistillationConfig
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
        model=ModelConfig(**values.get("model", {"channels": 64, "residual_blocks": 10})),
        training=TrainingConfig(**values["training"]),
        ppo=PpoConfig(**values["ppo"]),
        distillation=DistillationConfig(**values.get("distillation", {})),
        evaluation=EvaluationConfig(**values["evaluation"]),
        wandb=WandbConfig(**values["wandb"]),
    )
    for name, value in (
        ("model.channels", config.model.channels),
        ("model.residual_blocks", config.model.residual_blocks),
        ("training.iterations", config.training.iterations),
        ("training.rollout_episodes", config.training.rollout_episodes),
        ("training.epochs", config.training.epochs),
        ("training.batch_size", config.training.batch_size),
        ("training.micro_batch_size", config.training.micro_batch_size),
        ("training.rollout_processes", config.training.rollout_processes),
        ("training.learning_rate", config.training.learning_rate),
        ("training.gradient_clip_norm", config.training.gradient_clip_norm),
        ("training.checkpoint_interval", config.training.checkpoint_interval),
        ("training.max_hours", config.training.max_hours),
        ("training.inference_batch_size", config.training.inference_batch_size),
        ("ppo.gae_lambda", config.ppo.gae_lambda),
        ("ppo.clip_ratio", config.ppo.clip_ratio),
        ("ppo.value_clip", config.ppo.value_clip),
        ("ppo.value_coefficient", config.ppo.value_coefficient),
        ("ppo.logit_scale", config.ppo.logit_scale),
        ("ppo.target_kl", config.ppo.target_kl),
        ("evaluation.interval", config.evaluation.interval),
        ("evaluation.episodes", config.evaluation.episodes),
    ):
        _positive(name, value)
    if config.training.micro_batch_size > config.training.batch_size:
        raise ValueError("training.micro_batch_size must not exceed training.batch_size")
    if not config.run.name_prefix or not config.run.name_prefix.replace("-", "").isalnum():
        raise ValueError("run.name_prefix must contain only letters, numbers, and hyphens")
    if config.training.rollout_processes not in {1, 2}:
        raise ValueError("training.rollout_processes must be 1 or 2")
    if config.model.input_mode not in {"pretilt", "afterstate"}:
        raise ValueError("model.input_mode must be pretilt or afterstate")
    if config.model.future_mode not in {"none", "full_add", "full_late"}:
        raise ValueError("model.future_mode must be none, full_add, or full_late")
    if config.model.future_mode != "none" and config.model.input_mode != "afterstate":
        raise ValueError("future inputs currently support only afterstate input")
    if config.ppo.gamma != 1.0:
        raise ValueError("ppo.gamma must be 1.0 so potential shaping preserves the objective")
    if not 0 < config.ppo.gae_lambda <= 1:
        raise ValueError("ppo.gae_lambda must be in (0, 1]")
    if config.ppo.entropy_coefficient < 0:
        raise ValueError("ppo.entropy_coefficient must be non-negative")
    if config.ppo.policy_phi_coefficient_start < 0:
        raise ValueError("ppo.policy_phi_coefficient_start must be non-negative")
    if config.ppo.policy_phi_coefficient_end < 0:
        raise ValueError("ppo.policy_phi_coefficient_end must be non-negative")
    if config.ppo.policy_phi_anneal_hours < 0:
        raise ValueError("ppo.policy_phi_anneal_hours must be non-negative")
    if (
        config.ppo.policy_phi_coefficient_start != config.ppo.policy_phi_coefficient_end
        and config.ppo.policy_phi_anneal_hours == 0
    ):
        raise ValueError("ppo.policy_phi_anneal_hours must be positive when coefficients differ")
    if config.ppo.reward_mode not in {"potential_shaping", "terminal"}:
        raise ValueError("ppo.reward_mode must be potential_shaping or terminal")
    if config.distillation.coefficient_start < 0:
        raise ValueError("distillation.coefficient_start must be non-negative")
    if config.distillation.coefficient_end < 0:
        raise ValueError("distillation.coefficient_end must be non-negative")
    if config.distillation.anneal_hours < 0:
        raise ValueError("distillation.anneal_hours must be non-negative")
    if (
        config.distillation.coefficient_start != config.distillation.coefficient_end
        and config.distillation.anneal_hours == 0
    ):
        raise ValueError("distillation.anneal_hours must be positive when coefficients differ")
    if config.ppo.reward_mode == "terminal":
        if config.model.input_mode != "afterstate":
            raise ValueError("terminal reward mode currently supports only afterstate input")
        if config.ppo.policy_phi_coefficient_start != 0:
            raise ValueError("terminal reward mode requires policy Phi start coefficient 0")
        if config.ppo.policy_phi_coefficient_end != 0:
            raise ValueError("terminal reward mode requires policy Phi end coefficient 0")
        if config.ppo.gae_lambda != 1.0:
            raise ValueError("terminal reward mode requires ppo.gae_lambda = 1.0")
        if config.evaluation.phi_greedy_baseline:
            raise ValueError("terminal reward mode requires Phi-greedy evaluation to be disabled")
    if config.wandb.mode not in {"online", "offline", "disabled"}:
        raise ValueError("wandb.mode must be online, offline, or disabled")
    return config
