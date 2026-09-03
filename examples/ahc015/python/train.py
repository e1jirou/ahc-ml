from __future__ import annotations

import argparse
import atexit
import json
import math
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from ahc_ml.checkpoint import load_checkpoint, save_checkpoint
from ahc_ml.device import select_device
from ahc_ml.seed import seed_everything
from ahc_ml.tracking import WandbTracker
from ahc_ml.visualization import render_model_graph

from .afterstate_features import FUTURE_CHANNELS, FUTURE_LENGTH
from .afterstate_model import (
    AfterstatePpoNet,
    AfterstateValueNet,
    initialize_widened_afterstate_ppo,
)
from .afterstate_parallel_runtime import ParallelAfterstateRuntime
from .afterstate_ppo import collect_afterstate_ppo_rollout, update_afterstate_ppo
from .afterstate_simulation import evaluate_afterstate_policy
from .config import Ahc015Config, PpoConfig, load_config
from .features import BOARD_CHANNELS
from .game import SIDE
from .model import Ahc015PpoNet, dimensions_from_state_dict, parameter_count
from .parallel_runtime import ParallelAhc015Runtime
from .ppo import collect_ppo_rollout, ppo_update
from .simulation import evaluate_policy, generate_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the AHC015 policy with PPO")
    parser.add_argument("--config", type=Path, default=Path("examples/ahc015/config.toml"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--max-hours", type=float)
    parser.add_argument("--rollout-episodes", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument(
        "--data-parallel",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--rollout-processes", type=int, choices=(1, 2))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--evaluation-interval", type=int)
    parser.add_argument("--evaluation-episodes", type=int)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"))
    parser.add_argument("--experiment-log", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path, help="resume from a full PPO training checkpoint")
    parser.add_argument(
        "--initialize-from",
        type=Path,
        help="initialize a 2x-wider afterstate model from a full PPO checkpoint",
    )
    parser.add_argument(
        "--distill-from",
        type=Path,
        help="add actor KL distillation from an afterstate PPO training checkpoint",
    )
    return parser.parse_args()


def append_experiment_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as file:
        file.write(text.rstrip() + "\n")


def evaluate(
    evaluation_model: torch.nn.Module,
    training_model: torch.nn.Module,
    device: torch.device,
    config: Ahc015Config,
    parallel_runtime: ParallelAhc015Runtime | ParallelAfterstateRuntime | None,
    policy_phi_coefficient: float,
) -> dict[str, float]:
    flavors, ranks = generate_cases(config.evaluation.episodes, config.evaluation.seed)
    if parallel_runtime is None:
        policy_evaluator = (
            evaluate_afterstate_policy
            if config.model.input_mode == "afterstate"
            else evaluate_policy
        )
        learned_kwargs = {"inference_batch_size": config.training.inference_batch_size}
        if config.model.input_mode == "afterstate":
            learned_kwargs["policy_phi_coefficient"] = policy_phi_coefficient
            learned_kwargs["future_mode"] = config.model.future_mode
        learned_scores = policy_evaluator(
            evaluation_model,
            device,
            flavors,
            ranks,
            **learned_kwargs,
        ).scores
        greedy_scores = None
        if config.evaluation.phi_greedy_baseline:
            greedy_scores = policy_evaluator(
                None,
                device,
                flavors,
                ranks,
                inference_batch_size=config.training.inference_batch_size,
            ).scores
    else:
        greedy_scores, learned_scores = parallel_runtime.evaluate(
            training_model,
            flavors,
            ranks,
            inference_batch_size=config.training.inference_batch_size,
        )
    metrics = {
        "evaluation/mean_score": float(learned_scores.mean()),
        "evaluation/score_se": float(learned_scores.std(ddof=1) / math.sqrt(len(learned_scores))),
    }
    if greedy_scores is not None:
        difference = learned_scores - greedy_scores
        metrics.update(
            {
                "evaluation/greedy_mean_score": float(greedy_scores.mean()),
                "evaluation/paired_gain": float(difference.mean()),
                "evaluation/paired_gain_se": float(
                    difference.std(ddof=1) / math.sqrt(len(difference))
                ),
                "evaluation/win_rate": float(np.mean(difference > 0)),
            }
        )
    return metrics


def scheduled_policy_phi_coefficient(config: PpoConfig, elapsed_hours: float) -> float:
    if config.policy_phi_anneal_hours == 0:
        return config.policy_phi_coefficient_end
    progress = min(max(elapsed_hours / config.policy_phi_anneal_hours, 0.0), 1.0)
    return (
        config.policy_phi_coefficient_start
        + (config.policy_phi_coefficient_end - config.policy_phi_coefficient_start) * progress
    )


def scheduled_coefficient(
    start: float, end: float, anneal_hours: float, elapsed_hours: float
) -> float:
    if anneal_hours == 0:
        return end
    progress = min(max(elapsed_hours / anneal_hours, 0.0), 1.0)
    return start + (end - start) * progress


def load_distillation_teacher(path: Path) -> AfterstateValueNet:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    source_model_config = checkpoint.get("config", {}).get("model", {})
    if source_model_config.get("input_mode", "afterstate") != "afterstate":
        raise ValueError("distillation teacher must use afterstate input")
    state = checkpoint["model_state_dict"]
    actor_state = {
        name.removeprefix("actor."): value
        for name, value in state.items()
        if name.startswith("actor.")
        and not any(
            part in name
            for part in (".future_add.", ".future_encoder.", ".fusion.", ".correction.")
        )
    }
    if not actor_state:
        raise ValueError("distillation checkpoint must contain a full PPO actor")
    channels, residual_blocks = dimensions_from_state_dict(state, prefix="actor.")
    teacher = AfterstateValueNet(channels, residual_blocks, future_mode="none")
    teacher.load_state_dict(actor_state)
    teacher.requires_grad_(False)
    teacher.eval()
    return teacher


def load_training_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, object]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    checkpoint_state = checkpoint["model_state_dict"]
    incompatible = model.load_state_dict(checkpoint_state, strict=False)
    allowed_missing = {
        name
        for name in model.state_dict()
        if any(
            module_name in name
            for module_name in (
                ".future_add.",
                ".future_encoder.",
                ".fusion.",
                ".correction.",
            )
        )
        and name not in checkpoint_state
    }
    if set(incompatible.missing_keys) != allowed_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "incompatible checkpoint state: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )

    old_optimizer = checkpoint["optimizer_state_dict"]
    old_parameter_ids = [
        parameter_id for group in old_optimizer["param_groups"] for parameter_id in group["params"]
    ]
    old_parameter_names = list(checkpoint_state)
    if len(old_parameter_ids) != len(old_parameter_names):
        raise RuntimeError("checkpoint optimizer parameters do not match model parameters")
    old_state_by_name = {
        name: old_optimizer["state"].get(parameter_id, {})
        for name, parameter_id in zip(old_parameter_names, old_parameter_ids, strict=True)
    }
    new_optimizer = optimizer.state_dict()
    new_parameter_ids = [
        parameter_id for group in new_optimizer["param_groups"] for parameter_id in group["params"]
    ]
    new_parameter_names = [name for name, _ in model.named_parameters()]
    if len(new_parameter_ids) != len(new_parameter_names):
        raise RuntimeError("optimizer parameters do not match model parameters")
    new_optimizer["state"] = {
        parameter_id: old_state_by_name[name]
        for name, parameter_id in zip(new_parameter_names, new_parameter_ids, strict=True)
        if name in old_state_by_name and old_state_by_name[name]
    }
    optimizer.load_state_dict(new_optimizer)
    return checkpoint


def apply_overrides(config: Ahc015Config, args: argparse.Namespace) -> Ahc015Config:
    run = replace(
        config.run,
        seed=args.seed if args.seed is not None else config.run.seed,
        device=args.device or config.run.device,
        experiment_log=(
            str(args.experiment_log) if args.experiment_log else config.run.experiment_log
        ),
    )
    training = replace(
        config.training,
        iterations=(args.iterations if args.iterations is not None else config.training.iterations),
        max_hours=args.max_hours if args.max_hours is not None else config.training.max_hours,
        rollout_episodes=(
            args.rollout_episodes
            if args.rollout_episodes is not None
            else config.training.rollout_episodes
        ),
        batch_size=(args.batch_size if args.batch_size is not None else config.training.batch_size),
        micro_batch_size=(
            args.micro_batch_size
            if args.micro_batch_size is not None
            else config.training.micro_batch_size
        ),
        data_parallel=(
            args.data_parallel if args.data_parallel is not None else config.training.data_parallel
        ),
        rollout_processes=(
            args.rollout_processes
            if args.rollout_processes is not None
            else config.training.rollout_processes
        ),
        epochs=args.epochs if args.epochs is not None else config.training.epochs,
    )
    evaluation = replace(
        config.evaluation,
        interval=(
            args.evaluation_interval
            if args.evaluation_interval is not None
            else config.evaluation.interval
        ),
        episodes=(
            args.evaluation_episodes
            if args.evaluation_episodes is not None
            else config.evaluation.episodes
        ),
    )
    wandb = replace(config.wandb, mode=args.wandb_mode or config.wandb.mode)
    return replace(config, run=run, training=training, evaluation=evaluation, wandb=wandb)


def main() -> None:
    args = parse_args()
    if args.resume is not None and args.initialize_from is not None:
        raise ValueError("--resume and --initialize-from cannot be used together")
    config = apply_overrides(load_config(args.config), args)
    distillation_enabled = max(
        config.distillation.coefficient_start,
        config.distillation.coefficient_end,
    ) > 0
    if distillation_enabled and args.distill_from is None:
        raise ValueError("--distill-from is required when distillation is enabled")
    if args.distill_from is not None and not distillation_enabled:
        raise ValueError("a distillation coefficient must be positive with --distill-from")
    if distillation_enabled and (
        config.model.input_mode != "afterstate" or config.model.future_mode != "none"
    ):
        raise ValueError("distillation requires a future-free afterstate student")
    for name, value in (
        ("iterations", config.training.iterations),
        ("max_hours", config.training.max_hours),
        ("rollout_episodes", config.training.rollout_episodes),
        ("batch_size", config.training.batch_size),
        ("micro_batch_size", config.training.micro_batch_size),
        ("rollout_processes", config.training.rollout_processes),
        ("epochs", config.training.epochs),
        ("evaluation_interval", config.evaluation.interval),
        ("evaluation_episodes", config.evaluation.episodes),
    ):
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    seed_everything(config.run.seed, deterministic=config.run.deterministic)
    rng = np.random.default_rng(config.run.seed)
    device, device_info = select_device(config.run.device)

    run_name = datetime.now().strftime(f"{config.run.name_prefix}-%Y%m%d-%H%M%S")
    output_root = args.output_dir or Path(config.run.output_dir)
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    config_dict = config.to_dict()
    config_dict["device_info"] = device_info.to_dict()
    if args.initialize_from is not None:
        config_dict["initialization"] = {
            "mode": "function_preserving_widen_2x",
            "checkpoint": str(args.initialize_from),
            "split_perturbation": 0.05,
            "optimizer": "fresh",
        }
    if args.distill_from is not None:
        config_dict["distillation"]["checkpoint"] = str(args.distill_from)
        config_dict["distillation"]["target"] = "actor_policy"
        config_dict["distillation"]["direction"] = "KL(teacher || student)"
    (output_dir / "config.json").write_text(
        json.dumps(config_dict, indent=2, sort_keys=True) + "\n"
    )
    metrics_path = output_dir / "metrics.jsonl"
    tracker = WandbTracker(
        project=config.wandb.project,
        entity=config.wandb.entity,
        mode=config.wandb.mode,
        name=run_name,
        config=config_dict,
        directory=output_dir,
    )
    print(f"device: {device_info.selected} ({device_info.name})")
    print(f"W&B mode: {config.wandb.mode}, run ID: {tracker.run_id}")
    print(f"wall-clock limit: {config.training.max_hours:.3f} hours")
    print(
        "policy Phi coefficient: "
        f"{config.ppo.policy_phi_coefficient_start:g} -> "
        f"{config.ppo.policy_phi_coefficient_end:g} over "
        f"{config.ppo.policy_phi_anneal_hours:g} hours"
    )
    print(
        "distillation coefficient: "
        f"{config.distillation.coefficient_start:g} -> "
        f"{config.distillation.coefficient_end:g} over "
        f"{config.distillation.anneal_hours:g} hours"
    )

    if config.model.input_mode == "afterstate":
        model = AfterstatePpoNet(
            config.model.channels,
            config.model.residual_blocks,
            config.model.future_mode,
        )
        collect_rollout = collect_afterstate_ppo_rollout
        update_ppo = update_afterstate_ppo
    else:
        model = Ahc015PpoNet(config.model.channels, config.model.residual_blocks)
        collect_rollout = collect_ppo_rollout
        update_ppo = ppo_update
    initialization_checkpoint = None
    if args.initialize_from is not None:
        if config.model.input_mode != "afterstate":
            raise ValueError("--initialize-from supports only afterstate models")
        initialization_checkpoint = torch.load(
            args.initialize_from, map_location="cpu", weights_only=False
        )
        source_channels, source_blocks = initialize_widened_afterstate_ppo(
            model,
            initialization_checkpoint["model_state_dict"],
        )
        print(
            f"initialization: function-preserving widen from {source_channels} to "
            f"{config.model.channels} channels x {source_blocks} blocks; fresh optimizer"
        )
    teacher_actor = (
        load_distillation_teacher(args.distill_from) if args.distill_from is not None else None
    )
    if teacher_actor is not None:
        print(
            "distillation teacher: future-free actor from "
            f"{args.distill_from}; {teacher_actor.channels} channels x "
            f"{teacher_actor.residual_blocks} blocks"
        )
    print(
        f"model: {config.model.input_mode}, future={config.model.future_mode}, "
        f"{config.model.channels} channels x {config.model.residual_blocks} blocks, "
        f"{parameter_count(model.actor):,} actor parameters"
    )
    graph_input_size: tuple[int, ...] | list[tuple[int, ...]] = (
        [(1, BOARD_CHANNELS, SIDE, SIDE), (1, FUTURE_CHANNELS, FUTURE_LENGTH)]
        if config.model.future_mode != "none"
        else (1, BOARD_CHANNELS, SIDE, SIDE)
    )
    graph_svg, graph_png = render_model_graph(
        model.actor,
        input_size=graph_input_size,
        output_stem=output_dir / "model-graph",
    )
    tracker.log_image(
        graph_png,
        key="model/architecture",
        caption=f"AHC015 {config.model.input_mode} PPO actor",
    )
    print(f"actor graph: {graph_svg}")
    use_parallel_runtime = config.training.rollout_processes > 1
    model_device = torch.device("cpu") if use_parallel_runtime else device
    model = model.to(model_device)
    if teacher_actor is not None:
        teacher_actor = teacher_actor.to(model_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    update = 0
    environment_transitions = 0
    early_stop_count = 0
    start_iteration = 0
    best_score = -math.inf
    phi_schedule_elapsed_offset = 0.0
    distillation_schedule_elapsed_offset = 0.0
    if args.resume is not None:
        checkpoint = load_training_checkpoint(args.resume, model, optimizer, model_device)
        # Optimizer checkpoints also contain their old learning rate.  Keep the
        # moments, but let the new run's config control the resumed learning rate.
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = config.training.learning_rate
        start_iteration = int(checkpoint["epoch"]) + 1
        checkpoint_metrics = checkpoint.get("metrics", {})
        update = int(checkpoint_metrics.get("training/update", 0))
        environment_transitions = int(checkpoint_metrics.get("training/environment_transitions", 0))
        early_stop_count = int(checkpoint_metrics.get("training/early_stop_count", 0))
        phi_schedule_elapsed_offset = float(
            checkpoint_metrics.get("training/policy_phi_schedule_elapsed_hours", 0.0)
        )
        distillation_schedule_elapsed_offset = float(
            checkpoint_metrics.get("training/distillation_schedule_elapsed_hours", 0.0)
        )
        checkpoint_phi_coefficient = float(
            checkpoint_metrics.get(
                "training/policy_phi_coefficient",
                config.ppo.policy_phi_coefficient_start,
            )
        )
        checkpoint_is_target_policy = math.isclose(
            checkpoint_phi_coefficient,
            config.ppo.policy_phi_coefficient_end,
            abs_tol=1e-12,
        )
        if checkpoint_is_target_policy:
            best_score = float(checkpoint_metrics.get("evaluation/mean_score", -math.inf))
        if math.isfinite(best_score):
            save_checkpoint(
                output_dir / "best.pt",
                model=model.actor,
                optimizer=optimizer,
                epoch=start_iteration - 1,
                config=config_dict,
                metrics=checkpoint_metrics,
            )
            save_checkpoint(
                output_dir / "best-training.pt",
                model=model,
                optimizer=optimizer,
                epoch=start_iteration - 1,
                config=config_dict,
                metrics=checkpoint_metrics,
            )

    execution_model: torch.nn.Module = model
    evaluation_model: torch.nn.Module = model.actor
    if config.training.data_parallel:
        if device.type != "cuda" or torch.cuda.device_count() < 2:
            raise RuntimeError("data parallel training requires at least two CUDA devices")
        if use_parallel_runtime:
            parallel_backend = (
                "DDP" if config.model.input_mode == "afterstate" else "DataParallel"
            )
            print(
                f"{parallel_backend} PPO update: "
                f"{torch.cuda.device_count()} CUDA devices"
            )
        else:
            execution_model = torch.nn.DataParallel(model)
            evaluation_model = torch.nn.DataParallel(model.actor)
            if teacher_actor is not None:
                teacher_actor = torch.nn.DataParallel(teacher_actor)
            print(f"data parallel: {torch.cuda.device_count()} CUDA devices")

    parallel_runtime = None
    if config.training.rollout_processes > 1:
        if device.type != "cuda":
            raise RuntimeError("parallel rollout requires CUDA")
        if config.training.rollout_episodes % config.training.rollout_processes:
            raise ValueError("rollout episodes must be divisible by rollout processes")
        if config.evaluation.episodes % config.training.rollout_processes:
            raise ValueError("evaluation episodes must be divisible by rollout processes")
        runtime_class = (
            ParallelAfterstateRuntime
            if config.model.input_mode == "afterstate"
            else ParallelAhc015Runtime
        )
        parallel_runtime = runtime_class(
            workers=config.training.rollout_processes,
            episodes=config.training.rollout_episodes,
            channels=config.model.channels,
            residual_blocks=config.model.residual_blocks,
            seed=config.run.seed + 10_000,
        )
        print(
            f"parallel rollout/evaluation: {config.training.rollout_processes} "
            "independent CPU/GPU pipelines"
        )
        atexit.register(parallel_runtime.close)

    last_metrics: dict[str, float] = {"training/update": float(update)}
    last_iteration = start_iteration - 1
    stopped_by_time_limit = False
    log_path = Path(config.run.experiment_log)
    append_experiment_log(
        log_path,
        f"\n## {run_name}\n\n"
        f"- algorithm: PPO\n- status: started\n- output: `{output_dir}`\n"
        f"- input mode: {config.model.input_mode}\n"
        f"- future mode: {config.model.future_mode}\n"
        f"- device: {device_info.selected} ({device_info.name})\n"
        f"- seed: {config.run.seed}\n"
        f"- wall-clock limit: {config.training.max_hours:.3f} hours\n"
        f"- policy Phi coefficient: {config.ppo.policy_phi_coefficient_start:g} -> "
        f"{config.ppo.policy_phi_coefficient_end:g} over "
        f"{config.ppo.policy_phi_anneal_hours:g} hours\n"
        f"- reward mode: {config.ppo.reward_mode}\n"
        f"- distillation coefficient: {config.distillation.coefficient_start:g} -> "
        f"{config.distillation.coefficient_end:g} over "
        f"{config.distillation.anneal_hours:g} hours\n"
        f"- distillation teacher: `{args.distill_from}`\n"
        f"- Phi-greedy evaluation: {config.evaluation.phi_greedy_baseline}\n"
        f"- W&B: {config.wandb.mode}, run ID `{tracker.run_id}`\n",
    )

    if initialization_checkpoint is not None:
        initial_metrics = evaluate(
            evaluation_model,
            model,
            device,
            config,
            parallel_runtime,
            config.ppo.policy_phi_coefficient_start,
        )
        best_score = initial_metrics["evaluation/mean_score"]
        initial_metrics.update(
            {
                "training/update": 0.0,
                "training/environment_transitions": 0.0,
                "training/early_stop_count": 0.0,
                "training/policy_phi_coefficient": config.ppo.policy_phi_coefficient_start,
                "training/policy_phi_schedule_elapsed_hours": 0.0,
                "initialization/source_epoch": float(initialization_checkpoint.get("epoch", -1)),
            }
        )
        save_checkpoint(
            output_dir / "best.pt",
            model=model.actor,
            optimizer=optimizer,
            epoch=-1,
            config=config_dict,
            metrics=initial_metrics,
        )
        save_checkpoint(
            output_dir / "best-training.pt",
            model=model,
            optimizer=optimizer,
            epoch=-1,
            config=config_dict,
            metrics=initial_metrics,
        )
        print(
            f"initial widened model mean score: {best_score:.3f} "
            f"({config.evaluation.episodes} fixed cases)"
        )
        append_experiment_log(
            log_path,
            f"- initialization: function-preserving 2x channel widening from "
            f"`{args.initialize_from}`; optimizer state reset\n"
            f"- initial widened mean score: {best_score:.3f}\n",
        )

    training_started = time.monotonic()
    deadline = training_started + config.training.max_hours * 3600
    for iteration in range(start_iteration, config.training.iterations):
        if time.monotonic() >= deadline:
            stopped_by_time_limit = True
            break
        iteration_started = time.monotonic()
        phi_schedule_elapsed_hours = (
            phi_schedule_elapsed_offset + (iteration_started - training_started) / 3600
        )
        policy_phi_coefficient = scheduled_policy_phi_coefficient(
            config.ppo,
            phi_schedule_elapsed_hours,
        )
        distillation_schedule_elapsed_hours = (
            distillation_schedule_elapsed_offset + (iteration_started - training_started) / 3600
        )
        distillation_coefficient = scheduled_coefficient(
            config.distillation.coefficient_start,
            config.distillation.coefficient_end,
            config.distillation.anneal_hours,
            distillation_schedule_elapsed_hours,
        )
        rollout_started = time.monotonic()
        if parallel_runtime is None:
            collect_kwargs = dict(
                gamma=config.ppo.gamma,
                gae_lambda=config.ppo.gae_lambda,
                logit_scale=config.ppo.logit_scale,
                inference_batch_size=config.training.inference_batch_size,
            )
            if config.model.input_mode == "afterstate":
                collect_kwargs["policy_phi_coefficient"] = policy_phi_coefficient
                collect_kwargs["reward_mode"] = config.ppo.reward_mode
                collect_kwargs["future_mode"] = config.model.future_mode
            rollout, _, rollout_metrics = collect_rollout(
                execution_model,
                device,
                config.training.rollout_episodes,
                rng,
                **collect_kwargs,
            )
        else:
            parallel_collect_kwargs = dict(
                gamma=config.ppo.gamma,
                gae_lambda=config.ppo.gae_lambda,
                logit_scale=config.ppo.logit_scale,
                inference_batch_size=config.training.inference_batch_size,
            )
            if config.model.input_mode == "afterstate":
                parallel_collect_kwargs.update(
                    policy_phi_coefficient=policy_phi_coefficient,
                    reward_mode=config.ppo.reward_mode,
                    future_mode=config.model.future_mode,
                )
            rollout, _, rollout_metrics = parallel_runtime.collect(
                model, **parallel_collect_kwargs
            )
        environment_transitions += len(rollout)
        metrics: dict[str, float] = {
            "iteration": float(iteration),
            "timing/rollout_seconds": time.monotonic() - rollout_started,
            **rollout_metrics,
        }
        optimization_started = time.monotonic()
        if parallel_runtime is None:
            update_kwargs = dict(
                epochs=config.training.epochs,
                batch_size=config.training.batch_size,
                micro_batch_size=config.training.micro_batch_size,
                clip_ratio=config.ppo.clip_ratio,
                value_clip=config.ppo.value_clip,
                value_coefficient=config.ppo.value_coefficient,
                entropy_coefficient=config.ppo.entropy_coefficient,
                gradient_clip_norm=config.training.gradient_clip_norm,
                logit_scale=config.ppo.logit_scale,
                target_kl=config.ppo.target_kl,
            )
            if config.model.input_mode == "afterstate":
                update_kwargs["policy_phi_coefficient"] = policy_phi_coefficient
                update_kwargs["teacher_actor"] = teacher_actor
                update_kwargs["distillation_coefficient"] = distillation_coefficient
            update_metrics = update_ppo(
                execution_model,
                optimizer,
                rollout,
                device,
                rng,
                **update_kwargs,
            )
        else:
            parallel_update_kwargs = dict(
                epochs=config.training.epochs,
                batch_size=config.training.batch_size,
                micro_batch_size=config.training.micro_batch_size,
                learning_rate=config.training.learning_rate,
                weight_decay=config.training.weight_decay,
                clip_ratio=config.ppo.clip_ratio,
                value_clip=config.ppo.value_clip,
                value_coefficient=config.ppo.value_coefficient,
                entropy_coefficient=config.ppo.entropy_coefficient,
                gradient_clip_norm=config.training.gradient_clip_norm,
                logit_scale=config.ppo.logit_scale,
                target_kl=config.ppo.target_kl,
                data_parallel=config.training.data_parallel,
            )
            if config.model.input_mode == "afterstate":
                parallel_update_kwargs.update(
                    teacher_actor=teacher_actor,
                    distillation_coefficient=distillation_coefficient,
                )
            update_metrics = parallel_runtime.update(
                model, optimizer, **parallel_update_kwargs
            )
        update += int(update_metrics["training/updates_this_iteration"])
        early_stop_count += int(update_metrics["training/early_stop"])
        metrics.update(update_metrics)
        metrics["training/update"] = float(update)
        metrics["training/environment_transitions"] = float(environment_transitions)
        metrics["training/learning_rate"] = config.training.learning_rate
        metrics["training/policy_phi_coefficient"] = policy_phi_coefficient
        metrics["training/policy_phi_schedule_elapsed_hours"] = phi_schedule_elapsed_hours
        metrics["training/distillation_coefficient"] = distillation_coefficient
        metrics["training/distillation_schedule_elapsed_hours"] = (
            distillation_schedule_elapsed_hours
        )
        metrics["training/micro_batch_size"] = float(config.training.micro_batch_size)
        metrics["training/early_stop_count"] = float(early_stop_count)
        metrics["training/early_stop_rate"] = early_stop_count / (iteration + 1)
        metrics["timing/optimization_seconds"] = time.monotonic() - optimization_started

        should_evaluate = (iteration + 1) % config.evaluation.interval == 0
        if should_evaluate:
            evaluation_started = time.monotonic()
            metrics.update(
                evaluate(
                    evaluation_model,
                    model,
                    device,
                    config,
                    parallel_runtime,
                    policy_phi_coefficient,
                )
            )
            metrics["timing/evaluation_seconds"] = time.monotonic() - evaluation_started
            score = metrics["evaluation/mean_score"]
            target_policy_reached = math.isclose(
                policy_phi_coefficient,
                config.ppo.policy_phi_coefficient_end,
                abs_tol=1e-12,
            )
            target_distillation_reached = math.isclose(
                distillation_coefficient,
                config.distillation.coefficient_end,
                abs_tol=1e-12,
            )
            if target_policy_reached and target_distillation_reached and score > best_score:
                best_score = score
                save_checkpoint(
                    output_dir / "best.pt",
                    model=model.actor,
                    optimizer=optimizer,
                    epoch=iteration,
                    config=config_dict,
                    metrics=metrics,
                )
                save_checkpoint(
                    output_dir / "best-training.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=iteration,
                    config=config_dict,
                    metrics=metrics,
                )

        elapsed_seconds = time.monotonic() - training_started
        metrics["timing/iteration_seconds"] = time.monotonic() - iteration_started
        metrics["timing/elapsed_hours"] = elapsed_seconds / 3600
        metrics["timing/remaining_hours"] = max(0.0, (deadline - time.monotonic()) / 3600)
        if (iteration + 1) % config.training.checkpoint_interval == 0:
            save_checkpoint(
                output_dir / "last.pt",
                model=model,
                optimizer=optimizer,
                epoch=iteration,
                config=config_dict,
                metrics=metrics,
            )
        with metrics_path.open("a") as file:
            file.write(json.dumps(metrics, sort_keys=True) + "\n")
        tracker.log(metrics, step=iteration)
        print(json.dumps(metrics, sort_keys=True), flush=True)
        last_metrics = metrics
        last_iteration = iteration

    if "evaluation/mean_score" not in last_metrics:
        print("running final evaluation", flush=True)
        final_schedule_elapsed_hours = (
            phi_schedule_elapsed_offset + (time.monotonic() - training_started) / 3600
        )
        final_policy_phi_coefficient = scheduled_policy_phi_coefficient(
            config.ppo,
            final_schedule_elapsed_hours,
        )
        last_metrics["training/policy_phi_coefficient"] = final_policy_phi_coefficient
        last_metrics["training/policy_phi_schedule_elapsed_hours"] = final_schedule_elapsed_hours
        last_metrics.update(
            evaluate(
                evaluation_model,
                model,
                device,
                config,
                parallel_runtime,
                final_policy_phi_coefficient,
            )
        )
        score = last_metrics["evaluation/mean_score"]
        if score > best_score:
            best_score = score
            save_checkpoint(
                output_dir / "best.pt",
                model=model.actor,
                optimizer=optimizer,
                epoch=last_iteration,
                config=config_dict,
                metrics=last_metrics,
            )
            save_checkpoint(
                output_dir / "best-training.pt",
                model=model,
                optimizer=optimizer,
                epoch=last_iteration,
                config=config_dict,
                metrics=last_metrics,
            )
        final_metrics = {
            "final/mean_score": last_metrics["evaluation/mean_score"],
            "final/score_se": last_metrics["evaluation/score_se"],
        }
        for name in ("paired_gain", "paired_gain_se", "win_rate"):
            key = f"evaluation/{name}"
            if key in last_metrics:
                final_metrics[f"final/{name}"] = last_metrics[key]
        tracker.log(final_metrics, step=max(last_iteration + 1, 0))

    if parallel_runtime is not None:
        parallel_runtime.close()
        atexit.unregister(parallel_runtime.close)

    save_checkpoint(
        output_dir / "last.pt",
        model=model,
        optimizer=optimizer,
        epoch=last_iteration,
        config=config_dict,
        metrics={
            "training/update": float(update),
            "training/environment_transitions": float(environment_transitions),
            "training/early_stop_count": float(early_stop_count),
            "training/policy_phi_coefficient": float(
                last_metrics.get(
                    "training/policy_phi_coefficient",
                    config.ppo.policy_phi_coefficient_end,
                )
            ),
            "training/policy_phi_schedule_elapsed_hours": float(
                last_metrics.get(
                    "training/policy_phi_schedule_elapsed_hours",
                    phi_schedule_elapsed_offset + (time.monotonic() - training_started) / 3600,
                )
            ),
            "training/distillation_coefficient": float(
                last_metrics.get(
                    "training/distillation_coefficient",
                    config.distillation.coefficient_end,
                )
            ),
            "training/distillation_schedule_elapsed_hours": float(
                last_metrics.get(
                    "training/distillation_schedule_elapsed_hours",
                    distillation_schedule_elapsed_offset
                    + (time.monotonic() - training_started) / 3600,
                )
            ),
            "evaluation/best_mean_score": best_score,
        },
    )
    if not (output_dir / "best.pt").exists():
        best_score = float(last_metrics.get("evaluation/mean_score", -math.inf))
        save_checkpoint(
            output_dir / "best.pt",
            model=model.actor,
            optimizer=optimizer,
            epoch=last_iteration,
            config=config_dict,
            metrics=last_metrics,
        )
        save_checkpoint(
            output_dir / "best-training.pt",
            model=model,
            optimizer=optimizer,
            epoch=last_iteration,
            config=config_dict,
            metrics=last_metrics,
        )
    load_checkpoint(output_dir / "best.pt", model=model.actor, map_location=model_device)
    tracker.log_artifact(
        output_dir / "best.pt", name=f"{run_name}-actor-checkpoint", artifact_type="model"
    )
    tracker.log_artifact(
        output_dir / "best-training.pt",
        name=f"{run_name}-training-checkpoint",
        artifact_type="model",
    )
    tracker.finish()
    append_experiment_log(
        log_path,
        f"- status: {'time limit reached' if stopped_by_time_limit else 'completed'}\n"
        f"- elapsed: {(time.monotonic() - training_started) / 3600:.3f} hours\n"
        f"- updates: {update}\n- best mean score: {best_score:.3f}\n",
    )


if __name__ == "__main__":
    main()
