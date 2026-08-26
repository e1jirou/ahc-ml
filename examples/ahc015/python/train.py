from __future__ import annotations

import argparse
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
from ahc_ml.export import export_quantized_state_dict, export_state_dict
from ahc_ml.seed import seed_everything
from ahc_ml.tracking import WandbTracker
from ahc_ml.visualization import render_model_graph

from .config import Ahc015Config, load_config
from .features import BOARD_CHANNELS, FUTURE_CHANNELS, FUTURE_LENGTH
from .game import SIDE
from .model import STUDENT_CHANNELS, STUDENT_RESIDUAL_BLOCKS, Ahc015PpoNet, parameter_count
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
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--evaluation-episodes", type=int)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"))
    parser.add_argument("--experiment-log", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path, help="resume from a full PPO training checkpoint")
    return parser.parse_args()


def append_experiment_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as file:
        file.write(text.rstrip() + "\n")


def evaluate(
    model: torch.nn.Module, device: torch.device, config: Ahc015Config
) -> dict[str, float]:
    flavors, ranks = generate_cases(config.evaluation.episodes, config.evaluation.seed)
    greedy = evaluate_policy(
        None,
        device,
        flavors,
        ranks,
        inference_batch_size=config.training.inference_batch_size,
    )
    learned = evaluate_policy(
        model,
        device,
        flavors,
        ranks,
        inference_batch_size=config.training.inference_batch_size,
    )
    difference = learned.scores - greedy.scores
    return {
        "evaluation/mean_score": float(learned.scores.mean()),
        "evaluation/greedy_mean_score": float(greedy.scores.mean()),
        "evaluation/paired_gain": float(difference.mean()),
        "evaluation/paired_gain_se": float(difference.std(ddof=1) / math.sqrt(len(difference))),
        "evaluation/win_rate": float(np.mean(difference > 0)),
    }


def export_actor(
    model: torch.nn.Module,
    output_dir: Path,
    *,
    channels: int,
    residual_blocks: int,
) -> None:
    parameters = parameter_count(model)
    metadata = {
        "architecture": f"ahc015-ppo-actor-{channels}x{residual_blocks}-film-v3",
        "training_algorithm": "ppo",
        "channels": channels,
        "residual_blocks": residual_blocks,
        "parameter_count": parameters,
    }
    export_state_dict(model.state_dict(), output_dir / "model.bin", metadata=metadata)
    export_quantized_state_dict(
        model.state_dict(),
        output_dir / "model.q8.bin",
        metadata=metadata,
        rust_source=output_dir / "model_data.rs",
    )


def load_training_checkpoint(
    path: Path,
    model: Ahc015PpoNet,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, object]:
    return load_checkpoint(path, model=model, optimizer=optimizer, map_location=device)


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
        epochs=args.epochs if args.epochs is not None else config.training.epochs,
    )
    evaluation = replace(
        config.evaluation,
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
    config = apply_overrides(load_config(args.config), args)
    for name, value in (
        ("iterations", config.training.iterations),
        ("max_hours", config.training.max_hours),
        ("rollout_episodes", config.training.rollout_episodes),
        ("batch_size", config.training.batch_size),
        ("epochs", config.training.epochs),
        ("evaluation_episodes", config.evaluation.episodes),
    ):
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    seed_everything(config.run.seed, deterministic=config.run.deterministic)
    rng = np.random.default_rng(config.run.seed)
    device, device_info = select_device(config.run.device)

    run_name = datetime.now().strftime("ppo-%Y%m%d-%H%M%S")
    output_root = args.output_dir or Path(config.run.output_dir)
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    config_dict = config.to_dict()
    config_dict["device_info"] = device_info.to_dict()
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

    model = Ahc015PpoNet(config.model.channels, config.model.residual_blocks)
    print(
        f"model: {config.model.channels} channels x "
        f"{config.model.residual_blocks} blocks, {parameter_count(model.actor):,} actor parameters"
    )
    graph_svg, graph_png = render_model_graph(
        model.actor,
        input_size=[
            (1, BOARD_CHANNELS, SIDE, SIDE),
            (1, FUTURE_CHANNELS, FUTURE_LENGTH),
        ],
        output_stem=output_dir / "model-graph",
    )
    tracker.log_image(
        graph_png,
        key="model/architecture",
        caption="AHC015 PPO actor (the critic has the same architecture)",
    )
    print(f"actor graph: {graph_svg}")
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    update = 0
    environment_transitions = 0
    early_stop_count = 0
    start_iteration = 0
    best_gain = -math.inf
    if args.resume is not None:
        checkpoint = load_training_checkpoint(args.resume, model, optimizer, device)
        # Optimizer checkpoints also contain their old learning rate.  Keep the
        # moments, but let the new run's config control the resumed learning rate.
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = config.training.learning_rate
        start_iteration = int(checkpoint["epoch"]) + 1
        checkpoint_metrics = checkpoint.get("metrics", {})
        update = int(checkpoint_metrics.get("training/update", 0))
        environment_transitions = int(checkpoint_metrics.get("training/environment_transitions", 0))
        early_stop_count = int(checkpoint_metrics.get("training/early_stop_count", 0))
        best_gain = float(checkpoint_metrics.get("evaluation/paired_gain", -math.inf))
        if math.isfinite(best_gain):
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

    last_metrics: dict[str, float] = {"training/update": float(update)}
    last_iteration = start_iteration - 1
    training_started = time.monotonic()
    deadline = training_started + config.training.max_hours * 3600
    stopped_by_time_limit = False
    log_path = Path(config.run.experiment_log)
    append_experiment_log(
        log_path,
        f"\n## {run_name}\n\n"
        f"- algorithm: PPO\n- status: started\n- output: `{output_dir}`\n"
        f"- device: {device_info.selected} ({device_info.name})\n"
        f"- seed: {config.run.seed}\n"
        f"- wall-clock limit: {config.training.max_hours:.3f} hours\n"
        f"- W&B: {config.wandb.mode}, run ID `{tracker.run_id}`\n",
    )

    for iteration in range(start_iteration, config.training.iterations):
        if time.monotonic() >= deadline:
            stopped_by_time_limit = True
            break
        iteration_started = time.monotonic()
        rollout_started = time.monotonic()
        rollout, _, rollout_metrics = collect_ppo_rollout(
            model,
            device,
            config.training.rollout_episodes,
            rng,
            gamma=config.ppo.gamma,
            gae_lambda=config.ppo.gae_lambda,
            logit_scale=config.ppo.logit_scale,
            inference_batch_size=config.training.inference_batch_size,
        )
        environment_transitions += len(rollout)
        metrics: dict[str, float] = {
            "iteration": float(iteration),
            "timing/rollout_seconds": time.monotonic() - rollout_started,
            **rollout_metrics,
        }
        optimization_started = time.monotonic()
        update_metrics = ppo_update(
            model,
            optimizer,
            rollout,
            device,
            rng,
            epochs=config.training.epochs,
            batch_size=config.training.batch_size,
            clip_ratio=config.ppo.clip_ratio,
            value_clip=config.ppo.value_clip,
            value_coefficient=config.ppo.value_coefficient,
            entropy_coefficient=config.ppo.entropy_coefficient,
            gradient_clip_norm=config.training.gradient_clip_norm,
            logit_scale=config.ppo.logit_scale,
            target_kl=config.ppo.target_kl,
        )
        update += int(update_metrics["training/updates_this_iteration"])
        early_stop_count += int(update_metrics["training/early_stop"])
        metrics.update(update_metrics)
        metrics["training/update"] = float(update)
        metrics["training/environment_transitions"] = float(environment_transitions)
        metrics["training/learning_rate"] = config.training.learning_rate
        metrics["training/early_stop_count"] = float(early_stop_count)
        metrics["training/early_stop_rate"] = early_stop_count / (iteration + 1)
        metrics["timing/optimization_seconds"] = time.monotonic() - optimization_started

        should_evaluate = (iteration + 1) % config.evaluation.interval == 0
        if should_evaluate:
            metrics.update(evaluate(model.actor, device, config))
            gain = metrics["evaluation/paired_gain"]
            if gain > best_gain:
                best_gain = gain
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

    if "evaluation/paired_gain" not in last_metrics:
        print("running final paired evaluation", flush=True)
        last_metrics.update(evaluate(model.actor, device, config))
        gain = last_metrics["evaluation/paired_gain"]
        if gain > best_gain:
            best_gain = gain
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
        tracker.log(
            {
                "final/mean_score": last_metrics["evaluation/mean_score"],
                "final/paired_gain": last_metrics["evaluation/paired_gain"],
                "final/paired_gain_se": last_metrics["evaluation/paired_gain_se"],
                "final/win_rate": last_metrics["evaluation/win_rate"],
            },
            step=max(last_iteration + 1, 0),
        )

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
            "evaluation/best_paired_gain": best_gain,
        },
    )
    load_checkpoint(output_dir / "best.pt", model=model.actor, map_location=device)
    tracker.log_artifact(
        output_dir / "best.pt", name=f"{run_name}-actor-checkpoint", artifact_type="model"
    )
    tracker.log_artifact(
        output_dir / "best-training.pt",
        name=f"{run_name}-training-checkpoint",
        artifact_type="model",
    )
    if (
        config.model.channels == STUDENT_CHANNELS
        and config.model.residual_blocks == STUDENT_RESIDUAL_BLOCKS
    ):
        export_actor(
            model.actor,
            output_dir,
            channels=config.model.channels,
            residual_blocks=config.model.residual_blocks,
        )
        tracker.log_artifact(
            output_dir / "model.bin", name=f"{run_name}-rust-weights", artifact_type="model"
        )
        tracker.log_artifact(
            output_dir / "model.q8.bin",
            name=f"{run_name}-rust-weights-q8",
            artifact_type="model",
        )
    tracker.finish()
    append_experiment_log(
        log_path,
        f"- status: {'time limit reached' if stopped_by_time_limit else 'completed'}\n"
        f"- elapsed: {(time.monotonic() - training_started) / 3600:.3f} hours\n"
        f"- updates: {update}\n- best paired gain: {best_gain:.3f}\n",
    )


if __name__ == "__main__":
    main()
