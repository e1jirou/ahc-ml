from __future__ import annotations

import argparse
import copy
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

from .backup import bellman_residual_targets
from .config import Ahc015Config, load_config
from .features import FEATURE_CHANNELS, encode_afterstates
from .game import SIDE
from .model import PARAMETER_COUNT, Ahc015ValueNet, parameter_count
from .replay import ReplayBuffer
from .simulation import collect_rollouts, evaluate_policy, generate_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the AHC015 afterstate value model")
    parser.add_argument("--config", type=Path, default=Path("examples/ahc015/config.toml"))
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--max-hours", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--rollout-episodes", type=int)
    parser.add_argument("--updates-per-iteration", type=int)
    parser.add_argument("--target-update-interval", type=int)
    parser.add_argument("--evaluation-episodes", type=int)
    parser.add_argument("--replay-minimum-size", type=int)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"))
    parser.add_argument("--experiment-log", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def linear_schedule(start: float, end: float, step: int, duration: int) -> float:
    if duration <= 0:
        return end
    fraction = min(max(step / duration, 0.0), 1.0)
    return start + fraction * (end - start)


def append_experiment_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as file:
        file.write(text.rstrip() + "\n")


def evaluate(
    model: torch.nn.Module,
    device: torch.device,
    config: Ahc015Config,
) -> dict[str, float]:
    flavors, ranks = generate_cases(config.evaluation.episodes, config.evaluation.seed)
    greedy = evaluate_policy(
        None,
        device,
        flavors,
        ranks,
        inference_batch_size=config.backup.inference_batch_size,
    )
    learned = evaluate_policy(
        model,
        device,
        flavors,
        ranks,
        inference_batch_size=config.backup.inference_batch_size,
    )
    difference = learned.scores - greedy.scores
    return {
        "evaluation/mean_score": float(learned.scores.mean()),
        "evaluation/greedy_mean_score": float(greedy.scores.mean()),
        "evaluation/paired_gain": float(difference.mean()),
        "evaluation/paired_gain_se": float(difference.std(ddof=1) / math.sqrt(len(difference))),
        "evaluation/win_rate": float(np.mean(difference > 0)),
    }


def export_model(model: torch.nn.Module, output_dir: Path) -> None:
    metadata = {
        "architecture": "ahc015-afterstate-value-144x9-v1",
        "channels": 144,
        "residual_blocks": 9,
        "parameter_count": PARAMETER_COUNT,
    }
    export_state_dict(model.state_dict(), output_dir / "model.bin", metadata=metadata)
    export_quantized_state_dict(
        model.state_dict(),
        output_dir / "model.q8.bin",
        metadata=metadata,
        rust_source=output_dir / "model_data.rs",
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run = replace(
        config.run,
        device=args.device or config.run.device,
        experiment_log=(
            str(args.experiment_log) if args.experiment_log else config.run.experiment_log
        ),
    )
    training = replace(
        config.training,
        iterations=(args.iterations if args.iterations is not None else config.training.iterations),
        max_hours=(args.max_hours if args.max_hours is not None else config.training.max_hours),
        batch_size=(args.batch_size if args.batch_size is not None else config.training.batch_size),
        rollout_episodes=(
            args.rollout_episodes
            if args.rollout_episodes is not None
            else config.training.rollout_episodes
        ),
        updates_per_iteration=(
            args.updates_per_iteration
            if args.updates_per_iteration is not None
            else config.training.updates_per_iteration
        ),
        target_update_interval=(
            args.target_update_interval
            if args.target_update_interval is not None
            else config.training.target_update_interval
        ),
    )
    replay_config = replace(
        config.replay,
        minimum_size=(
            args.replay_minimum_size
            if args.replay_minimum_size is not None
            else config.replay.minimum_size
        ),
    )
    evaluation = replace(
        config.evaluation,
        episodes=(
            args.evaluation_episodes
            if args.evaluation_episodes is not None
            else config.evaluation.episodes
        ),
    )
    wandb = replace(
        config.wandb,
        mode=args.wandb_mode or config.wandb.mode,
    )
    config = replace(
        config,
        run=run,
        training=training,
        replay=replay_config,
        evaluation=evaluation,
        wandb=wandb,
    )
    for name, value in (
        ("iterations", config.training.iterations),
        ("max_hours", config.training.max_hours),
        ("batch_size", config.training.batch_size),
        ("rollout_episodes", config.training.rollout_episodes),
        ("updates_per_iteration", config.training.updates_per_iteration),
        ("target_update_interval", config.training.target_update_interval),
        ("evaluation_episodes", config.evaluation.episodes),
        ("replay_minimum_size", config.replay.minimum_size),
    ):
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if config.replay.minimum_size > config.replay.capacity:
        raise ValueError("replay minimum size must not exceed its capacity")
    seed_everything(config.run.seed, deterministic=config.run.deterministic)
    rng = np.random.default_rng(config.run.seed)
    device, device_info = select_device(config.run.device)

    run_name = datetime.now().strftime("afterstate-%Y%m%d-%H%M%S")
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

    model = Ahc015ValueNet()
    if parameter_count(model) != PARAMETER_COUNT:
        raise RuntimeError("AHC015 model parameter count changed unexpectedly")
    graph_svg, graph_png = render_model_graph(
        model,
        input_size=(1, FEATURE_CHANNELS, SIDE, SIDE),
        output_stem=output_dir / "model-graph",
    )
    tracker.log_image(
        graph_png,
        key="model/architecture",
        caption="AHC015 afterstate value network architecture",
    )
    print(f"model graph: {graph_svg}")
    model = model.to(device)
    target_model = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    update = 0
    start_iteration = 0
    if args.resume is not None:
        checkpoint = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            map_location=device,
        )
        target_model.load_state_dict(model.state_dict())
        start_iteration = int(checkpoint["epoch"]) + 1
        update = int(checkpoint.get("metrics", {}).get("training/update", 0))

    replay = ReplayBuffer(config.replay.capacity)
    best_gain = -math.inf
    last_metrics: dict[str, float] = {"training/update": float(update)}
    last_iteration = start_iteration - 1
    training_started = time.monotonic()
    deadline = training_started + config.training.max_hours * 60 * 60
    stopped_by_time_limit = False
    log_path = Path(config.run.experiment_log)
    append_experiment_log(
        log_path,
        f"\n## {run_name}\n\n"
        f"- status: started\n- output: `{output_dir}`\n"
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
        epsilon = linear_schedule(
            config.exploration.epsilon_start,
            config.exploration.epsilon_end,
            iteration,
            config.exploration.epsilon_decay_iterations,
        )
        rollout_started = time.monotonic()
        rollout, rollout_result = collect_rollouts(
            model,
            device,
            config.training.rollout_episodes,
            rng,
            epsilon=epsilon,
            inference_batch_size=config.backup.inference_batch_size,
        )
        rollout_seconds = time.monotonic() - rollout_started
        replay.add(rollout)
        metrics: dict[str, float] = {
            "iteration": float(iteration),
            "rollout/epsilon": epsilon,
            "rollout/mean_score": float(rollout_result.scores.mean()),
            "replay/size": float(len(replay)),
            "timing/rollout_seconds": rollout_seconds,
        }

        losses = []
        gradient_norms = []
        target_seconds = 0.0
        optimization_seconds = 0.0
        if len(replay) >= config.replay.minimum_size:
            minimum_placed = round(
                linear_schedule(
                    config.backup.minimum_placed_start,
                    1,
                    iteration,
                    config.backup.curriculum_iterations,
                )
            )
            mc_beta = linear_schedule(
                config.backup.mc_beta_start,
                config.backup.mc_beta_end,
                iteration,
                config.backup.mc_beta_decay_iterations,
            )
            for _ in range(config.training.updates_per_iteration):
                if time.monotonic() >= deadline:
                    stopped_by_time_limit = True
                    break
                batch = replay.sample(
                    config.training.batch_size,
                    rng,
                    minimum_placed=minimum_placed,
                )
                target_started = time.monotonic()
                targets = bellman_residual_targets(
                    batch,
                    model,
                    target_model,
                    device,
                    rng,
                    placement_samples=config.backup.placement_samples,
                    enumerate_threshold=config.backup.enumerate_threshold,
                    inference_batch_size=config.backup.inference_batch_size,
                )
                has_mc = np.isfinite(batch.mc_targets)
                targets[has_mc] = (
                    mc_beta * batch.mc_targets[has_mc] + (1 - mc_beta) * targets[has_mc]
                )
                target_seconds += time.monotonic() - target_started
                optimization_started = time.monotonic()
                features = encode_afterstates(
                    batch.boards.reshape(-1, SIDE, SIDE),
                    batch.actions,
                    batch.placed,
                    batch.flavors,
                )
                inputs = torch.from_numpy(features).to(device)
                target_tensor = torch.from_numpy(targets).to(device)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                predictions = model(inputs)
                loss = torch.nn.functional.smooth_l1_loss(predictions, target_tensor)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.training.gradient_clip_norm
                )
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                gradient_norms.append(float(gradient_norm.detach().cpu()))
                optimization_seconds += time.monotonic() - optimization_started
                update += 1
                if update % config.training.target_update_interval == 0:
                    target_model.load_state_dict(model.state_dict())
            if losses:
                metrics["training/loss"] = float(np.mean(losses))
                metrics["training/gradient_norm"] = float(np.mean(gradient_norms))
                metrics["training/update"] = float(update)
                metrics["training/updates_this_iteration"] = float(len(losses))
                metrics["training/minimum_placed"] = float(minimum_placed)
                metrics["training/mc_beta"] = mc_beta
                metrics["timing/target_seconds"] = target_seconds
                metrics["timing/optimization_seconds"] = optimization_seconds

        should_evaluate = (
            iteration + 1
        ) % config.evaluation.interval == 0 and not stopped_by_time_limit
        if should_evaluate:
            metrics.update(evaluate(model, device, config))
            gain = metrics["evaluation/paired_gain"]
            if gain > best_gain:
                best_gain = gain
                save_checkpoint(
                    output_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=iteration,
                    config=config_dict,
                    metrics=metrics,
                )

        if (iteration + 1) % config.training.checkpoint_interval == 0 or stopped_by_time_limit:
            save_checkpoint(
                output_dir / "last.pt",
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
        with metrics_path.open("a") as file:
            file.write(json.dumps(metrics, sort_keys=True) + "\n")
        tracker.log(metrics, step=iteration)
        print(json.dumps(metrics, sort_keys=True), flush=True)
        last_metrics = metrics
        last_iteration = iteration
        if stopped_by_time_limit:
            break

    if "evaluation/paired_gain" not in last_metrics:
        print("running final paired evaluation", flush=True)
        last_metrics.update(evaluate(model, device, config))
        gain = last_metrics["evaluation/paired_gain"]
        if gain > best_gain:
            best_gain = gain
            save_checkpoint(
                output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=last_iteration,
                config=config_dict,
                metrics=last_metrics,
            )
        tracker.log(
            {
                "final/mean_score": last_metrics["evaluation/mean_score"],
                "final/greedy_mean_score": last_metrics["evaluation/greedy_mean_score"],
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
        metrics={"training/update": float(update), "evaluation/best_paired_gain": best_gain},
    )
    load_checkpoint(output_dir / "best.pt", model=model, map_location=device)
    export_model(model, output_dir)
    tracker.log_artifact(
        output_dir / "best.pt", name=f"{run_name}-checkpoint", artifact_type="model"
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
