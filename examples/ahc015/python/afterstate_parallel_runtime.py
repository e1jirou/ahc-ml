from __future__ import annotations

import io
import math
import multiprocessing as mp
import os
import tempfile
import time
import traceback
from dataclasses import dataclass
from datetime import timedelta
from multiprocessing.connection import Connection
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from numpy.typing import NDArray
from torch.nn.parallel import DistributedDataParallel

from .afterstate_model import AfterstatePpoNet, AfterstateValueNet
from .afterstate_ppo import (
    AfterstatePpoRollout,
    AfterstatePpoRolloutStorage,
    collect_afterstate_ppo_rollout,
    update_afterstate_ppo,
)
from .afterstate_simulation import evaluate_afterstate_policy
from .features import BOARD_CHANNELS
from .game import ACTION_COUNT, CELL_COUNT, SIDE
from .simulation import EvaluationResult


@dataclass(frozen=True, slots=True)
class _SharedArray:
    buffer: Any
    shape: tuple[int, ...]
    dtype: str

    def array(self) -> np.ndarray:
        return np.frombuffer(self.buffer, dtype=np.dtype(self.dtype)).reshape(self.shape)


@dataclass(frozen=True, slots=True)
class _SharedAfterstateRollout:
    board_features: _SharedArray
    actions: _SharedArray
    old_log_probs: _SharedArray
    old_values: _SharedArray
    advantages: _SharedArray
    returns: _SharedArray

    def worker_storage(self, worker: int) -> AfterstatePpoRolloutStorage:
        return AfterstatePpoRolloutStorage(
            board_features=self.board_features.array()[worker],
            future_features=None,
            candidate_potentials=None,
            actions=self.actions.array()[worker],
            old_log_probs=self.old_log_probs.array()[worker],
            old_values=self.old_values.array()[worker],
            advantages=self.advantages.array()[worker],
            returns=self.returns.array()[worker],
        )

    def as_rollout(self) -> AfterstatePpoRollout:
        transitions = self.actions.array().size
        return AfterstatePpoRollout(
            board_features=self.board_features.array().reshape(
                transitions, ACTION_COUNT, BOARD_CHANNELS, SIDE, SIDE
            ),
            future_features=None,
            candidate_potentials=None,
            actions=self.actions.array().reshape(transitions),
            old_log_probs=self.old_log_probs.array().reshape(transitions),
            old_values=self.old_values.array().reshape(transitions),
            advantages=self.advantages.array().reshape(transitions),
            returns=self.returns.array().reshape(transitions),
        )


def _shared_array(context: Any, shape: tuple[int, ...], dtype: np.dtype[Any]) -> _SharedArray:
    numpy_dtype = np.dtype(dtype)
    typecodes = {
        np.dtype(np.uint8): "B",
        np.dtype(np.float32): "f",
        np.dtype(np.int64): "q",
    }
    return _SharedArray(
        context.RawArray(typecodes[numpy_dtype], math.prod(shape)),
        shape,
        numpy_dtype.str,
    )


def _create_shared_rollout(
    context: Any, workers: int, episodes_per_worker: int
) -> _SharedAfterstateRollout:
    prefix = (workers, CELL_COUNT - 1, episodes_per_worker)
    return _SharedAfterstateRollout(
        board_features=_shared_array(
            context,
            (*prefix, ACTION_COUNT, BOARD_CHANNELS, SIDE, SIDE),
            np.dtype(np.uint8),
        ),
        actions=_shared_array(context, prefix, np.dtype(np.int64)),
        old_log_probs=_shared_array(context, prefix, np.dtype(np.float32)),
        old_values=_shared_array(context, prefix, np.dtype(np.float32)),
        advantages=_shared_array(context, prefix, np.dtype(np.float32)),
        returns=_shared_array(context, prefix, np.dtype(np.float32)),
    )


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _load_bytes(data: bytes) -> dict[str, Any]:
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)


def _receive_worker_responses(
    connections: list[Connection],
    processes: list[mp.Process],
    *,
    timeout_seconds: float = 1800.0,
) -> list[tuple[Any, ...]]:
    responses: list[tuple[Any, ...] | None] = [None] * len(connections)
    pending = set(range(len(connections)))
    deadline = time.monotonic() + timeout_seconds
    while pending:
        for index in list(pending):
            if connections[index].poll():
                try:
                    response = connections[index].recv()
                except EOFError as error:
                    raise RuntimeError(
                        f"parallel worker {index} exited without a response"
                    ) from error
                if response[0] == "error":
                    raise RuntimeError(f"parallel worker {response[1]} failed:\n{response[2]}")
                responses[index] = response
                pending.remove(index)
            elif processes[index].exitcode not in (None, 0):
                raise RuntimeError(
                    f"parallel worker {index} exited with code {processes[index].exitcode}"
                )
        if pending and time.monotonic() >= deadline:
            raise TimeoutError(
                f"parallel workers did not finish within {timeout_seconds:g} seconds"
            )
        if pending:
            time.sleep(0.05)
    return [response for response in responses if response is not None]


def _rollout_worker_main(
    worker: int,
    connection: Connection,
    shared_rollout: _SharedAfterstateRollout,
    channels: int,
    residual_blocks: int,
    seed: int,
    command: str,
    payload: dict[str, Any],
) -> None:
    try:
        torch.set_num_threads(1)
        device = torch.device(f"cuda:{worker}")
        torch.cuda.set_device(device)
        model = AfterstatePpoNet(channels, residual_blocks, future_mode="none").to(device)
        model.load_state_dict(_load_bytes(payload["model_state_bytes"]))
        if command == "collect":
            _, result, metrics = collect_afterstate_ppo_rollout(
                model,
                device,
                shared_rollout.worker_storage(worker).actions.shape[1],
                np.random.default_rng(seed),
                gamma=payload["gamma"],
                gae_lambda=payload["gae_lambda"],
                logit_scale=payload["logit_scale"],
                inference_batch_size=payload["inference_batch_size"],
                policy_phi_coefficient=0.0,
                reward_mode="potential_shaping",
                future_mode="none",
                storage=shared_rollout.worker_storage(worker),
            )
            torch.cuda.synchronize(device)
            connection.send(("collect", result.scores, result.potentials, metrics))
        elif command == "evaluate":
            result = evaluate_afterstate_policy(
                model.actor,
                device,
                payload["flavors"],
                payload["ranks"],
                inference_batch_size=payload["inference_batch_size"],
                policy_phi_coefficient=0.0,
                future_mode="none",
            )
            torch.cuda.synchronize(device)
            connection.send(("evaluate", result.scores))
        else:
            raise ValueError(f"unknown afterstate parallel command: {command}")
    except BaseException:
        connection.send(("error", worker, traceback.format_exc()))
    finally:
        connection.close()


def _ddp_update_worker_main(
    rank: int,
    world_size: int,
    connection: Connection,
    rendezvous_path: str,
    shared_rollout: _SharedAfterstateRollout,
    channels: int,
    residual_blocks: int,
    training_state_bytes: bytes,
    teacher_state_bytes: bytes | None,
    teacher_channels: int | None,
    teacher_blocks: int | None,
    parameters: dict[str, Any],
    seed: int,
) -> None:
    try:
        torch.set_num_threads(1)
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
        dist.init_process_group(
            backend="nccl",
            init_method=f"file://{rendezvous_path}",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=10),
        )
        training_state = _load_bytes(training_state_bytes)
        model = AfterstatePpoNet(channels, residual_blocks, future_mode="none").to(device)
        model.load_state_dict(training_state["model_state_dict"])
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=parameters["learning_rate"],
            weight_decay=parameters["weight_decay"],
        )
        optimizer.load_state_dict(training_state["optimizer_state_dict"])
        execution_model = DistributedDataParallel(
            model,
            device_ids=[rank],
            output_device=rank,
        )
        teacher_actor = None
        if teacher_state_bytes is not None:
            assert teacher_channels is not None and teacher_blocks is not None
            teacher_actor = AfterstateValueNet(
                teacher_channels, teacher_blocks, future_mode="none"
            ).to(device)
            teacher_actor.load_state_dict(_load_bytes(teacher_state_bytes))
            teacher_actor.requires_grad_(False)
            teacher_actor.eval()

        metrics = update_afterstate_ppo(
            execution_model,
            optimizer,
            shared_rollout.worker_storage(rank).as_rollout(),
            device,
            np.random.default_rng(seed + rank),
            epochs=1,
            batch_size=parameters["batch_size"] // world_size,
            micro_batch_size=parameters["micro_batch_size"] // world_size,
            clip_ratio=parameters["clip_ratio"],
            value_clip=parameters["value_clip"],
            value_coefficient=parameters["value_coefficient"],
            entropy_coefficient=parameters["entropy_coefficient"],
            gradient_clip_norm=parameters["gradient_clip_norm"],
            logit_scale=parameters["logit_scale"],
            target_kl=parameters["target_kl"],
            policy_phi_coefficient=0.0,
            teacher_actor=teacher_actor,
            distillation_coefficient=parameters["distillation_coefficient"],
            advantage_mean=parameters["advantage_mean"],
            advantage_std=parameters["advantage_std"],
        )
        metric_names = sorted(metrics)
        metric_values = torch.tensor(
            [metrics[name] for name in metric_names], device=device, dtype=torch.float64
        )
        dist.all_reduce(metric_values, op=dist.ReduceOp.SUM)
        metric_values /= world_size
        averaged_metrics = dict(zip(metric_names, metric_values.cpu().tolist(), strict=True))
        if rank == 0:
            updated = io.BytesIO()
            torch.save(
                {
                    "model_state_dict": _cpu_tree(model.state_dict()),
                    "optimizer_state_dict": _cpu_tree(optimizer.state_dict()),
                },
                updated,
            )
            connection.send(("update", averaged_metrics, updated.getvalue()))
        else:
            connection.send(("done",))
    except BaseException:
        connection.send(("error", rank, traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        connection.close()


class ParallelAfterstateRuntime:
    """Two-process, two-GPU runtime for future-free alpha-zero afterstate PPO."""

    def __init__(
        self,
        *,
        workers: int,
        episodes: int,
        channels: int,
        residual_blocks: int,
        seed: int,
    ) -> None:
        if workers != 2:
            raise ValueError("afterstate parallel runtime currently requires two workers")
        if episodes % workers:
            raise ValueError("rollout episodes must be divisible by parallel workers")
        if not torch.cuda.is_available() or torch.cuda.device_count() < workers:
            raise RuntimeError("afterstate parallel runtime requires two CUDA devices")
        self.workers = workers
        self.episodes = episodes
        self._context = mp.get_context("spawn")
        self._shared = _create_shared_rollout(self._context, workers, episodes // workers)
        self._channels = channels
        self._residual_blocks = residual_blocks
        self._seeds = [seed + worker for worker in range(workers)]
        self._update_seed = seed + workers
        self._closed = False

    @staticmethod
    def _state_bytes(state: dict[str, Any]) -> bytes:
        buffer = io.BytesIO()
        torch.save(_cpu_tree(state), buffer)
        return buffer.getvalue()

    def _model_state_bytes(self, model: torch.nn.Module) -> bytes:
        return self._state_bytes(model.state_dict())

    def _run_rollout_workers(
        self, command: str, payloads: list[dict[str, Any]]
    ) -> list[tuple[Any, ...]]:
        connections: list[Connection] = []
        processes: list[mp.Process] = []
        for worker, payload in enumerate(payloads):
            parent, child = self._context.Pipe()
            process = self._context.Process(
                target=_rollout_worker_main,
                args=(
                    worker,
                    child,
                    self._shared,
                    self._channels,
                    self._residual_blocks,
                    self._seeds[worker],
                    command,
                    payload,
                ),
            )
            process.start()
            child.close()
            connections.append(parent)
            processes.append(process)
        try:
            return _receive_worker_responses(connections, processes)
        finally:
            self._finish_processes(processes, connections)

    @staticmethod
    def _finish_processes(
        processes: list[mp.Process], connections: list[Connection]
    ) -> None:
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join()
        for connection in connections:
            connection.close()

    def collect(
        self,
        model: AfterstatePpoNet,
        *,
        gamma: float,
        gae_lambda: float,
        logit_scale: float,
        inference_batch_size: int,
        policy_phi_coefficient: float,
        reward_mode: str,
        future_mode: str,
    ) -> tuple[AfterstatePpoRollout, EvaluationResult, dict[str, float]]:
        if policy_phi_coefficient != 0.0 or reward_mode != "potential_shaping":
            raise ValueError("parallel afterstate rollout requires alpha=0 and potential shaping")
        if future_mode != "none":
            raise ValueError("parallel afterstate rollout does not accept future inputs")
        payload = {
            "model_state_bytes": self._model_state_bytes(model),
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "logit_scale": logit_scale,
            "inference_batch_size": inference_batch_size,
        }
        responses = self._run_rollout_workers("collect", [payload] * self.workers)
        self._seeds = [seed + self.workers for seed in self._seeds]
        scores = np.concatenate([response[1] for response in responses])
        potentials = np.concatenate([response[2] for response in responses])
        metric_rows = [response[3] for response in responses]
        metrics = {
            key: float(np.mean([row[key] for row in metric_rows]))
            for key in metric_rows[0]
            if key != "rollout/feature_buffer_gib"
        }
        metrics["rollout/feature_buffer_gib"] = float(
            sum(row["rollout/feature_buffer_gib"] for row in metric_rows)
        )
        return self._shared.as_rollout(), EvaluationResult(scores, potentials), metrics

    def update(
        self,
        model: AfterstatePpoNet,
        optimizer: torch.optim.Optimizer,
        *,
        epochs: int,
        batch_size: int,
        micro_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        clip_ratio: float,
        value_clip: float,
        value_coefficient: float,
        entropy_coefficient: float,
        gradient_clip_norm: float,
        logit_scale: float,
        target_kl: float,
        data_parallel: bool,
        teacher_actor: AfterstateValueNet | None,
        distillation_coefficient: float,
    ) -> dict[str, float]:
        if epochs != 1:
            raise ValueError("DDP afterstate update currently requires one epoch")
        if not data_parallel:
            raise ValueError("DDP afterstate update requires data_parallel=true")
        if batch_size % self.workers or micro_batch_size % self.workers:
            raise ValueError("batch sizes must be divisible by the DDP worker count")
        training_state = self._state_bytes(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
        )
        teacher_state = (
            self._state_bytes(teacher_actor.state_dict()) if teacher_actor is not None else None
        )
        descriptor, rendezvous_path = tempfile.mkstemp(prefix="ahc015-ddp-", dir="/tmp")
        os.close(descriptor)
        os.unlink(rendezvous_path)
        connections: list[Connection] = []
        processes: list[mp.Process] = []
        parameters = {
            "batch_size": batch_size,
            "micro_batch_size": micro_batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "clip_ratio": clip_ratio,
            "value_clip": value_clip,
            "value_coefficient": value_coefficient,
            "entropy_coefficient": entropy_coefficient,
            "gradient_clip_norm": gradient_clip_norm,
            "logit_scale": logit_scale,
            "target_kl": target_kl,
            "distillation_coefficient": distillation_coefficient,
            "advantage_mean": float(self._shared.advantages.array().mean()),
            "advantage_std": float(self._shared.advantages.array().std()),
        }
        for rank in range(self.workers):
            parent, child = self._context.Pipe()
            process = self._context.Process(
                target=_ddp_update_worker_main,
                args=(
                    rank,
                    self.workers,
                    child,
                    rendezvous_path,
                    self._shared,
                    self._channels,
                    self._residual_blocks,
                    training_state,
                    teacher_state,
                    teacher_actor.channels if teacher_actor is not None else None,
                    teacher_actor.residual_blocks if teacher_actor is not None else None,
                    parameters,
                    self._update_seed,
                ),
            )
            process.start()
            child.close()
            connections.append(parent)
            processes.append(process)
        try:
            responses = _receive_worker_responses(connections, processes)
            response = responses[0]
            if response[0] != "update":
                raise RuntimeError(f"unexpected rank-zero response: {response[0]}")
            updated = _load_bytes(response[2])
            model.load_state_dict(updated["model_state_dict"])
            optimizer.load_state_dict(updated["optimizer_state_dict"])
            self._update_seed += 1
            return response[1]
        finally:
            self._finish_processes(processes, connections)
            if os.path.exists(rendezvous_path):
                os.unlink(rendezvous_path)

    def evaluate(
        self,
        model: AfterstatePpoNet,
        flavors: NDArray[np.uint8],
        ranks: NDArray[np.uint8],
        *,
        inference_batch_size: int,
    ) -> tuple[None, NDArray[np.int64]]:
        if len(flavors) % self.workers:
            raise ValueError("evaluation episodes must be divisible by parallel workers")
        state = self._model_state_bytes(model)
        shard = len(flavors) // self.workers
        payloads = [
            {
                "model_state_bytes": state,
                "flavors": flavors[worker * shard : (worker + 1) * shard],
                "ranks": ranks[worker * shard : (worker + 1) * shard],
                "inference_batch_size": inference_batch_size,
            }
            for worker in range(self.workers)
        ]
        responses = self._run_rollout_workers("evaluate", payloads)
        return None, np.concatenate([response[1] for response in responses])

    def close(self) -> None:
        self._closed = True
