from __future__ import annotations

import io
import math
import multiprocessing as mp
import traceback
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from .features import BOARD_CHANNELS, FUTURE_CHANNELS, FUTURE_LENGTH
from .game import ACTION_COUNT, CELL_COUNT, SIDE
from .model import Ahc015PpoNet
from .ppo import PpoRollout, PpoRolloutStorage, collect_ppo_rollout, ppo_update
from .simulation import EvaluationResult, evaluate_policy


@dataclass(frozen=True, slots=True)
class _SharedArray:
    buffer: Any
    shape: tuple[int, ...]
    dtype: str

    def array(self) -> np.ndarray:
        return np.frombuffer(self.buffer, dtype=np.dtype(self.dtype)).reshape(self.shape)


@dataclass(frozen=True, slots=True)
class _SharedRollout:
    board_features: _SharedArray
    future_features: _SharedArray
    candidate_potentials: _SharedArray
    actions: _SharedArray
    old_log_probs: _SharedArray
    old_values: _SharedArray
    advantages: _SharedArray
    returns: _SharedArray

    def worker_storage(self, worker: int) -> PpoRolloutStorage:
        return PpoRolloutStorage(
            **{field: getattr(self, field).array()[worker] for field in self.__dataclass_fields__}
        )

    def as_rollout(self) -> PpoRollout:
        arrays = {field: getattr(self, field).array() for field in self.__dataclass_fields__}
        transitions = arrays["actions"].size
        return PpoRollout(
            board_features=arrays["board_features"].reshape(
                transitions, ACTION_COUNT, BOARD_CHANNELS, SIDE, SIDE
            ),
            future_features=arrays["future_features"].reshape(
                transitions, ACTION_COUNT, FUTURE_CHANNELS, FUTURE_LENGTH
            ),
            candidate_potentials=arrays["candidate_potentials"].reshape(transitions, ACTION_COUNT),
            actions=arrays["actions"].reshape(transitions),
            old_log_probs=arrays["old_log_probs"].reshape(transitions),
            old_values=arrays["old_values"].reshape(transitions),
            advantages=arrays["advantages"].reshape(transitions),
            returns=arrays["returns"].reshape(transitions),
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


def _create_shared_rollout(context: Any, workers: int, episodes_per_worker: int) -> _SharedRollout:
    prefix = (workers, CELL_COUNT - 1, episodes_per_worker)
    return _SharedRollout(
        board_features=_shared_array(
            context,
            (*prefix, ACTION_COUNT, BOARD_CHANNELS, SIDE, SIDE),
            np.dtype(np.uint8),
        ),
        future_features=_shared_array(
            context,
            (*prefix, ACTION_COUNT, FUTURE_CHANNELS, FUTURE_LENGTH),
            np.dtype(np.uint8),
        ),
        candidate_potentials=_shared_array(context, (*prefix, ACTION_COUNT), np.dtype(np.float32)),
        actions=_shared_array(context, prefix, np.dtype(np.int64)),
        old_log_probs=_shared_array(context, prefix, np.dtype(np.float32)),
        old_values=_shared_array(context, prefix, np.dtype(np.float32)),
        advantages=_shared_array(context, prefix, np.dtype(np.float32)),
        returns=_shared_array(context, prefix, np.dtype(np.float32)),
    )


def _clear_cuda_cache(device: torch.device) -> None:
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()


def _worker_main(
    worker: int,
    connection: Connection,
    shared_rollout: _SharedRollout,
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
        model = Ahc015PpoNet(channels, residual_blocks).to(device)
        rng = np.random.default_rng(seed)
        storage = shared_rollout.worker_storage(worker)
        model.load_state_dict(
            torch.load(
                io.BytesIO(payload["model_state_bytes"]),
                map_location="cpu",
                weights_only=True,
            )
        )
        if command == "collect":
            _, result, metrics = collect_ppo_rollout(
                model,
                device,
                storage.actions.shape[1],
                rng,
                gamma=payload["gamma"],
                gae_lambda=payload["gae_lambda"],
                logit_scale=payload["logit_scale"],
                inference_batch_size=payload["inference_batch_size"],
                storage=storage,
            )
            _clear_cuda_cache(device)
            connection.send(("collect", result.scores, result.potentials, metrics))
        elif command == "evaluate":
            flavors = payload["flavors"]
            ranks = payload["ranks"]
            greedy = evaluate_policy(
                None,
                device,
                flavors,
                ranks,
                inference_batch_size=payload["inference_batch_size"],
            )
            learned = evaluate_policy(
                model.actor,
                device,
                flavors,
                ranks,
                inference_batch_size=payload["inference_batch_size"],
            )
            _clear_cuda_cache(device)
            connection.send(("evaluate", greedy.scores, learned.scores))
        else:
            raise ValueError(f"unknown parallel runtime command: {command}")
    except BaseException:
        connection.send(("error", worker, traceback.format_exc()))
    finally:
        connection.close()


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


def _ppo_worker_main(
    connection: Connection,
    shared_rollout: _SharedRollout,
    channels: int,
    residual_blocks: int,
    training_state_bytes: bytes,
    parameters: dict[str, Any],
    seed: int,
) -> None:
    try:
        torch.set_num_threads(1)
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        training_state = torch.load(
            io.BytesIO(training_state_bytes), map_location="cpu", weights_only=False
        )
        model = Ahc015PpoNet(channels, residual_blocks).to(device)
        model.load_state_dict(training_state["model_state_dict"])
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=parameters["learning_rate"],
            weight_decay=parameters["weight_decay"],
        )
        optimizer.load_state_dict(training_state["optimizer_state_dict"])
        execution_model: torch.nn.Module = model
        if parameters["data_parallel"]:
            if torch.cuda.device_count() < 2:
                raise RuntimeError("parallel PPO requires two CUDA devices")
            execution_model = torch.nn.DataParallel(model)
        metrics = ppo_update(
            execution_model,
            optimizer,
            shared_rollout.as_rollout(),
            device,
            np.random.default_rng(seed),
            epochs=parameters["epochs"],
            batch_size=parameters["batch_size"],
            micro_batch_size=parameters["micro_batch_size"],
            clip_ratio=parameters["clip_ratio"],
            value_clip=parameters["value_clip"],
            value_coefficient=parameters["value_coefficient"],
            entropy_coefficient=parameters["entropy_coefficient"],
            gradient_clip_norm=parameters["gradient_clip_norm"],
            logit_scale=parameters["logit_scale"],
            target_kl=parameters["target_kl"],
        )
        torch.cuda.synchronize()
        updated = io.BytesIO()
        torch.save(
            {
                "model_state_dict": _cpu_tree(model.state_dict()),
                "optimizer_state_dict": _cpu_tree(optimizer.state_dict()),
            },
            updated,
        )
        connection.send(("update", metrics, updated.getvalue()))
    except BaseException:
        connection.send(("error", 0, traceback.format_exc()))
    finally:
        connection.close()


class ParallelAhc015Runtime:
    """Independent CPU/GPU pipelines with a copy-free shared rollout buffer."""

    def __init__(
        self,
        *,
        workers: int,
        episodes: int,
        channels: int,
        residual_blocks: int,
        seed: int,
    ) -> None:
        if workers < 2:
            raise ValueError("parallel runtime requires at least two workers")
        if episodes % workers:
            raise ValueError("rollout episodes must be divisible by parallel workers")
        if not torch.cuda.is_available() or torch.cuda.device_count() < workers:
            raise RuntimeError(f"parallel runtime requires {workers} CUDA devices")
        self.workers = workers
        self.episodes = episodes
        self._context = mp.get_context("spawn")
        self._shared = _create_shared_rollout(self._context, workers, episodes // workers)
        self._channels = channels
        self._residual_blocks = residual_blocks
        self._seeds = [seed + worker for worker in range(workers)]
        self._update_seed = seed + workers
        self._closed = False

    def _run_workers(self, command: str, payloads: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
        if self._closed:
            raise RuntimeError("parallel runtime is closed")
        connections: list[Connection] = []
        processes: list[mp.Process] = []
        for worker, payload in enumerate(payloads):
            parent, child = self._context.Pipe()
            process = self._context.Process(
                target=_worker_main,
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
            responses = [connection.recv() for connection in connections]
            for response in responses:
                self._raise_worker_error(response)
            return responses
        finally:
            for process in processes:
                process.join(timeout=10)
                if process.is_alive():
                    process.terminate()
                    process.join()
            for connection in connections:
                connection.close()

    @staticmethod
    def _model_state_bytes(model: torch.nn.Module) -> bytes:
        buffer = io.BytesIO()
        torch.save(
            {name: value.detach().cpu() for name, value in model.state_dict().items()},
            buffer,
        )
        return buffer.getvalue()

    @staticmethod
    def _training_state_bytes(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> bytes:
        buffer = io.BytesIO()
        torch.save(
            {
                "model_state_dict": _cpu_tree(model.state_dict()),
                "optimizer_state_dict": _cpu_tree(optimizer.state_dict()),
            },
            buffer,
        )
        return buffer.getvalue()

    @staticmethod
    def _raise_worker_error(message: tuple[Any, ...]) -> None:
        if message[0] == "error":
            raise RuntimeError(f"parallel worker {message[1]} failed:\n{message[2]}")

    def collect(
        self,
        model: Ahc015PpoNet,
        *,
        gamma: float,
        gae_lambda: float,
        logit_scale: float,
        inference_batch_size: int,
    ) -> tuple[PpoRollout, EvaluationResult, dict[str, float]]:
        model_state_bytes = self._model_state_bytes(model)
        payload = {
            "model_state_bytes": model_state_bytes,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "logit_scale": logit_scale,
            "inference_batch_size": inference_batch_size,
        }
        responses = self._run_workers("collect", [payload] * self.workers)
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
        return (
            self._shared.as_rollout(),
            EvaluationResult(scores, potentials),
            metrics,
        )

    def update(
        self,
        model: Ahc015PpoNet,
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
    ) -> dict[str, float]:
        if self._closed:
            raise RuntimeError("parallel runtime is closed")
        parent, child = self._context.Pipe()
        process = self._context.Process(
            target=_ppo_worker_main,
            args=(
                child,
                self._shared,
                self._channels,
                self._residual_blocks,
                self._training_state_bytes(model, optimizer),
                {
                    "epochs": epochs,
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
                    "data_parallel": data_parallel,
                },
                self._update_seed,
            ),
        )
        process.start()
        child.close()
        try:
            response = parent.recv()
            self._raise_worker_error(response)
            if response[0] != "update":
                raise RuntimeError(f"unexpected PPO worker response: {response[0]}")
            updated = torch.load(io.BytesIO(response[2]), map_location="cpu", weights_only=False)
            model.load_state_dict(updated["model_state_dict"])
            optimizer.load_state_dict(updated["optimizer_state_dict"])
            self._update_seed += 1
            return response[1]
        finally:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join()
            parent.close()

    def evaluate(
        self,
        model: Ahc015PpoNet,
        flavors: NDArray[np.uint8],
        ranks: NDArray[np.uint8],
        *,
        inference_batch_size: int,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        if len(flavors) % self.workers:
            raise ValueError("evaluation episodes must be divisible by parallel workers")
        model_state_bytes = self._model_state_bytes(model)
        shard = len(flavors) // self.workers
        payloads = []
        for worker in range(self.workers):
            start = worker * shard
            stop = start + shard
            payloads.append(
                {
                    "model_state_bytes": model_state_bytes,
                    "flavors": flavors[start:stop],
                    "ranks": ranks[start:stop],
                    "inference_batch_size": inference_batch_size,
                }
            )
        responses = self._run_workers("evaluate", payloads)
        return (
            np.concatenate([response[1] for response in responses]),
            np.concatenate([response[2] for response in responses]),
        )

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> ParallelAhc015Runtime:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
