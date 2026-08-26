from __future__ import annotations

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
from .ppo import PpoRollout, PpoRolloutStorage, collect_ppo_rollout
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
) -> None:
    try:
        torch.set_num_threads(1)
        device = torch.device(f"cuda:{worker}")
        torch.cuda.set_device(device)
        model = Ahc015PpoNet(channels, residual_blocks).to(device)
        rng = np.random.default_rng(seed)
        storage = shared_rollout.worker_storage(worker)
        connection.send(("ready", worker))
        while True:
            command, payload = connection.recv()
            if command == "close":
                break
            model.load_state_dict(payload["model_state_dict"])
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


class ParallelAhc015Runtime:
    """Persistent independent CPU/GPU pipelines for rollout and evaluation."""

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
        self._connections: list[Connection] = []
        self._processes: list[mp.Process] = []
        for worker in range(workers):
            parent, child = self._context.Pipe()
            process = self._context.Process(
                target=_worker_main,
                args=(
                    worker,
                    child,
                    self._shared,
                    channels,
                    residual_blocks,
                    seed + worker,
                ),
            )
            process.start()
            child.close()
            self._connections.append(parent)
            self._processes.append(process)
        for connection in self._connections:
            message = connection.recv()
            self._raise_worker_error(message)
            if message[0] != "ready":
                raise RuntimeError(f"unexpected parallel runtime response: {message[0]}")

    @staticmethod
    def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
        return {name: value.detach().cpu() for name, value in model.state_dict().items()}

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
        state_dict = self._cpu_state_dict(model)
        payload = {
            "model_state_dict": state_dict,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "logit_scale": logit_scale,
            "inference_batch_size": inference_batch_size,
        }
        for connection in self._connections:
            connection.send(("collect", payload))
        responses = [connection.recv() for connection in self._connections]
        for response in responses:
            self._raise_worker_error(response)
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
        state_dict = self._cpu_state_dict(model)
        shard = len(flavors) // self.workers
        for worker, connection in enumerate(self._connections):
            start = worker * shard
            stop = start + shard
            connection.send(
                (
                    "evaluate",
                    {
                        "model_state_dict": state_dict,
                        "flavors": flavors[start:stop],
                        "ranks": ranks[start:stop],
                        "inference_batch_size": inference_batch_size,
                    },
                )
            )
        responses = [connection.recv() for connection in self._connections]
        for response in responses:
            self._raise_worker_error(response)
        return (
            np.concatenate([response[1] for response in responses]),
            np.concatenate([response[2] for response in responses]),
        )

    def close(self) -> None:
        for connection in self._connections:
            try:
                connection.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for process in self._processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join()
        for connection in self._connections:
            connection.close()
        self._connections.clear()
        self._processes.clear()

    def __enter__(self) -> ParallelAhc015Runtime:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
