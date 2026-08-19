from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .game import CELL_COUNT


@dataclass(frozen=True, slots=True)
class ReplayBatch:
    boards: NDArray[np.uint8]
    actions: NDArray[np.uint8]
    placed: NDArray[np.uint8]
    flavors: NDArray[np.uint8]
    mc_targets: NDArray[np.float32]


class ReplayBuffer:
    """A compact ring buffer of unexpanded afterstates."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("replay capacity must be positive")
        self.capacity = capacity
        self.boards = np.empty((capacity, CELL_COUNT), dtype=np.uint8)
        self.actions = np.empty(capacity, dtype=np.uint8)
        self.placed = np.empty(capacity, dtype=np.uint8)
        self.flavors = np.empty((capacity, CELL_COUNT), dtype=np.uint8)
        self.mc_targets = np.empty(capacity, dtype=np.float32)
        self.size = 0
        self.cursor = 0

    def __len__(self) -> int:
        return self.size

    def add(self, batch: ReplayBatch) -> None:
        count = len(batch.placed)
        if not (
            batch.boards.shape == (count, CELL_COUNT)
            and batch.actions.shape == (count,)
            and batch.flavors.shape == (count, CELL_COUNT)
            and batch.mc_targets.shape == (count,)
        ):
            raise ValueError("invalid replay batch shape")
        if count >= self.capacity:
            start = count - self.capacity
            batch = ReplayBatch(
                batch.boards[start:],
                batch.actions[start:],
                batch.placed[start:],
                batch.flavors[start:],
                batch.mc_targets[start:],
            )
            count = self.capacity
        first_count = min(count, self.capacity - self.cursor)
        second_count = count - first_count
        destination = slice(self.cursor, self.cursor + first_count)
        self._copy(destination, batch, slice(0, first_count))
        if second_count:
            self._copy(slice(0, second_count), batch, slice(first_count, count))
        self.cursor = (self.cursor + count) % self.capacity
        self.size = min(self.capacity, self.size + count)

    def _copy(self, destination: slice, batch: ReplayBatch, source: slice) -> None:
        self.boards[destination] = batch.boards[source]
        self.actions[destination] = batch.actions[source]
        self.placed[destination] = batch.placed[source]
        self.flavors[destination] = batch.flavors[source]
        self.mc_targets[destination] = batch.mc_targets[source]

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
        *,
        minimum_placed: int = 1,
    ) -> ReplayBatch:
        eligible = np.flatnonzero(self.placed[: self.size] >= minimum_placed)
        if len(eligible) == 0:
            raise ValueError(f"replay has no samples with placed >= {minimum_placed}")
        indices = rng.choice(eligible, size=batch_size, replace=len(eligible) < batch_size)
        return ReplayBatch(
            self.boards[indices].copy(),
            self.actions[indices].copy(),
            self.placed[indices].copy(),
            self.flavors[indices].copy(),
            self.mc_targets[indices].copy(),
        )
