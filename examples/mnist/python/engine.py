from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float
    samples_per_second: float
    duration_seconds: float

    def with_prefix(self, prefix: str) -> dict[str, float]:
        return {
            f"{prefix}/loss": self.loss,
            f"{prefix}/accuracy": self.accuracy,
            f"{prefix}/samples_per_second": self.samples_per_second,
            f"{prefix}/duration_seconds": self.duration_seconds,
        }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> EpochMetrics:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    _synchronize(device)
    started = time.perf_counter()

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.shape[0]
        total_loss += loss.detach().item() * batch_size
        total_correct += (logits.detach().argmax(dim=1) == targets).sum().item()
        total_samples += batch_size

    _synchronize(device)
    duration = time.perf_counter() - started
    return EpochMetrics(
        loss=total_loss / total_samples,
        accuracy=total_correct / total_samples,
        samples_per_second=total_samples / duration,
        duration_seconds=duration,
    )


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> EpochMetrics:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    _synchronize(device)
    started = time.perf_counter()

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")
        logits = model(inputs)
        loss = criterion(logits, targets)
        batch_size = targets.shape[0]
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_samples += batch_size

    _synchronize(device)
    duration = time.perf_counter() - started
    return EpochMetrics(
        loss=total_loss / total_samples,
        accuracy=total_correct / total_samples,
        samples_per_second=total_samples / duration,
        duration_seconds=duration,
    )
