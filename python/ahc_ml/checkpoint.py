from __future__ import annotations

import os
from collections.abc import Collection
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

CHECKPOINT_VERSION = 1


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    config: dict[str, Any],
    metrics: dict[str, float],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload = {
        "format_version": CHECKPOINT_VERSION,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "metrics": metrics,
    }
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    map_location: str | torch.device = "cpu",
    allowed_missing_keys: Collection[str] = (),
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    version = checkpoint.get("format_version")
    if version != CHECKPOINT_VERSION:
        raise ValueError(f"unsupported checkpoint version: {version}")
    if allowed_missing_keys:
        incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        unexpected_missing = set(incompatible.missing_keys) - set(allowed_missing_keys)
        if unexpected_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "incompatible checkpoint state: "
                f"missing={sorted(unexpected_missing)}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
