from __future__ import annotations

from pathlib import Path
from typing import Any


class WandbTracker:
    """Small W&B boundary so training can also run disabled or offline."""

    def __init__(
        self,
        *,
        project: str,
        entity: str | None,
        mode: str,
        name: str,
        config: dict[str, Any],
        directory: str | Path,
    ) -> None:
        if mode not in {"online", "offline", "disabled"}:
            raise ValueError("W&B mode must be online, offline, or disabled")

        import wandb

        self._mode = mode
        self._wandb = wandb
        self._run = wandb.init(
            project=project,
            entity=entity or None,
            mode=mode,
            name=name,
            config=config,
            dir=str(directory),
        )

    @property
    def run_id(self) -> str | None:
        return self._run.id if self._run is not None else None

    def log(self, values: dict[str, Any], *, step: int | None = None) -> None:
        if self._run is not None:
            self._run.log(values, step=step)

    def log_image(self, path: str | Path, *, key: str, caption: str | None = None) -> None:
        """Log an image to the run workspace."""
        if self._run is not None and self._mode != "disabled":
            self._run.log(
                {key: self._wandb.Image(str(path), caption=caption)},
                step=0,
                commit=False,
            )

    def log_artifact(self, path: str | Path, *, name: str, artifact_type: str) -> None:
        if self._run is None:
            return
        artifact = self._wandb.Artifact(name=name, type=artifact_type)
        artifact.add_file(str(path))
        self._run.log_artifact(artifact)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
