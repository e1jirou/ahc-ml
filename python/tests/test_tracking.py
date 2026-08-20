from __future__ import annotations

import sys
from types import SimpleNamespace

from ahc_ml.tracking import WandbTracker


class FakeRun:
    def __init__(self) -> None:
        self.log_calls: list[dict[str, object]] = []
        self.log_options: list[dict[str, object]] = []

    def log(self, values: dict[str, object], **options: object) -> None:
        self.log_calls.append(values)
        self.log_options.append(options)


def test_log_image_adds_an_image_to_the_run(monkeypatch, tmp_path) -> None:
    run = FakeRun()
    fake_wandb = SimpleNamespace(
        init=lambda **_kwargs: run,
        Image=lambda path, caption: (path, caption),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    tracker = WandbTracker(
        project="test",
        entity=None,
        mode="offline",
        name="test-run",
        config={},
        directory=tmp_path,
    )
    image_path = tmp_path / "model.png"

    tracker.log_image(image_path, key="model/architecture", caption="Model graph")

    assert run.log_calls == [{"model/architecture": (str(image_path), "Model graph")}]
    assert run.log_options == [{"step": 0, "commit": False}]


def test_log_image_is_skipped_when_wandb_is_disabled(monkeypatch, tmp_path) -> None:
    run = FakeRun()
    fake_wandb = SimpleNamespace(
        init=lambda **_kwargs: run,
        Image=lambda path, caption: (path, caption),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    tracker = WandbTracker(
        project="test",
        entity=None,
        mode="disabled",
        name="test-run",
        config={},
        directory=tmp_path,
    )

    tracker.log_image(tmp_path / "model.png", key="model/architecture")

    assert run.log_calls == []


def test_log_artifact_is_skipped_when_wandb_is_disabled(monkeypatch, tmp_path) -> None:
    run = FakeRun()
    fake_wandb = SimpleNamespace(
        init=lambda **_kwargs: run,
        Artifact=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must not be called")),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    tracker = WandbTracker(
        project="test",
        entity=None,
        mode="disabled",
        name="test-run",
        config={},
        directory=tmp_path,
    )

    tracker.log_artifact(tmp_path / "model.pt", name="model", artifact_type="model")
