from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from ahc_ml.device import select_device
from ahc_ml.export import read_exported_state_dict
from numpy.typing import NDArray

from .afterstate_features import encode_afterstates
from .afterstate_model import AfterstateValueNet
from .game import ACTION_COUNT, CELL_COUNT, SIDE, afterstates_batch, place_at_ranks
from .mcts_prior import TinyMctsPrior, handcrafted_prior_features
from .simulation import generate_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill a tiny structural MCTS prior")
    parser.add_argument("--teacher", type=Path, default=Path("outputs/ahc015/model.bin"))
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--collect-from", type=int, default=55)
    parser.add_argument("--collect-until", type=int, default=93)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/ahc015/mcts-prior.pt")
    )
    parser.add_argument(
        "--rust-output",
        type=Path,
        default=Path("examples/ahc015/rust/src/generated_mcts_prior.rs"),
    )
    return parser.parse_args()


def load_teacher(path: Path, device: torch.device) -> AfterstateValueNet:
    exported = read_exported_state_dict(path)
    state = {name: torch.from_numpy(tensor.values) for name, tensor in exported.items()}
    model = AfterstateValueNet(128, 10, "none")
    model.load_state_dict(state)
    return model.to(device).eval()


def collect_dataset(
    teacher: AfterstateValueNet,
    device: torch.device,
    episodes: int,
    seed: int,
    collect_from: int,
    collect_until: int,
    batch_size: int,
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    flavors, ranks = generate_cases(episodes, seed)
    boards = np.zeros((episodes, SIDE, SIDE), dtype=np.uint8)
    feature_rows: list[NDArray[np.float32]] = []
    label_rows: list[NDArray[np.int64]] = []
    for turn in range(CELL_COUNT - 1):
        placed = turn + 1
        boards = place_at_ranks(boards, ranks[:, turn], flavors[:, turn])
        candidates = afterstates_batch(boards)
        encoded = encode_afterstates(
            candidates.reshape(episodes * ACTION_COUNT, SIDE, SIDE),
            np.tile(np.arange(ACTION_COUNT, dtype=np.int64), episodes),
            placed,
            np.repeat(flavors, ACTION_COUNT, axis=0),
        )
        values = np.empty(episodes * ACTION_COUNT, dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(encoded), batch_size):
                stop = min(start + batch_size, len(encoded))
                values[start:stop] = (
                    teacher(torch.from_numpy(encoded[start:stop]).to(device)).cpu().numpy()
                )
        values = values.reshape(episodes, ACTION_COUNT)
        selected = np.argmax(values, axis=1)
        if collect_from <= placed < collect_until:
            feature_rows.append(handcrafted_prior_features(candidates, flavors, placed))
            label_rows.append(selected.astype(np.int64))
        boards = candidates[np.arange(episodes), selected]
    return np.concatenate(feature_rows), np.concatenate(label_rows)


def train_prior(
    features: NDArray[np.float32],
    labels: NDArray[np.int64],
    epochs: int,
    batch_size: int,
    seed: int,
) -> tuple[TinyMctsPrior, dict[str, float]]:
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(features), generator=generator)
    split = len(features) * 9 // 10
    train_indices, validation_indices = order[:split], order[split:]
    inputs = torch.from_numpy(features)
    targets = torch.from_numpy(labels)
    model = TinyMctsPrior()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    for _ in range(epochs):
        permutation = train_indices[torch.randperm(len(train_indices), generator=generator)]
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            logits = model(inputs[indices])
            loss = torch.nn.functional.cross_entropy(logits, targets[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    with torch.inference_mode():
        train_correct = model(inputs[train_indices]).argmax(1) == targets[train_indices]
        train_accuracy = float(train_correct.float().mean())
        validation_correct = (
            model(inputs[validation_indices]).argmax(1) == targets[validation_indices]
        )
        validation_accuracy = float(
            validation_correct.float().mean()
        )
    return model.eval(), {
        "train_accuracy": train_accuracy,
        "validation_accuracy": validation_accuracy,
    }


def rust_array(name: str, values: np.ndarray) -> str:
    flat = values.reshape(-1)
    body = ",".join(f"{float(value):.9e}" for value in flat)
    return f"pub const {name}: [f32; {len(flat)}] = [{body}];\n"


def write_rust(model: TinyMctsPrior, path: Path) -> None:
    state = model.state_dict()
    source = "// Generated by train_mcts_prior.py.\n"
    source += rust_array("HIDDEN_WEIGHT", state["hidden.weight"].numpy())
    source += rust_array("HIDDEN_BIAS", state["hidden.bias"].numpy())
    source += rust_array("OUTPUT_WEIGHT", state["output.weight"].numpy())
    source += rust_array("OUTPUT_BIAS", state["output.bias"].numpy())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)


def main() -> None:
    args = parse_args()
    if not 0 <= args.collect_from < args.collect_until <= CELL_COUNT:
        raise ValueError("collection turns must satisfy 0 <= from < until <= 100")
    started = time.monotonic()
    device, description = select_device(args.device)
    teacher = load_teacher(args.teacher, device)
    features, labels = collect_dataset(
        teacher,
        device,
        args.episodes,
        args.seed,
        args.collect_from,
        args.collect_until,
        args.batch_size,
    )
    model, metrics = train_prior(features, labels, args.epochs, args.batch_size, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "metrics": metrics}, args.output)
    write_rust(model, args.rust_output)
    print(
        {
            "device": description,
            "samples": len(features),
            **metrics,
            "seconds": time.monotonic() - started,
        }
    )


if __name__ == "__main__":
    main()
