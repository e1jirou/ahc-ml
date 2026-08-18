import torch
from ahc_ml.checkpoint import load_checkpoint, save_checkpoint
from torch import nn


def test_checkpoint_round_trip(tmp_path) -> None:
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        epoch=4,
        config={"seed": 1},
        metrics={"accuracy": 0.9},
    )

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    checkpoint = load_checkpoint(path, model=model, optimizer=optimizer)

    assert checkpoint["epoch"] == 4
    assert checkpoint["metrics"]["accuracy"] == 0.9
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name])
