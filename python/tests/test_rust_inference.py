import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from ahc_ml.export import export_quantized_state_dict, read_quantized_state_dict

REPOSITORY_ROOT = Path(__file__).parents[2]
MNIST_PYTHON = REPOSITORY_ROOT / "examples/mnist/python"
sys.path.insert(0, str(MNIST_PYTHON))

from model import MnistCnn  # noqa: E402


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_quantized_rust_logits_match_pytorch(tmp_path) -> None:
    torch.manual_seed(7)
    model = MnistCnn(channels=(2, 3), hidden_size=4).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.reshape(-1)[::2] = 0
    model_path = tmp_path / "model.q8.bin"
    manifest = export_quantized_state_dict(model.state_dict(), model_path)
    assert manifest["compression"] == "canonical-huffman"
    restored = read_quantized_state_dict(model_path)
    model.load_state_dict(
        {name: torch.from_numpy(tensor.values) for name, tensor in restored.items()}
    )

    pixels = np.arange(28 * 28, dtype=np.uint16).astype(np.uint8)
    images_path = tmp_path / "images.idx"
    images_path.write_bytes(struct.pack(">IIII", 2051, 1, 28, 28) + pixels.tobytes())
    inputs = torch.from_numpy(pixels.copy()).reshape(1, 1, 28, 28).float() / 255.0
    expected = model(inputs).detach().numpy()[0]

    result = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "-p",
            "mnist-inference",
            "--",
            "--model",
            str(model_path),
            "--images",
            str(images_path),
            "--index",
            "0",
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    logits_line = next(line for line in result.stdout.splitlines() if line.startswith("logits="))
    actual = np.fromstring(logits_line.removeprefix("logits="), sep=",")
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
