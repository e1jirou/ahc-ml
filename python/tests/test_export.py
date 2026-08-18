import base64
import json

import numpy as np
import pytest
import torch
from ahc_ml.export import (
    encode_base93,
    export_quantized_state_dict,
    export_state_dict,
    read_exported_state_dict,
    read_quantized_state_dict,
)


def test_export_round_trip(tmp_path) -> None:
    path = tmp_path / "model.bin"
    state_dict = {
        "linear.weight": torch.tensor([[1.0, -2.5], [3.25, 4.0]]),
        "linear.bias": torch.tensor([0.5, -0.75]),
        "scalar": torch.tensor(2.0, dtype=torch.float64),
    }
    manifest = export_state_dict(state_dict, path, metadata={"architecture": "test"})
    restored = read_exported_state_dict(path)

    assert list(restored) == sorted(state_dict)
    for name, expected in state_dict.items():
        assert restored[name].shape == tuple(expected.shape)
        np.testing.assert_array_equal(restored[name].values, expected.float().numpy())

    assert manifest["tensor_count"] == 3
    saved_manifest = json.loads(path.with_suffix(".bin.json").read_text())
    assert saved_manifest["metadata"]["architecture"] == "test"


def test_reader_rejects_invalid_magic(tmp_path) -> None:
    path = tmp_path / "bad.bin"
    path.write_bytes(b"not a model")
    with pytest.raises(ValueError, match="invalid model magic"):
        read_exported_state_dict(path)


def test_quantized_export_round_trip_and_rust_source(tmp_path) -> None:
    path = tmp_path / "model.q8.bin"
    rust_source = tmp_path / "model_data.rs"
    state_dict = {
        "linear.weight": torch.tensor([[0.0, -0.25, 1.0], [8.0, -4.0, 0.125]], dtype=torch.float32),
        "linear.bias": torch.tensor([0.25, -0.75]),
        "scalar": torch.tensor(2.0),
    }
    manifest = export_quantized_state_dict(
        state_dict,
        path,
        metadata={"architecture": "test"},
        rust_source=rust_source,
    )
    restored = read_quantized_state_dict(path)

    assert list(restored) == sorted(state_dict)
    for name, expected in state_dict.items():
        assert restored[name].shape == tuple(expected.shape)
        maximum_scale = max(float(expected.abs().max()), 1.0) / 127.0
        np.testing.assert_allclose(
            restored[name].values,
            expected.numpy(),
            atol=maximum_scale,
            rtol=0,
        )
    assert manifest["compression"] in {"canonical-huffman", "stored"}
    assert manifest["compressed_size"] == path.stat().st_size
    source = rust_source.read_text()
    assert "MODEL_DATA_BASE93" in source
    encoded = "".join(
        line.removeprefix('    "').removesuffix('",') for line in source.splitlines()[2:-2]
    )
    assert len(encoded) < len(base64.b64encode(path.read_bytes()))


def test_base93_known_values_and_literal_safety() -> None:
    cases = [
        (b"", ""),
        (bytes([0]), "  "),
        (bytes([1]), "! "),
        (bytes([92]), "~ "),
        (bytes([93]), " !"),
        (bytes([0, 0]), "   "),
        (b"Base93", "T}!^g4T "),
    ]
    for raw, encoded in cases:
        assert encode_base93(raw) == encoded
    encoded = encode_base93(bytes(range(256)) * 4)
    assert set(encoded) <= set(
        " !#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "[]^_`abcdefghijklmnopqrstuvwxyz{|}~"
    )
    assert '"' not in encoded
    assert "\\" not in encoded


def test_quantized_reader_rejects_invalid_magic(tmp_path) -> None:
    path = tmp_path / "bad.q8.bin"
    path.write_bytes(b"not a model")
    with pytest.raises(ValueError, match="invalid compressed model magic"):
        read_quantized_state_dict(path)
