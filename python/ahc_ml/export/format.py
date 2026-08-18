from __future__ import annotations

import heapq
import json
import os
import struct
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import torch

MAGIC = b"AHCRLWT1"
FORMAT_VERSION = 1
DTYPE_FLOAT32 = 1
QUANTIZED_MAGIC = b"AHCRLQ81"
QUANTIZED_FORMAT_VERSION = 1
COMPRESSED_MAGIC = b"AHCRLZ01"
COMPRESSED_FORMAT_VERSION = 1
BASE93_ALPHABET = (
    " !#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~"
)
BASE93_THRESHOLD = len(BASE93_ALPHABET) ** 2 - (1 << 13) - 1


@dataclass(frozen=True)
class ExportedTensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    values: np.ndarray


def _canonical_codes(lengths: list[int]) -> dict[int, tuple[int, int]]:
    entries = sorted((length, symbol) for symbol, length in enumerate(lengths) if length)
    codes = {}
    code = 0
    previous_length = 0
    for length, symbol in entries:
        code <<= length - previous_length
        codes[symbol] = (code, length)
        code += 1
        previous_length = length
    return codes


def _huffman_compress(data: bytes) -> tuple[bytes, list[int]]:
    frequencies = Counter(data)
    heap = []
    order = 0
    for symbol, frequency in frequencies.items():
        heapq.heappush(heap, (frequency, order, symbol))
        order += 1
    if not heap:
        return b"", [0] * 256

    while len(heap) > 1:
        left_frequency, _, left = heapq.heappop(heap)
        right_frequency, _, right = heapq.heappop(heap)
        heapq.heappush(heap, (left_frequency + right_frequency, order, (left, right)))
        order += 1

    lengths = [0] * 256
    stack = [(heap[0][2], 0)]
    while stack:
        node, depth = stack.pop()
        if isinstance(node, int):
            lengths[node] = max(depth, 1)
        else:
            left, right = node
            stack.append((left, depth + 1))
            stack.append((right, depth + 1))
    if max(lengths) > 32:
        return data, []

    codes = _canonical_codes(lengths)
    output = bytearray()
    accumulator = 0
    bit_count = 0
    for symbol in data:
        code, length = codes[symbol]
        accumulator = (accumulator << length) | code
        bit_count += length
        while bit_count >= 8:
            bit_count -= 8
            output.append((accumulator >> bit_count) & 0xFF)
            accumulator &= (1 << bit_count) - 1
    if bit_count:
        output.append((accumulator << (8 - bit_count)) & 0xFF)
    return bytes(output), lengths


def _huffman_decompress(data: bytes, lengths: list[int], expected_size: int) -> bytes:
    if len(lengths) != 256 or max(lengths, default=0) > 32:
        raise ValueError("invalid Huffman code lengths")
    codes = {(length, code): symbol for symbol, (code, length) in _canonical_codes(lengths).items()}
    output = bytearray()
    code = 0
    length = 0
    for byte in data:
        for shift in range(7, -1, -1):
            code = (code << 1) | ((byte >> shift) & 1)
            length += 1
            symbol = codes.get((length, code))
            if symbol is not None:
                output.append(symbol)
                code = 0
                length = 0
                if len(output) == expected_size:
                    return bytes(output)
            elif length >= 32:
                raise ValueError("invalid Huffman bitstream")
    raise ValueError("truncated Huffman bitstream")


def _write_exact(file: BinaryIO, data: bytes) -> None:
    written = file.write(data)
    if written != len(data):
        raise OSError("failed to write the complete model file")


def export_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Export named tensors as contiguous little-endian float32 data."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    tensors: list[tuple[str, np.ndarray]] = []

    for name in sorted(state_dict):
        encoded_name = name.encode("utf-8")
        if len(encoded_name) > 0xFFFF:
            raise ValueError(f"tensor name is too long: {name!r}")
        tensor = state_dict[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"state_dict entry {name!r} is not a tensor")
        array = tensor.detach().cpu().to(torch.float32).contiguous().numpy()
        array = np.asarray(array, dtype="<f4")
        tensors.append((name, array))

    with temporary.open("wb") as file:
        _write_exact(file, MAGIC)
        _write_exact(file, struct.pack("<II", FORMAT_VERSION, len(tensors)))
        for name, array in tensors:
            encoded_name = name.encode("utf-8")
            if array.ndim > 0xFF:
                raise ValueError(f"tensor has too many dimensions: {name!r}")
            raw = array.tobytes(order="C")
            _write_exact(file, struct.pack("<H", len(encoded_name)))
            _write_exact(file, encoded_name)
            _write_exact(file, struct.pack("<BB", DTYPE_FLOAT32, array.ndim))
            for dimension in array.shape:
                if dimension < 0 or dimension > 0xFFFFFFFF:
                    raise ValueError(f"invalid tensor dimension in {name!r}: {dimension}")
                _write_exact(file, struct.pack("<I", dimension))
            _write_exact(file, struct.pack("<Q", len(raw)))
            _write_exact(file, raw)
    os.replace(temporary, destination)

    manifest = {
        "format": "ahc-ml-weights",
        "format_version": FORMAT_VERSION,
        "byte_order": "little",
        "tensor_count": len(tensors),
        "tensors": [
            {"name": name, "dtype": "float32", "shape": list(array.shape)}
            for name, array in tensors
        ],
        "metadata": dict(metadata or {}),
    }
    manifest_path = destination.with_suffix(destination.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _quantize_tensor(array: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    if array.ndim >= 2:
        quantization_axis = 0
        rows = array.reshape(array.shape[0], -1)
        maximum = np.max(np.abs(rows), axis=1)
        scales = np.maximum(maximum / 127.0, np.finfo(np.float32).tiny).astype("<f4")
        quantized = np.rint(rows / scales[:, None]).clip(-127, 127).astype(np.int8)
        return quantization_axis, scales, quantized.reshape(array.shape)

    quantization_axis = -1
    maximum = float(np.max(np.abs(array))) if array.size else 0.0
    scale = max(maximum / 127.0, float(np.finfo(np.float32).tiny))
    scales = np.asarray([scale], dtype="<f4")
    quantized = np.rint(array / scale).clip(-127, 127).astype(np.int8)
    return quantization_axis, scales, quantized


def _encode_quantized_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> tuple[bytes, list[dict[str, Any]]]:
    output = bytearray(QUANTIZED_MAGIC)
    output.extend(struct.pack("<II", QUANTIZED_FORMAT_VERSION, len(state_dict)))
    tensor_manifest = []

    for name in sorted(state_dict):
        encoded_name = name.encode("utf-8")
        if len(encoded_name) > 0xFFFF:
            raise ValueError(f"tensor name is too long: {name!r}")
        tensor = state_dict[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"state_dict entry {name!r} is not a tensor")
        array = np.asarray(
            tensor.detach().cpu().to(torch.float32).contiguous().numpy(), dtype="<f4"
        )
        if array.ndim > 0x7F:
            raise ValueError(f"tensor has too many dimensions: {name!r}")
        axis, scales, quantized = _quantize_tensor(array)
        raw = quantized.tobytes(order="C")

        output.extend(struct.pack("<H", len(encoded_name)))
        output.extend(encoded_name)
        output.extend(struct.pack("<Bb", array.ndim, axis))
        for dimension in array.shape:
            if dimension < 0 or dimension > 0xFFFFFFFF:
                raise ValueError(f"invalid tensor dimension in {name!r}: {dimension}")
            output.extend(struct.pack("<I", dimension))
        output.extend(struct.pack("<I", len(scales)))
        output.extend(scales.tobytes(order="C"))
        output.extend(struct.pack("<Q", len(raw)))
        output.extend(raw)

        dequantized = quantized.astype(np.float32)
        if axis == 0:
            dequantized = (dequantized.reshape(array.shape[0], -1) * scales[:, None]).reshape(
                array.shape
            )
        else:
            dequantized *= scales[0]
        maximum_error = float(np.max(np.abs(array - dequantized))) if array.size else 0.0
        tensor_manifest.append(
            {
                "name": name,
                "dtype": "int8",
                "shape": list(array.shape),
                "quantization_axis": axis,
                "scale_count": len(scales),
                "maximum_absolute_error": maximum_error,
            }
        )
    return bytes(output), tensor_manifest


def encode_base93(data: bytes) -> str:
    """Encode bytes using printable ASCII that is safe in a Rust string literal."""

    buffer = 0
    bit_count = 0
    output = []
    for byte in data:
        buffer |= byte << bit_count
        bit_count += 8
        if bit_count > 13:
            value = buffer & ((1 << 13) - 1)
            if value > BASE93_THRESHOLD:
                buffer >>= 13
                bit_count -= 13
            else:
                value = buffer & ((1 << 14) - 1)
                buffer >>= 14
                bit_count -= 14
            output.append(BASE93_ALPHABET[value % 93])
            output.append(BASE93_ALPHABET[value // 93])

    if bit_count:
        output.append(BASE93_ALPHABET[buffer % 93])
        if bit_count > 7 or buffer >= 93:
            output.append(BASE93_ALPHABET[buffer // 93])
    return "".join(output)


def write_rust_model_data(compressed: bytes, path: str | Path) -> None:
    """Write compressed model bytes as a dependency-free Rust Base93 constant."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = encode_base93(compressed)
    lines = [
        "// Generated by ahc-ml. Do not edit.",
        "pub const MODEL_DATA_BASE93: &str = concat!(",
    ]
    lines.extend(
        f'    "{encoded[offset : offset + 100]}",' for offset in range(0, len(encoded), 100)
    )
    lines.extend(['    "",', ");", ""])
    destination.write_text("\n".join(lines))


def export_quantized_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
    rust_source: str | Path | None = None,
) -> dict[str, Any]:
    """Quantize tensors to int8, Huffman-compress them, and optionally emit Rust source."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw, tensors = _encode_quantized_state_dict(state_dict)
    huffman_payload, code_lengths = _huffman_compress(raw)
    huffman = bytes(code_lengths) + huffman_payload if code_lengths else raw
    if code_lengths and len(huffman) < len(raw):
        codec = 1
        payload = huffman
        compression = "canonical-huffman"
    else:
        codec = 0
        payload = raw
        compression = "stored"
    compressed = (
        COMPRESSED_MAGIC + struct.pack("<IQB", COMPRESSED_FORMAT_VERSION, len(raw), codec) + payload
    )
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(compressed)
    os.replace(temporary, destination)
    if rust_source is not None:
        write_rust_model_data(compressed, rust_source)

    manifest = {
        "format": "ahc-ml-quantized-weights",
        "format_version": QUANTIZED_FORMAT_VERSION,
        "compression": compression,
        "uncompressed_size": len(raw),
        "compressed_size": len(compressed),
        "compression_ratio": len(compressed) / len(raw),
        "tensor_count": len(tensors),
        "tensors": tensors,
        "metadata": dict(metadata or {}),
    }
    destination.with_suffix(destination.suffix + ".json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def read_quantized_state_dict(path: str | Path) -> dict[str, ExportedTensor]:
    """Reference decoder for the quantized format used by the Rust implementation."""

    compressed = Path(path).read_bytes()
    header_size = len(COMPRESSED_MAGIC) + 13
    if len(compressed) < header_size or compressed[: len(COMPRESSED_MAGIC)] != COMPRESSED_MAGIC:
        raise ValueError("invalid compressed model magic")
    version, raw_size, codec = struct.unpack(
        "<IQB", compressed[len(COMPRESSED_MAGIC) : header_size]
    )
    if version != COMPRESSED_FORMAT_VERSION:
        raise ValueError(f"unsupported compressed model version: {version}")
    payload = compressed[header_size:]
    if codec == 0:
        if len(payload) != raw_size:
            raise ValueError("invalid stored model length")
        raw = payload
    elif codec == 1:
        if len(payload) < 256:
            raise ValueError("truncated Huffman header")
        raw = _huffman_decompress(payload[256:], list(payload[:256]), raw_size)
    else:
        raise ValueError(f"unsupported compression codec: {codec}")
    cursor = 0

    def take(size: int) -> bytes:
        nonlocal cursor
        end = cursor + size
        if end > len(raw):
            raise ValueError("truncated quantized model")
        value = raw[cursor:end]
        cursor = end
        return value

    if take(len(QUANTIZED_MAGIC)) != QUANTIZED_MAGIC:
        raise ValueError("invalid quantized model magic")
    format_version, tensor_count = struct.unpack("<II", take(8))
    if format_version != QUANTIZED_FORMAT_VERSION:
        raise ValueError(f"unsupported quantized model version: {format_version}")

    result = {}
    for _ in range(tensor_count):
        (name_length,) = struct.unpack("<H", take(2))
        name = take(name_length).decode("utf-8")
        rank, axis = struct.unpack("<Bb", take(2))
        shape = tuple(struct.unpack("<I", take(4))[0] for _ in range(rank))
        (scale_count,) = struct.unpack("<I", take(4))
        scales = np.frombuffer(take(scale_count * 4), dtype="<f4")
        (data_length,) = struct.unpack("<Q", take(8))
        expected_length = int(np.prod(shape, dtype=np.int64))
        if data_length != expected_length:
            raise ValueError(f"invalid byte length for tensor {name!r}")
        quantized = np.frombuffer(take(data_length), dtype=np.int8).astype(np.float32)
        if axis == 0:
            if not shape or scale_count != shape[0]:
                raise ValueError(f"invalid per-channel scales for tensor {name!r}")
            values = (quantized.reshape(shape[0], -1) * scales[:, None]).reshape(shape)
        elif axis == -1:
            if scale_count != 1:
                raise ValueError(f"invalid per-tensor scale for tensor {name!r}")
            values = (quantized * scales[0]).reshape(shape)
        else:
            raise ValueError(f"unsupported quantization axis for tensor {name!r}: {axis}")
        if name in result:
            raise ValueError(f"duplicate tensor name: {name!r}")
        result[name] = ExportedTensor(name, "float32", shape, values.copy())
    if cursor != len(raw):
        raise ValueError("unexpected trailing quantized model data")
    return result


def _read_exact(file: BinaryIO, size: int) -> bytes:
    data = file.read(size)
    if len(data) != size:
        raise ValueError("truncated model file")
    return data


def read_exported_state_dict(path: str | Path) -> dict[str, ExportedTensor]:
    """Reference reader used to test the format before the Rust reader exists."""

    result: dict[str, ExportedTensor] = {}
    with Path(path).open("rb") as file:
        if _read_exact(file, len(MAGIC)) != MAGIC:
            raise ValueError("invalid model magic")
        version, tensor_count = struct.unpack("<II", _read_exact(file, 8))
        if version != FORMAT_VERSION:
            raise ValueError(f"unsupported model format version: {version}")

        for _ in range(tensor_count):
            (name_length,) = struct.unpack("<H", _read_exact(file, 2))
            name = _read_exact(file, name_length).decode("utf-8")
            dtype_code, rank = struct.unpack("<BB", _read_exact(file, 2))
            if dtype_code != DTYPE_FLOAT32:
                raise ValueError(f"unsupported dtype code: {dtype_code}")
            shape = tuple(struct.unpack("<I", _read_exact(file, 4))[0] for _ in range(rank))
            (byte_length,) = struct.unpack("<Q", _read_exact(file, 8))
            expected_length = int(np.prod(shape, dtype=np.int64)) * 4
            if byte_length != expected_length:
                raise ValueError(f"invalid byte length for tensor {name!r}")
            flat_values = np.frombuffer(_read_exact(file, byte_length), dtype="<f4")
            values = flat_values.reshape(shape).copy()
            if name in result:
                raise ValueError(f"duplicate tensor name: {name!r}")
            result[name] = ExportedTensor(name, "float32", shape, values)

        if file.read(1):
            raise ValueError("unexpected trailing data in model file")
    return result
