from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class DeviceInfo:
    requested: str
    selected: str
    name: str
    cuda_available: bool
    mps_available: bool

    def to_dict(self) -> dict[str, str | bool]:
        return asdict(self)


def mps_is_available() -> bool:
    return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())


def select_device(requested: str = "auto") -> tuple[torch.device, DeviceInfo]:
    requested = requested.lower()
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"unsupported device {requested!r}; use auto, cpu, cuda, or mps")

    cuda_available = torch.cuda.is_available()
    mps_available = mps_is_available()

    if requested == "auto":
        selected = "cuda" if cuda_available else "mps" if mps_available else "cpu"
    else:
        selected = requested

    if selected == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested but is not available")
    if selected == "mps" and not mps_available:
        raise RuntimeError("MPS was requested but is not available")

    if selected == "cuda":
        name = torch.cuda.get_device_name(torch.cuda.current_device())
    elif selected == "mps":
        name = "Apple Metal Performance Shaders"
    else:
        name = "CPU"

    device = torch.device(selected)
    return device, DeviceInfo(
        requested=requested,
        selected=selected,
        name=name,
        cuda_available=cuda_available,
        mps_available=mps_available,
    )
