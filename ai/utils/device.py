"""Device auto-detect helper.

Cavekit: cavekit-model-exploration.md R8.

Priority order: CUDA -> ROCm -> CPU. Returns a `torch.device` instance plus
a human-readable device name. No `.cuda()` calls anywhere -- all placement
must use `.to(device)` so the same code works on AMD/ROCm and CPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceInfo:
    device: torch.device
    backend: str  # "cuda" | "rocm" | "cpu"
    name: str


def _is_rocm_build() -> bool:
    """torch.version.hip is set on official ROCm wheels."""
    return getattr(torch.version, "hip", None) is not None


def get_device() -> DeviceInfo:
    """Pick the best available device in priority order CUDA -> ROCm -> CPU.

    Both CUDA and ROCm builds of PyTorch expose `torch.cuda.is_available()` and
    use `torch.device("cuda")`. The two backends are distinguished by
    `torch.version.hip`.
    """
    if torch.cuda.is_available():
        idx = 0
        device = torch.device(f"cuda:{idx}")
        if _is_rocm_build():
            backend = "rocm"
            name = f"ROCm: {torch.cuda.get_device_name(idx)}"
        else:
            backend = "cuda"
            name = f"CUDA: {torch.cuda.get_device_name(idx)}"
        return DeviceInfo(device=device, backend=backend, name=name)

    try:
        import torch_directml
        device = torch_directml.device()
        return DeviceInfo(device=device, backend="directml", name="DirectML: AMD GPU")
    except ImportError:
        pass

    return DeviceInfo(device=torch.device("cpu"), backend="cpu", name="CPU")


def describe_device(info: DeviceInfo | None = None) -> str:
    """Return a one-line summary suitable for printing at notebook startup."""
    info = info or get_device()
    return f"[device] backend={info.backend} device={info.device} name={info.name}"


if __name__ == "__main__":
    print(describe_device())
