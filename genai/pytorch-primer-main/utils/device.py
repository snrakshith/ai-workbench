"""
Shared hardware-detection helper used across every notebook and script in this
primer.

PyTorch code should be written to be *device-agnostic*: it should run
unmodified whether the machine has an NVIDIA GPU (CUDA), an Apple Silicon
chip (MPS), or neither (CPU). This module centralizes that detection logic
so every notebook/script in this project uses the exact same policy instead
of re-implementing it.

Modern PyTorch (2.x) ships a unified `torch.accelerator` API that abstracts
over CUDA, MPS, XPU, and MTIA in one call - this is now the officially
recommended way to write device-agnostic code, replacing the older pattern of
manually checking `torch.cuda.is_available()` then `torch.backends.mps.is_available()`
one by one. We use `torch.accelerator` when available and transparently fall
back to the manual CUDA -> MPS -> CPU check on older PyTorch builds that don't
have it yet.

Priority order: torch.accelerator (CUDA/MPS/XPU/MTIA, whichever is present)
> manual CUDA check > manual MPS (Apple Silicon) check > CPU.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field

import torch


@dataclass
class HardwareReport:
    torch_version: str
    python_version: str
    platform: str
    device: torch.device
    cuda_available: bool
    cuda_device_count: int = 0
    cuda_device_names: list[str] = field(default_factory=list)
    mps_available: bool = False
    accelerator_api_used: bool = False
    accelerator_type: str | None = None

    def summary(self) -> str:
        lines = [
            f"PyTorch version : {self.torch_version}",
            f"Python version  : {self.python_version}",
            f"Platform        : {self.platform}",
            f"Selected device : {self.device}",
        ]
        if self.accelerator_api_used:
            lines.append(f"torch.accelerator : available (type={self.accelerator_type})")
        else:
            lines.append("torch.accelerator : not available on this PyTorch build - using manual CUDA/MPS checks")
        lines.append(f"CUDA available  : {self.cuda_available}")
        if self.cuda_available:
            lines.append(f"CUDA GPU count  : {self.cuda_device_count}")
            for i, name in enumerate(self.cuda_device_names):
                lines.append(f"  - cuda:{i}       : {name}")
        lines.append(f"MPS available   : {self.mps_available}")
        return "\n".join(lines)


def _accelerator_device() -> torch.device | None:
    """
    Try PyTorch's modern unified `torch.accelerator` API (CUDA/MPS/XPU/MTIA in
    one call). Returns None if this PyTorch build predates that API, so the
    caller can fall back to manual per-backend checks.
    """
    accelerator = getattr(torch, "accelerator", None)
    if accelerator is None:
        return None
    try:
        if accelerator.is_available():
            return torch.device(accelerator.current_accelerator().type)
    except Exception:
        # Be defensive: some pre-release torch.accelerator builds have a
        # slightly different call signature. Fall back rather than crash.
        return None
    return None


def get_device() -> torch.device:
    """
    Return the best available torch.device for this machine.

    Preference order: torch.accelerator (modern, unified CUDA/MPS/XPU/MTIA
    detection) -> manual CUDA check -> manual MPS (Apple Silicon) check -> CPU.

    This is the single function every notebook/script in this repo calls
    instead of hardcoding "cuda" or "cpu".
    """
    accel_device = _accelerator_device()
    if accel_device is not None:
        return accel_device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_hardware_report() -> HardwareReport:
    """Collect a structured snapshot of the available compute hardware."""
    cuda_available = torch.cuda.is_available()
    cuda_count = torch.cuda.device_count() if cuda_available else 0
    cuda_names = (
        [torch.cuda.get_device_name(i) for i in range(cuda_count)]
        if cuda_available
        else []
    )
    mps_available = torch.backends.mps.is_available()

    accel_device = _accelerator_device()

    return HardwareReport(
        torch_version=torch.__version__,
        python_version=platform.python_version(),
        platform=platform.platform(),
        device=get_device(),
        cuda_available=cuda_available,
        cuda_device_count=cuda_count,
        cuda_device_names=cuda_names,
        mps_available=mps_available,
        accelerator_api_used=accel_device is not None,
        accelerator_type=accel_device.type if accel_device is not None else None,
    )


def print_hardware_report() -> torch.device:
    """
    Print a human-readable hardware report and return the recommended
    torch.device. Meant to be the first cell/first line of every
    notebook/script: `device = print_hardware_report()`.
    """
    report = get_hardware_report()
    print(report.summary())
    return report.device


if __name__ == "__main__":
    print_hardware_report()
