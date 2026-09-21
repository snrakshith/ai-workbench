# Hardware Detection

Every notebook and script in this primer starts with the same call:

```python
from utils.device import print_hardware_report
device = print_hardware_report()
```

This page explains what that call does and why it's written the way it is — the logic lives in
[`utils/device.py`](https://github.com/sourangshupal/pytorch-primer/blob/main/utils/device.py).

## The problem it solves

PyTorch code should be **device-agnostic**: the exact same script should run correctly whether
the machine has an NVIDIA GPU (CUDA), an Apple Silicon chip (MPS), or neither (CPU) — without the
student having to edit a single line. Getting this right by hand means checking several APIs in
the right order and handling machines where none of them are available.

## Detection priority

`get_device()` picks a `torch.device` using this priority order:

1. **`torch.accelerator`** — PyTorch 2.x's modern, unified API. One call
   (`torch.accelerator.current_accelerator()`) abstracts over CUDA, MPS, XPU (Intel), and MTIA
   (Meta's own accelerator) — whichever is present — replacing the older pattern of checking
   `torch.cuda.is_available()` then `torch.backends.mps.is_available()` one at a time. This is
   now the officially recommended way to write device-agnostic PyTorch code.
2. **Manual CUDA check** (`torch.cuda.is_available()`) — fallback for older PyTorch builds that
   predate `torch.accelerator`.
3. **Manual MPS check** (`torch.backends.mps.is_available()`) — Apple Silicon fallback.
4. **CPU** — always available, the final fallback.

```python
def get_device() -> torch.device:
    accel_device = _accelerator_device()
    if accel_device is not None:
        return accel_device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

`_accelerator_device()` is written defensively — it wraps the `torch.accelerator` call in a
`try`/`except` and returns `None` on any failure, since some pre-release `torch.accelerator`
builds have had slightly different call signatures. If anything about that modern API doesn't
behave as expected, the code silently falls back to the manual checks rather than crashing.

## `print_hardware_report()`

The function every notebook actually calls. It builds a `HardwareReport` (PyTorch version, Python
version, platform string, selected device, CUDA availability + GPU names, MPS availability,
whether the modern accelerator API was used), prints a human-readable summary, and **returns the
`torch.device`** so you can immediately use it:

```python
device = print_hardware_report()
model.to(device)
```

Example output on a CUDA machine with two GPUs:

```text
PyTorch version : 2.13.0
Python version  : 3.12.4
Platform        : Linux-6.8.0-x86_64-with-glibc2.35
Selected device : cuda
torch.accelerator : available (type=cuda)
CUDA available  : True
CUDA GPU count  : 2
  - cuda:0       : NVIDIA A100-SXM4-80GB
  - cuda:1       : NVIDIA A100-SXM4-80GB
MPS available   : False
```

## Why centralize this at all?

Without a shared helper, every notebook would either hardcode `"cuda"` (breaking on Apple Silicon
and CPU-only machines) or re-implement the same three-way check independently — and any future
fix (e.g. adding XPU support, or working around a `torch.accelerator` quirk) would need to be
applied in 17 different places. Centralizing it in `utils/device.py` means every notebook and
script in this repository uses the *exact same* detection policy, and improving it once improves
it everywhere.

## Where this is used

- [Track 1 — GPU Training](track1-raschka/08-gpu-training.md) uses `get_device()` explicitly to
  move a model and data onto whatever device is available.
- [Track 2 — Build Model Deep Dive](track2-official/14-build-model-deep-dive.md) discusses the
  `torch.accelerator` API directly, since that's the tutorial this notebook was adapted from.
- Every other notebook calls `print_hardware_report()` in its first cell, purely as a sanity
  check, even when the notebook's content doesn't otherwise touch devices.
