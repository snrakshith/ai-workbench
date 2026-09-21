# Getting Started

This page gets your machine ready to run every notebook and script in this repository — whether
you're on an NVIDIA GPU workstation, an Apple Silicon laptop, or a plain CPU machine.

## 1. Install `uv`

This project uses [`uv`](https://docs.astral.sh/uv/) as its package manager instead of `pip` or
`conda` directly — it resolves dependencies faster and, importantly for this primer, picks the
**correct PyTorch build for your platform automatically**.

If you don't have it yet:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS / Linux
```

(See the [uv installation docs](https://docs.astral.sh/uv/getting-started/installation/) for
Windows and other options.)

## 2. Clone the repository and install dependencies

```bash
git clone https://github.com/sourangshupal/pytorch-primer.git
cd pytorch-primer
uv sync
```

`uv sync` reads `pyproject.toml` and installs everything needed: `torch`, `torchvision`, `numpy`,
`matplotlib`, `pandas`, and Jupyter. Crucially, it also figures out **which PyTorch build your
machine needs**:

| Your machine | What `uv sync` installs |
|---|---|
| NVIDIA GPU (CUDA) | CUDA-enabled PyTorch build, automatically |
| Apple Silicon (M1/M2/M3/M4+) | Standard PyTorch build — MPS (Metal) acceleration ships built in, no extra flags |
| CPU only | CPU-only build — every example here still runs correctly, just without GPU speedups (the models are intentionally tiny) |

!!! tip "Need a specific CUDA version?"
    If you need a non-default CUDA build (e.g. matching an older driver), get the exact index
    URL for your platform from [pytorch.org](https://pytorch.org) and run:
    ```bash
    uv add torch --index-url <cpu-or-cuda-index-url-from-pytorch.org>
    ```

## 3. Run the sanity check

```bash
uv run main.py
```

This prints your PyTorch version, Python version, platform, and which device (CUDA / MPS / CPU)
was detected — the exact same check every notebook and script runs as its first step. If this
prints without errors, your environment is ready.

Example output on an Apple Silicon laptop:

```text
pytorch-primer: environment check
----------------------------------
PyTorch version : 2.13.0
Python version  : 3.12.4
Platform        : macOS-14.5-arm64-arm-64bit
Selected device : mps
torch.accelerator : available (type=mps)
CUDA available  : False
MPS available   : True
```

## 4. Open the notebooks

```bash
uv run jupyter lab notebooks/
```

Work through them in numeric order — `01_what_is_pytorch.ipynb` through `09_multi_gpu_ddp.ipynb`
for Track 1, then `10_official_quickstart.ipynb` through `17_official_save_load_and_predict.ipynb`
for Track 2. Each Track 2 notebook names the Track 1 notebook it complements, so you always know
what you're building on.

## 5. Or run the `.py` scripts directly

Notebooks 01-08 each have a plain-Python equivalent in `scripts/` — useful if you prefer an
editor + terminal workflow over Jupyter, or want to diff/version the code more easily:

```bash
uv run scripts/06_training_loop.py
```

The one exception is `scripts/ddp_train.py`, which **must** be launched with `torchrun` (see
[Multi-GPU DDP](track1-raschka/09-multi-gpu-ddp.md) and the
[DDP script walkthrough](ddp-script-walkthrough.md)) — it needs one OS process per GPU, which a
single `python`/notebook process can't provide.

## Next steps

- New to PyTorch entirely? Start at [Track 1 → What is PyTorch?](track1-raschka/01-what-is-pytorch.md)
- Want to understand the hardware-detection helper first? See [Hardware Detection](hardware-detection.md)
- Teaching a class? See the [Teaching Guide](teaching-guide.md) for a timing table
