# Setup Guide

This repository uses [`uv`](https://docs.astral.sh/uv/) for environment and
dependency management. The instructions below should work on macOS, Linux, and
Windows (with WSL recommended).

---

## 1. Install uv

Follow the official instructions: https://docs.astral.sh/uv/getting-started/installation/

Quick checks:

```bash
uv --version
# uv 0.9.13 or newer is recommended
```

---

## 2. Sync the project environment

From the repository root:

```bash
uv sync
```

This creates a `.venv/` directory with Python 3.12 and all project dependencies.
The exact versions are recorded in `uv.lock`.

---

## 3. Download datasets

```bash
uv run python scripts/download_data.py
```

This downloads `names.txt` (the baby-names dataset used by makemore) into
`data/names.txt`.

---

## 4. Install Graphviz (required for micrograd plots)

The `graphviz` Python package is already installed in the virtual environment,
but it needs the system Graphviz binaries to render images.

### macOS

```bash
brew install graphviz
```

### Ubuntu / Debian

```bash
sudo apt-get update
sudo apt-get install graphviz
```

### Windows

Download and install from https://graphviz.org/download/ and add the `bin`
directory to your system `PATH`.

Verify with:

```bash
dot -V
```

---

## 5. Launch Jupyter

```bash
uv run jupyter lab
```

Open any notebook under `notebooks/` and run the cells from top to bottom.

---

## 6. Running tests

```bash
uv run pytest
```

To execute all notebooks (slower):

```bash
uv run python scripts/run_all_notebooks.py
```

---

## GPU support

By default the project installs the CPU build of PyTorch, which is enough for
the course. If you have an NVIDIA GPU and want CUDA acceleration, edit
`pyproject.toml` to use the CUDA index URL for `torch` and re-run `uv sync`.
Most notebooks run in seconds on CPU anyway.

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'nnzero'` | Not running inside the venv | Use `uv run ...` or activate `.venv` |
| `ExecutableNotFound: failed to execute ['dot']` | Graphviz binaries missing | Install Graphviz system package |
| `FileNotFoundError: data/names.txt` | Dataset not downloaded | Run `uv run python scripts/download_data.py` |
| Notebook cells produce different random numbers | Global RNG vs. local generator | Notebooks use `set_seed`; restart kernel and re-run |
