# Agent Guide: nn-zero-to-hero

This file is written for AI coding agents working on the **Neural Networks: Zero to Hero** repository. It is an educational Python project that packages the from-scratch neural-network implementations from Andrej Karpathy's course into a modern, reproducible environment.

---

## Project overview

- **Name:** `nn-zero-to-hero`
- **Package:** `nnzero`
- **Language:** Python 3.12+
- **Purpose:** Classroom-ready teaching materials for the *Neural Networks: Zero to Hero* course, refreshed for June 2026 tooling.
- **Audience:** Students learning derivatives, backpropagation, MLPs, embeddings, BatchNorm, and character-level language models.

The repository contains:

- A reusable `nnzero` Python package with from-scratch implementations.
- Seven Jupyter notebooks under `notebooks/` that correspond to the course lectures.
- Lecture guides under `docs/lectures/` with key concepts, pitfalls, and extensions.
- Supporting docs: `docs/setup.md`, `docs/exercises.md`, `docs/glossary.md`.
- A small pytest suite under `tests/`.
- Helper scripts under `scripts/` for data download and notebook execution.

---

## Technology stack

- **Environment manager:** [`uv`](https://docs.astral.sh/uv/)
- **Build backend:** `hatchling`
- **Python version:** 3.12 (recorded in `.python-version`)
- **Core dependencies:**
  - `torch>=2.13.0` (CPU build by default)
  - `numpy>=2.5.1`
  - `scipy>=1.18.0`
  - `matplotlib>=3.11.1`
  - `graphviz>=0.21` (Python package; system Graphviz binaries also required for plots)
  - `jupyterlab>=4.6.1`, `notebook>=7.6.0`, `ipywidgets>=8.1.8`
  - `tqdm>=4.69.0`
- **Dev dependencies:**
  - `pytest>=9.1.1`
  - `ruff>=0.15.22`
  - `nbstripout>=0.9.1`
  - `nbval>=0.11.0`
  - `papermill>=2.7.0`

---

## Build and run commands

All commands should be run from the repository root. Use `uv run` so commands execute inside the project virtual environment (`.venv/`).

### Sync the environment

```bash
uv sync
```

This installs Python 3.12 and all dependencies recorded in `uv.lock`.

### Download datasets

```bash
uv run python scripts/download_data.py
```

Downloads `data/names.txt` (baby-names dataset used by the makemore notebooks).

### Launch Jupyter

```bash
uv run jupyter lab
```

Open any notebook under `notebooks/` and run cells top to bottom.

### Run tests

```bash
uv run pytest
```

Runs the pytest suite in `tests/`. Configured in `pyproject.toml` with `testpaths = ["tests"]` and verbose output.

### Execute all notebooks

```bash
uv run python scripts/run_all_notebooks.py
```

By default this runs in quick mode (`NNZERO_QUICK_RUN=1`), using fewer training steps. To run the full notebooks:

```bash
NNZERO_QUICK_RUN=0 uv run python scripts/run_all_notebooks.py
```

### Code style checks

```bash
uv run ruff check nnzero/ scripts/ tests/
uv run ruff format --check nnzero/ scripts/ tests/
```

To auto-fix issues:

```bash
uv run ruff check --fix nnzero/ scripts/ tests/
uv run ruff format nnzero/ scripts/ tests/
```

---

## Code organization

```
.
├── nnzero/                 # Reusable package
│   ├── __init__.py         # Public API exports
│   ├── micrograd.py        # Scalar autograd: Value, Neuron, Layer, MLP, draw_dot
│   ├── layers.py           # PyTorch-like layers: Linear, BatchNorm1d, Tanh, Embedding, FlattenConsecutive, Sequential
│   └── utils.py            # Dataset helpers, sampling, reproducibility
├── notebooks/              # Seven course notebooks (outputs stripped)
├── tests/                  # pytest tests
│   ├── test_micrograd.py
│   ├── test_layers.py
│   └── test_utils.py
├── scripts/                # Automation scripts
│   ├── download_data.py
│   └── run_all_notebooks.py
├── docs/                   # Documentation and lecture guides
│   ├── setup.md
│   ├── exercises.md
│   ├── glossary.md
│   └── lectures/
├── pyproject.toml          # Project metadata, dependencies, tool config
├── uv.lock                 # Pinned dependency lockfile
├── .python-version         # Python 3.12
└── README.md
```

### Main modules

- `nnzero/micrograd.py` — scalar autograd engine and tiny MLP.
  - `Value`: scalar node supporting `+`, `*`, `/`, `**`, `tanh`, `exp`, `relu`, and `backward()`.
  - `Neuron`, `Layer`, `MLP`: hand-built network classes.
  - `trace`, `draw_dot`: Graphviz visualization helpers.

- `nnzero/layers.py` — PyTorch-style tensor layers.
  - `Module`: base class with `__call__`, `parameters()`, `zero_grad()`.
  - `Linear`: fully connected layer with Kaiming init.
  - `BatchNorm1d`: batch normalization with running stats and train/eval modes.
  - `Tanh`, `Embedding`, `FlattenConsecutive`, `Sequential`.

- `nnzero/utils.py` — shared helpers.
  - `set_seed`, `build_vocab`, `build_dataset`, `split_dataset`, `load_names`, `sample_from_model`.

### Notebooks

| Notebook | Topic |
|---|---|
| `01_micrograd_autograd.ipynb` | Scalar autograd and computation graphs |
| `02_micrograd_mlp.ipynb` | Tiny MLP with micrograd |
| `03_makemore_bigrams.ipynb` | Bigram language models |
| `04_makemore_mlp.ipynb` | MLP language model |
| `05_makemore_batchnorm.ipynb` | Deep networks, Kaiming init, BatchNorm |
| `06_makemore_backprop_ninja.ipynb` | Manual backpropagation through tensors |
| `07_makemore_wavenet.ipynb` | Hierarchical WaveNet-style character model |

---

## Code style guidelines

The project uses **Ruff** for linting and formatting, configured in `pyproject.toml`:

- Target Python version: **3.12**
- Line length: **100** (formatter enforces it; `E501` is ignored in lint)
- Quote style: **double**
- Indent style: **spaces**
- Docstring convention: **Google**
- Lint rules enabled: `E`, `F`, `I`, `N`, `W`, `UP`, `B`, `C4`

Additional conventions observed in the codebase:

- Use `from __future__ import annotations` at the top of each module.
- Use type hints; prefer `collections.abc` for abstract types.
- Prefer local `torch.Generator` instances over the global RNG for reproducibility.
- Keep the `nnzero` package dependency-light and educational in style.
- Notebooks are **excluded** from Ruff (`extend-exclude = ["notebooks"]`).

---

## Testing instructions

- Unit tests live in `tests/` and are run with `uv run pytest`.
- Tests cover:
  - `Value` operations and gradients against finite differences.
  - `Neuron`, `Layer`, `MLP`, and `draw_dot` behavior.
  - Layer shapes, parameters, BatchNorm running stats, and train/eval modes.
  - Utility helpers for vocab, dataset, split, seeding, and file loading.
- Notebook execution is tested via `uv run python scripts/run_all_notebooks.py`.
- `nbval` is installed in the dev group but is not part of the default pytest invocation.

When adding a new feature, add a corresponding test in `tests/` and run the full test suite before finishing.

---

## Development workflow conventions

- **Notebook outputs are stripped.** The repo uses `nbstripout` to keep notebooks clean. Do not commit executed notebook outputs.
- **Pinned dependencies.** `uv.lock` is the source of truth. After changing `pyproject.toml`, run `uv lock` or `uv sync` to update it.
- **Dataset is downloaded, not committed.** `data/names.txt` is produced by `scripts/download_data.py`. `data/raw/` and `data/processed/` are gitignored.
- **Quick-run mode.** `scripts/run_all_notebooks.py` sets `NNZERO_QUICK_RUN=1` automatically so CI stays fast.

---

## Security considerations

- `scripts/download_data.py` downloads a public dataset from `raw.githubusercontent.com` using `urllib.request.urlopen`. It is intentional and low risk.
- No API keys, credentials, or environment-secret handling is present in the codebase.
- System Graphviz binaries are required for `draw_dot` visualization cells. Students must install them separately; see `docs/setup.md` for platform instructions.
- By default the project installs the CPU build of PyTorch. CUDA support requires editing `pyproject.toml` to use the CUDA index URL and re-running `uv sync`.

---

## Common issues

| Problem | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'nnzero'` | Not running inside the venv | Use `uv run ...` or activate `.venv` |
| `ExecutableNotFound: failed to execute ['dot']` | Graphviz system binaries missing | Install Graphviz (`brew install graphviz`, `apt-get install graphviz`, etc.) |
| `FileNotFoundError: data/names.txt` | Dataset not downloaded | Run `uv run python scripts/download_data.py` |
| Notebook randomness differs between runs | Global RNG vs. local generator | Restart the kernel and re-run; notebooks use `set_seed` |

---

## Useful references

- `pyproject.toml` — dependencies, Ruff config, pytest config.
- `docs/setup.md` — detailed environment setup.
- `docs/exercises.md` — exercise index across notebooks.
- `docs/glossary.md` — course terminology.
- `docs/lectures/*.md` — per-lecture concept guides and extension ideas.
