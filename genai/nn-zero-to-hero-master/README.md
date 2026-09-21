# Neural Networks: Zero to Hero

A modernized, classroom-ready collection of materials for Andrej Karpathy's
*Neural Networks: Zero to Hero* course. The course starts from absolute
basics—scalar derivatives and backpropagation—and gradually builds up to
modern character-level language models.

This repository has been refreshed for **June 2026** Python tooling:

- Reproducible environment managed with [`uv`](https://docs.astral.sh/uv/)
- Pinned dependencies in `pyproject.toml` and `uv.lock`
- A reusable `nnzero` package containing the from-scratch implementations
- Modernized notebooks with learning objectives, "why" explanations, and exercises
- Lightweight tests and notebook execution checks

---

## Quick start

```bash
# 1. Install uv if you don't have it
#    https://docs.astral.sh/uv/getting-started/installation/

# 2. Clone or open this repository and sync dependencies
uv sync

# 3. Download the datasets used by the notebooks
uv run python scripts/download_data.py

# 4. Start JupyterLab
uv run jupyter lab
```

Open any notebook under `notebooks/` and run the cells from top to bottom.

> **Note:** Graphviz is required for the micrograd visualization cells. On macOS
> install it with `brew install graphviz`; on Ubuntu/Debian use
> `sudo apt-get install graphviz`. See [`docs/setup.md`](docs/setup.md) for
> platform-specific help.

---

## What's inside

| Notebook | Topic | What you build |
|---|---|---|
| `notebooks/01_micrograd_autograd.ipynb` | Scalar autograd | A tiny `Value` class that supports `+`, `*`, `tanh`, `exp`, `.backward()` |
| `notebooks/02_micrograd_mlp.ipynb` | Multi-layer perceptron | `Neuron`, `Layer`, `MLP`, and a toy training loop |
| `notebooks/03_makemore_bigrams.ipynb` | Bigram language model | Count-based and neural-network bigram models over names |
| `notebooks/04_makemore_mlp.ipynb` | MLP language model | Embeddings, hidden layers, softmax, cross-entropy, train/val/test splits |
| `notebooks/05_makemore_batchnorm.ipynb` | Activations & BatchNorm | Kaiming init, BatchNorm, deep-network diagnostics |
| `notebooks/06_makemore_backprop_ninja.ipynb` | Manual backprop | Backprop through every tensor operation by hand |
| `notebooks/07_makemore_wavenet.ipynb` | WaveNet-style model | Hierarchical character model with `FlattenConsecutive` |

The reusable implementations live in the `nnzero/` package:

- `nnzero/micrograd.py` — `Value`, `Neuron`, `Layer`, `MLP`, `draw_dot`
- `nnzero/layers.py` — `Linear`, `BatchNorm1d`, `Tanh`, `Embedding`, `FlattenConsecutive`, `Sequential`
- `nnzero/utils.py` — `build_vocab`, `build_dataset`, `split_dataset`, `load_names`, `sample_from_model`, `set_seed`

---

## Teaching with this repo

Each notebook now includes:

- **Learning objectives** at the top
- **"Why this step"** Markdown cells explaining the intuition behind key ideas
- **"Try it yourself"** exercises with hidden solutions
- Clean, output-stripped cells so students run everything themselves

For a concept map, common pitfalls, and extension ideas, see the guides in
[`docs/lectures/`](docs/lectures/).

---

## Documentation site

An interactive MkDocs Material site covers all seven modules in depth, with
Mermaid diagrams and progressive-disclosure sections (click to expand):

```bash
uv sync --group docs
uv run mkdocs serve   # http://127.0.0.1:8000
```

Source lives in [`docs-site/`](docs-site/); config is [`mkdocs.yml`](mkdocs.yml).
It deploys to GitHub Pages automatically on push to `master`
(see [`.github/workflows/docs.yml`](.github/workflows/docs.yml)).

---

## Development

```bash
# Run the test suite
uv run pytest

# Check code style
uv run ruff check nnzero/ scripts/ tests/
uv run ruff format --check nnzero/ scripts/ tests/

# Execute all notebooks (slow)
uv run python scripts/run_all_notebooks.py
```

Notebook outputs are stripped automatically by `nbstripout` on commit.

---

## Videos

The original YouTube lectures are linked in each notebook and in the guides
under `docs/lectures/`.

---

## License

MIT
