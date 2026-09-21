# pytorch-primer

📖 **[Read the full documentation](https://sourangshupal.github.io/pytorch-primer/)**

A hands-on "PyTorch in ~1-2 hours" teaching kit. Two complementary tracks:

- **Notebooks 01-09**: adapted from Sebastian Raschka's [*PyTorch in One Hour*](https://sebastianraschka.com/teaching/pytorch-1h/)
  (2025) - builds every concept from small, hand-written toy tensors. Covers tensors, autograd,
  `nn.Module`, data loading, training loops, saving/loading models, and single-/multi-GPU training.
- **Notebooks 10-17**: adapted from the official [PyTorch "Learn the Basics"](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
  series - fills in what the blog doesn't cover (Quickstart on a real dataset, Transforms, the
  NumPy tensor bridge, `torch.accelerator`, and more) using the real FashionMNIST dataset.

## Quick start

```bash
uv sync                              # installs the latest PyTorch + deps for your platform
uv run main.py                       # sanity check: prints PyTorch version + detected hardware
uv run jupyter lab notebooks/        # open the notebooks
```

`uv sync` automatically installs the right PyTorch build for your machine: CUDA-enabled on an
NVIDIA GPU box, MPS-ready on Apple Silicon, CPU-only otherwise. See the
[Getting Started](https://sourangshupal.github.io/pytorch-primer/getting-started/) docs page if
you need a non-default install (e.g. a specific CUDA version). Notebooks 10-17 additionally use
`torchvision` (FashionMNIST + transforms) and `pandas` (one illustrative custom-Dataset example) -
both already declared in `pyproject.toml`.

## What's in here

```
notebooks/   17 notebooks: 01-09 (Raschka's primer) + 10-17 (official PyTorch basics gap-fill)
scripts/     Standalone .py equivalents of notebooks 01-08, plus scripts/ddp_train.py
             (the only file meant to be run via `torchrun`, not directly)
docs/        Source for the published documentation site (mkdocs) - one page per notebook,
             plus a Getting Started guide and a session timing table for a 1-2 hour class
utils/       device.py - shared hardware-detection helper used everywhere (torch.accelerator
             when available, falling back to manual CUDA/MPS checks)
```

### Track 1: notebooks 01-09 (Raschka's *PyTorch in One Hour*)

| # | Topic                                    | Notebook                                   | Script |
|---|-------------------------------------------|----------------------------------------------|--------|
| 1 | What is PyTorch + installation            | `01_what_is_pytorch.ipynb`                   | `01_what_is_pytorch.py` |
| 2 | Tensors                                   | `02_tensors.ipynb`                           | `02_tensors.py` |
| 3-4 | Computation graphs + autograd            | `03_computation_graphs_and_autograd.ipynb`   | `03_computation_graphs_and_autograd.py` |
| 5 | Building a neural network (`nn.Module`)   | `04_building_neural_networks.ipynb`          | `04_building_neural_networks.py` |
| 6 | Data loaders (`Dataset` / `DataLoader`)    | `05_data_loaders.ipynb`                      | `05_data_loaders.py` |
| 7 | Training loop                             | `06_training_loop.ipynb`                     | `06_training_loop.py` |
| 8 | Saving / loading models                   | `07_saving_and_loading_models.ipynb`         | `07_saving_and_loading_models.py` |
| 9.1-9.2 | GPU training (device-agnostic)      | `08_gpu_training.ipynb`                      | `08_gpu_training.py` |
| 9.3 | Multi-GPU training (DDP) - concept + code walkthrough | `09_multi_gpu_ddp.ipynb`   | `../scripts/ddp_train.py` (run via `torchrun`) |

### Track 2: notebooks 10-17 (official PyTorch "Learn the Basics", gap-fill)

Each of these complements (doesn't replace) its Track 1 counterpart - they cover what the blog
didn't, using the real FashionMNIST dataset instead of toy tensors.

| # | Topic | Notebook | Complements |
|---|-------|----------|--------------|
| 10 | Quickstart - full end-to-end pipeline on real data | `10_official_quickstart.ipynb` | all of 01-09 |
| 11 | Tensors deep dive (indexing, `cat`, in-place ops, NumPy bridge) | `11_official_tensors_deep_dive.ipynb` | `02_tensors.ipynb` |
| 12 | Real dataset + custom file-backed `Dataset` | `12_official_datasets_and_dataloaders.ipynb` | `05_data_loaders.ipynb` |
| 13 | Transforms (`v2.ToImage`/`ToDtype`, `Lambda`, one-hot) | `13_official_transforms.ipynb` | *(new topic)* |
| 14 | Build Model deep dive (`nn.Flatten`, `nn.Softmax`, named params) | `14_official_build_model_deep_dive.ipynb` | `04_building_neural_networks.ipynb` |
| 15 | Autograd deep dive (`detach`, disabling tracking, Jacobians) | `15_official_autograd_deep_dive.ipynb` | `03_computation_graphs_and_autograd.ipynb` |
| 16 | Optimization loop (hyperparameters, `train_loop`/`test_loop`) | `16_official_optimization_loop.ipynb` | `06_training_loop.ipynb` |
| 17 | Save/load with shapes + labeled prediction | `17_official_save_load_and_predict.ipynb` | `07_saving_and_loading_models.ipynb` |

## Running the multi-GPU demo

`scripts/ddp_train.py` needs 2+ CUDA GPUs and must run as a standalone script (not in a
notebook - DDP needs one OS process per GPU):

```bash
torchrun --nproc_per_node=2 scripts/ddp_train.py
```

`notebooks/09_multi_gpu_ddp.ipynb` walks through the same code without requiring multiple GPUs
to follow along - use it for the live conceptual explanation, and point to the script for anyone
who wants to run it afterward on real multi-GPU hardware.

## Hardware detection

Every notebook/script starts by calling `utils/device.py`:

```python
from utils.device import print_hardware_report
device = print_hardware_report()   # prints PyTorch/Python version + CUDA/MPS/CPU info
                                    # returns a torch.device: CUDA > MPS > CPU
```

This is what makes every example here run unmodified whether the student's machine has an
NVIDIA GPU, Apple Silicon, or neither.

## Teaching this in a session

See the [Teaching Guide](https://sourangshupal.github.io/pytorch-primer/teaching-guide/) for a
suggested minute-by-minute timing table for a 1-hour lightning session or a 2-hour session with
exercises.
