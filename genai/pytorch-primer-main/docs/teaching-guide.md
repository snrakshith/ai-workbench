# Teaching Guide

This page is for instructors running this primer as a live session. It's adapted from the
original written handout and mirrors Track 1 (`notebooks/01-09`).

**Audience:** anyone new to PyTorch who wants the essentials needed to read and write deep
learning code (including LLM code) without wading through the entire library.

**Format:** works as a 1-hour lightning session (skip the optional/deep-dive callouts) or a
2-hour session with hands-on exercises after each part.

## Suggested session timing

| Time | Section | Notebook / Script |
|---|---|---|
| 0:00-0:10 | 1. What is PyTorch + install + hardware check | `notebooks/01_what_is_pytorch.ipynb` |
| 0:10-0:20 | 2. Tensors | `notebooks/02_tensors.ipynb` |
| 0:20-0:35 | 3-4. Computation graphs + autograd | `notebooks/03_computation_graphs_and_autograd.ipynb` |
| 0:35-0:50 | 5. Building a neural network (`nn.Module`) | `notebooks/04_building_neural_networks.ipynb` |
| 0:50-1:00 | 6. Data loaders | `notebooks/05_data_loaders.ipynb` |
| *(break for a 1-hr session; continue for 2-hr)* | | |
| 1:00-1:15 | 7. Training loop | `notebooks/06_training_loop.ipynb` |
| 1:15-1:20 | 8. Saving / loading models | `notebooks/07_saving_and_loading_models.ipynb` |
| 1:20-1:35 | 9.1-9.2 GPU training | `notebooks/08_gpu_training.ipynb` |
| 1:35-1:50 | 9.3 Multi-GPU (DDP) — concept + code walkthrough | `notebooks/09_multi_gpu_ddp.ipynb`, `scripts/ddp_train.py` |
| 1:50-2:00 | Wrap-up, summary, further reading, Q&A | this page |

For a strict 1-hour session: cover sections 1-7 hands-on, and treat 8-9 as a fast walkthrough
(saving/loading is one code cell; GPU/DDP is mostly conceptual, using the same figures described
in [GPU Training](track1-raschka/08-gpu-training.md) and [Multi-GPU DDP](track1-raschka/09-multi-gpu-ddp.md)).

If time allows, Track 2 (`notebooks/10-17`) makes good **homework/follow-up material** — it
revisits every Track 1 topic against a real dataset (FashionMNIST) rather than toy tensors, for
students who want to see the full picture after the live session.

## Summary — what students should walk away with

- PyTorch = a tensor library + an automatic differentiation engine (autograd) + a deep learning
  utility library.
- Tensors are array-like data structures (scalars, vectors, matrices, and higher-rank arrays),
  similar to NumPy arrays, that can run on CPU or GPU.
- Autograd computes gradients automatically via `.backward()` — no manual calculus needed to
  train a neural network.
- `torch.nn.Module` provides building blocks for custom neural network architectures.
- `Dataset` + `DataLoader` set up efficient, batched, shuffled data pipelines.
- Training on a CPU or single GPU is straightforward and requires no special setup.
- `DistributedDataParallel` is the simplest way to scale training across multiple GPUs.

## Further reading to hand out

- *Machine Learning with PyTorch and Scikit-Learn* (2022) — Raschka, Liu, Mirjalili. ISBN 978-1801819312
- *Deep Learning with PyTorch* (2021) — Stevens, Antiga, Viehmann. ISBN 978-1617295263
- Lecture: [Tensors in Deep Learning](https://www.youtube.com/watch?v=JXfDlgrfOBY) (15 min)
- [Model Evaluation, Model Selection, and Algorithm Selection in ML](https://arxiv.org/abs/1811.12808) — Raschka (2018)
- [Introduction to Calculus](https://sebastianraschka.com/pdf/supplementary/calculus.pdf) — Raschka
- [Finetuning LLMs on a Single GPU Using Gradient Accumulation](https://sebastianraschka.com/blog/2023/llm-grad-accumulation.html) — Raschka
- [Introducing PyTorch Fully Sharded Data Parallel (FSDP) API](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/)
- Original source Track 1 is adapted from: [PyTorch in One Hour](https://sebastianraschka.com/teaching/pytorch-1h/) — Sebastian Raschka (2025)
- Original source Track 2 is adapted from: [PyTorch "Learn the Basics"](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
