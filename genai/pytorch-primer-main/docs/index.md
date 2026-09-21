# PyTorch Primer

A hands-on **"PyTorch in ~1-2 hours"** teaching kit for students who are new to PyTorch. It's
built to be read, run, and taught from — every notebook and script here is small enough to
understand line by line, and works unmodified whether your machine has an NVIDIA GPU, Apple
Silicon, or neither.

This site is the written, browsable version of the notebooks in
[`notebooks/`](https://github.com/sourangshupal/pytorch-primer/tree/main/notebooks) and the
scripts in [`scripts/`](https://github.com/sourangshupal/pytorch-primer/tree/main/scripts). Read
it here, or open the actual `.ipynb` files and run the code yourself — both stay in sync.

## Two complementary tracks

<div class="grid cards" markdown>

- :material-book-open-variant: **Track 1 — PyTorch in One Hour**

    ---

    Adapted from Sebastian Raschka's [*PyTorch in One Hour*](https://sebastianraschka.com/teaching/pytorch-1h/)
    (2025). Builds every concept from small, hand-written toy tensors — no dataset downloads, no
    distractions. Covers tensors, autograd, `nn.Module`, data loading, training loops,
    saving/loading models, and single-/multi-GPU training.

    [:octicons-arrow-right-24: Start Track 1](track1-raschka/01-what-is-pytorch.md)

- :material-file-document-multiple: **Track 2 — Official PyTorch Basics**

    ---

    Adapted from the official [PyTorch "Learn the Basics"](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
    series. Fills in what Track 1 doesn't cover — an end-to-end Quickstart on a real dataset,
    Transforms, the NumPy tensor bridge, `torch.accelerator`, and more — using the real
    **FashionMNIST** dataset.

    [:octicons-arrow-right-24: Start Track 2](track2-official/10-quickstart.md)

</div>

## Why two tracks?

Track 1 teaches every mechanic from tiny, hand-written tensors — nothing to download, nothing
to hide behind. Track 2 takes those same mechanics and applies them to a real dataset end to
end, so you see what "real" PyTorch code looks like once toy tensors are replaced with actual
images. Each Track 2 notebook explicitly says which Track 1 notebook it complements — work
through Track 1 first, then Track 2 to see the same ideas at real scale.

## Quick start

```bash
uv sync                              # installs the latest PyTorch + deps for your platform
uv run main.py                       # sanity check: prints PyTorch version + detected hardware
uv run jupyter lab notebooks/        # open the notebooks
```

See [Getting Started](getting-started.md) for the full walkthrough, including what to do if you
don't have `uv` installed yet, and how hardware detection works.

## What's in the repository

| Folder | Contents |
|---|---|
| `notebooks/` | 17 notebooks: 01-09 (Raschka's primer) + 10-17 (official PyTorch basics gap-fill) |
| `scripts/` | Standalone `.py` equivalents of notebooks 01-08, plus `scripts/ddp_train.py` (run via `torchrun`, not directly) |
| `docs/` | This documentation site's source, plus the original written handout |
| `utils/` | `device.py` — the shared hardware-detection helper used everywhere |

## Teaching this in a session

If you're planning to run this as a live class, see the [Teaching Guide](teaching-guide.md) for
a suggested minute-by-minute timing table for both a 1-hour lightning session and a 2-hour
session with exercises.
