"""Tests for nnzero utility helpers."""

from __future__ import annotations

import torch

from nnzero.utils import build_dataset, build_vocab, load_names, set_seed, split_dataset


def test_build_vocab() -> None:
    words = ["emma", "olivia"]
    stoi, itos, vocab_size = build_vocab(words)
    assert vocab_size == len(stoi)
    assert stoi["."] == 0
    assert itos[stoi["e"]] == "e"


def test_build_dataset() -> None:
    words = ["emma", "olivia"]
    stoi, itos, _ = build_vocab(words)
    x, y = build_dataset(words, stoi, block_size=3)
    assert x.ndim == 2
    assert y.ndim == 1
    assert x.shape[0] == y.shape[0]
    assert x.shape[1] == 3


def test_split_dataset() -> None:
    words = list("abcdefghij")
    train, val, test = split_dataset(words, train_frac=0.7, val_frac=0.2, shuffle=False)
    assert len(train) == 7
    assert len(val) == 2
    assert len(test) == 1
    assert set(train + val + test) == set(words)


def test_set_seed_returns_generator() -> None:
    g = set_seed(42)
    assert isinstance(g, torch.Generator)


def test_load_names(tmp_path) -> None:
    path = tmp_path / "names.txt"
    path.write_text("emma\nolivia\nava\n")
    names = load_names(path)
    assert names == ["emma", "olivia", "ava"]


def test_load_names_default_path_requires_download() -> None:
    """The default path only exists after running scripts/download_data.py."""
    # This documents the expected behavior; we don't actually assert existence
    # here to avoid requiring the dataset during unit tests.
    pass
