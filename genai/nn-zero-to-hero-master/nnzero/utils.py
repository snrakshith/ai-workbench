"""Shared helpers for building datasets, sampling, and reproducibility."""

from __future__ import annotations

import random
from pathlib import Path

import torch
from torch import Tensor


def set_seed(seed: int) -> torch.Generator:
    """Seed Python ``random`` and return a PyTorch CPU generator.

    Using a dedicated ``torch.Generator`` (rather than the global RNG) makes
    experiments easier to reproduce and avoids surprising side effects when
    multiple notebooks run in the same process.
    """
    random.seed(seed)
    return torch.Generator().manual_seed(seed)


def build_vocab(words: list[str]) -> tuple[dict[str, int], dict[int, str], int]:
    """Build character-to-index and index-to-character mappings.

    The special ``.`` token represents both the start-of-word and end-of-word
    marker. It is assigned index 0; other characters are sorted alphabetically.

    Returns:
        A tuple ``(stoi, itos, vocab_size)``.
    """
    chars = sorted(set("".join(words)))
    stoi: dict[str, int] = {s: i + 1 for i, s in enumerate(chars)}
    stoi["."] = 0
    itos: dict[int, str] = {i: s for s, i in stoi.items()}
    return stoi, itos, len(itos)


def build_dataset(
    words: list[str],
    stoi: dict[str, int],
    block_size: int = 3,
    generator: torch.Generator | None = None,
    shuffle: bool = False,
) -> tuple[Tensor, Tensor]:
    """Create input/target tensors for a character-level language model.

    For each word, a sliding window of length ``block_size`` predicts the next
    character. The initial window is filled with the start token (index 0).

    Args:
        words: List of words (strings).
        stoi: Character-to-index mapping.
        block_size: Context length.
        generator: Optional generator for shuffling.
        shuffle: Whether to shuffle the order of words before processing.

    Returns:
        ``(X, Y)`` tensors where ``X[i]`` is the context and ``Y[i]`` is the
        target character index.
    """
    if shuffle:
        words = words.copy()
        if generator is not None:
            # Use torch-derived randomness when a generator is supplied.
            rng = random.Random(generator.initial_seed())
            rng.shuffle(words)
        else:
            random.shuffle(words)

    x_list, y_list = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + ".":
            ix = stoi[ch]
            x_list.append(context)
            y_list.append(ix)
            context = context[1:] + [ix]

    x = torch.tensor(x_list)
    if block_size == 1:
        # Bigram models expect a 1-D input of previous-character indices.
        x = x.squeeze(1)
    return x, torch.tensor(y_list)


def split_dataset(
    words: list[str],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    shuffle: bool = True,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """Split a list of words into train/validation/test sets.

    The default ``80/10/10`` split follows the makemore lectures.
    """
    if shuffle:
        words = words.copy()
        random.seed(seed)
        random.shuffle(words)

    n = len(words)
    n1 = int(train_frac * n)
    n2 = int((train_frac + val_frac) * n)
    return words[:n1], words[n1:n2], words[n2:]


def load_names(path: str | Path = "data/names.txt") -> list[str]:
    """Load newline-separated words from a text file."""
    return Path(path).read_text().splitlines()


def sample_from_model(
    model,
    stoi: dict[str, int],
    itos: dict[int, str],
    block_size: int,
    generator: torch.Generator,
    n_samples: int = 20,
) -> list[str]:
    """Generate ``n_samples`` words from a makemore-style model.

    Args:
        model: A callable that accepts a ``(1, block_size)`` integer tensor
            and returns logits of shape ``(1, vocab_size)``.
        stoi: Character-to-index mapping.
        itos: Index-to-character mapping.
        block_size: Context length expected by the model.
        generator: PyTorch generator for sampling.
        n_samples: Number of words to generate.

    Returns:
        A list of generated strings.
    """
    results: list[str] = []
    for _ in range(n_samples):
        out: list[int] = []
        context = [0] * block_size
        while True:
            logits = model(torch.tensor([context]))
            probs = torch.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=generator).item()
            context = context[1:] + [ix]
            out.append(ix)
            if ix == 0:
                break
        results.append("".join(itos[i] for i in out))
    return results
