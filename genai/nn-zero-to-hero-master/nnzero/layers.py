"""Tiny PyTorch-like layer implementations used in the makemore lectures.

These classes intentionally mirror the ``torch.nn.Module`` API
(``__call__`` forward pass + ``parameters()``) while being written from
scratch so students can read and modify them. They are sufficient to build
bigram models, MLPs, and WaveNet-style hierarchical character-level language
models.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor


class Module:
    """Base class matching a tiny subset of ``torch.nn.Module``.

    Subclasses implement ``__call__`` for the forward pass and optionally
    override ``parameters()``.
    """

    def __call__(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def parameters(self) -> list[Tensor]:
        return []

    def zero_grad(self) -> None:
        """Set gradients of all parameters to None (PyTorch idiom for zero)."""
        for p in self.parameters():
            p.grad = None


class Linear(Module):
    """A fully-connected linear layer: ``out = x @ W + b``.

    Weights are initialized with Kaiming normalization ``1 / sqrt(fan_in)``,
    which keeps the scale of activations roughly constant through deep stacks.

    Args:
        fan_in: Number of input features.
        fan_out: Number of output features.
        bias: Whether to include a bias term.
        generator: Optional ``torch.Generator`` for reproducible init.
    """

    def __init__(
        self,
        fan_in: int,
        fan_out: int,
        bias: bool = True,
        generator: torch.Generator | None = None,
    ) -> None:
        self.weight = torch.randn((fan_in, fan_out), generator=generator) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x: Tensor) -> Tensor:
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self) -> list[Tensor]:
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d(Module):
    """1-D batch normalization over the channel/feature dimension.

    During training, normalizes using the current mini-batch statistics and
    updates running estimates. During evaluation, uses the running statistics.

    Args:
        dim: Number of features/channels.
        eps: Small constant for numerical stability.
        momentum: Exponential moving-average coefficient for running stats.
    """

    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        self.eps = eps
        self.momentum = momentum
        self.training = True

        # Trained with backprop.
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

        # Trained with a running momentum update (buffers, not gradients).
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x: Tensor) -> Tensor:
        if self.training:
            # For 2D input (B, C) average over batch dim 0.
            # For 3D input (B, T, C) average over batch and time dims (0, 1).
            dim = 0 if x.ndim == 2 else (0, 1)
            xmean = x.mean(dim, keepdim=True)
            xvar = x.var(dim, keepdim=True, unbiased=False)
        else:
            # running stats are stored as 1-D vectors; broadcast to input shape.
            shape = [1] * (x.ndim - 1) + [-1]
            xmean = self.running_mean.view(shape)
            xvar = self.running_var.view(shape)

        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta

        if self.training:
            with torch.no_grad():
                # Store 1-D running stats so parameters stay simple vectors.
                self.running_mean = (
                    1 - self.momentum
                ) * self.running_mean + self.momentum * xmean.squeeze()
                self.running_var = (
                    1 - self.momentum
                ) * self.running_var + self.momentum * xvar.squeeze()

        return self.out

    def parameters(self) -> list[Tensor]:
        return [self.gamma, self.beta]


class Tanh(Module):
    """Hyperbolic tangent element-wise non-linearity."""

    def __call__(self, x: Tensor) -> Tensor:
        self.out = torch.tanh(x)
        return self.out


class Embedding(Module):
    """A simple embedding table lookup: ``out = weight[IX]``.

    Args:
        num_embeddings: Vocabulary size.
        embedding_dim: Dimension of each embedding vector.
        generator: Optional ``torch.Generator`` for reproducible init.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        generator: torch.Generator | None = None,
    ) -> None:
        self.weight = torch.randn((num_embeddings, embedding_dim), generator=generator)

    def __call__(self, ix: Tensor) -> Tensor:
        self.out = self.weight[ix]
        return self.out

    def parameters(self) -> list[Tensor]:
        return [self.weight]


class FlattenConsecutive(Module):
    """Reshape consecutive time steps into feature vectors.

    Given input of shape ``(B, T, C)`` and ``n`` consecutive steps to merge,
    produces ``(B, T//n, C*n)``. If ``T == n`` the time dimension is squeezed
    out, yielding ``(B, C*n)``.

    This is the key layer that turns a flat MLP into a hierarchical WaveNet
    style model in makemore part 5.
    """

    def __init__(self, n: int) -> None:
        self.n = n

    def __call__(self, x: Tensor) -> Tensor:
        b, t, c = x.shape
        x = x.view(b, t // self.n, c * self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        self.out = x
        return self.out


class Sequential(Module):
    """Stack modules so that ``model(x)`` calls each layer in order."""

    def __init__(self, layers: Iterable[Module]) -> None:
        self.layers = list(layers)

    def __call__(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out

    def parameters(self) -> list[Tensor]:
        return [p for layer in self.layers for p in layer.parameters()]

    def train(self, mode: bool = True) -> None:
        """Set all layers with a ``training`` attribute to train/eval mode."""
        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.training = mode

    def eval(self) -> None:
        self.train(False)
