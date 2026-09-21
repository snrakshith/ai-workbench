"""Tests for the from-scratch PyTorch-like layers."""

from __future__ import annotations

import torch

from nnzero.layers import (
    BatchNorm1d,
    Embedding,
    FlattenConsecutive,
    Linear,
    Sequential,
    Tanh,
)


def test_linear_shape() -> None:
    layer = Linear(10, 5)
    x = torch.randn(4, 10)
    out = layer(x)
    assert out.shape == (4, 5)
    assert len(layer.parameters()) == 2


def test_linear_no_bias() -> None:
    layer = Linear(10, 5, bias=False)
    assert len(layer.parameters()) == 1


def test_batchnorm_shape() -> None:
    bn = BatchNorm1d(8)
    x = torch.randn(16, 8)
    out = bn(x)
    assert out.shape == (16, 8)
    assert len(bn.parameters()) == 2


def test_batchnorm_running_stats() -> None:
    bn = BatchNorm1d(8)
    x = torch.randn(16, 8)
    _ = bn(x)
    assert bn.running_mean.shape == (8,)
    assert bn.running_var.shape == (8,)


def test_batchnorm_eval_uses_running_stats() -> None:
    bn = BatchNorm1d(8)
    x = torch.randn(16, 8)
    bn(x)
    running_mean_before = bn.running_mean.clone()
    bn.training = False
    _ = bn(x)
    assert torch.allclose(bn.running_mean, running_mean_before)


def test_tanh_shape() -> None:
    t = Tanh()
    x = torch.randn(4, 5)
    out = t(x)
    assert out.shape == (4, 5)
    assert len(t.parameters()) == 0


def test_embedding_shape() -> None:
    emb = Embedding(27, 10)
    ix = torch.tensor([[0, 1, 2], [3, 4, 5]])
    out = emb(ix)
    assert out.shape == (2, 3, 10)
    assert len(emb.parameters()) == 1


def test_flatten_consecutive_shape() -> None:
    layer = FlattenConsecutive(2)
    x = torch.randn(4, 8, 3)
    out = layer(x)
    assert out.shape == (4, 4, 6)


def test_flatten_consecutive_squeeze() -> None:
    layer = FlattenConsecutive(8)
    x = torch.randn(4, 8, 3)
    out = layer(x)
    assert out.shape == (4, 24)


def test_sequential_shape() -> None:
    model = Sequential(
        [
            Linear(10, 20),
            Tanh(),
            Linear(20, 5),
        ]
    )
    x = torch.randn(8, 10)
    out = model(x)
    assert out.shape == (8, 5)
    assert len(model.parameters()) == 4


def test_sequential_train_eval() -> None:
    model = Sequential(
        [
            Linear(10, 20),
            BatchNorm1d(20),
            Tanh(),
        ]
    )
    model.eval()
    assert not model.layers[1].training
    model.train()
    assert model.layers[1].training
