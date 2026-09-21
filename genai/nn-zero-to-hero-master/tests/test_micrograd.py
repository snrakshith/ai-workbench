"""Tests for the scalar autograd engine."""

from __future__ import annotations

import math

from nnzero.micrograd import MLP, Neuron, Value, draw_dot, trace


def test_value_addition() -> None:
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    assert c.data == 5.0


def test_value_multiplication() -> None:
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    assert c.data == 6.0


def test_gradient_finite_difference() -> None:
    """Check that .backward() agrees with a numerical derivative."""
    x = Value(2.0)
    y = 3 * x**2 - 4 * x + 5
    y.backward()

    h = 1e-6
    numerical = (3 * (2.0 + h) ** 2 - 4 * (2.0 + h) + 5 - y.data) / h
    assert abs(x.grad - numerical) < 1e-4


def test_tanh_gradient() -> None:
    x = Value(0.5)
    y = x.tanh()
    y.backward()

    h = 1e-6
    numerical = (math.tanh(0.5 + h) - math.tanh(0.5)) / h
    assert abs(x.grad - numerical) < 1e-4


def test_reused_value_gradient_accumulation() -> None:
    """A value used twice should accumulate both gradient contributions."""
    a = Value(3.0)
    b = a + a
    b.backward()
    # d(a + a)/da = 2
    assert a.grad == 2.0


def test_neuron_forward() -> None:
    n = Neuron(3)
    out = n([1.0, 2.0, 3.0])
    assert isinstance(out, Value)
    assert -1.0 < out.data < 1.0


def test_mlp_forward() -> None:
    model = MLP(3, [4, 1])
    out = model([1.0, 2.0, 3.0])
    assert isinstance(out, Value)


def test_trace_and_draw_dot() -> None:
    a = Value(2.0, label="a")
    b = Value(3.0, label="b")
    c = a * b
    nodes, edges = trace(c)
    assert a in nodes
    assert b in nodes
    assert c in nodes
    assert (a, c) in edges

    dot = draw_dot(c)
    assert dot.format == "svg"
    # Graphviz records the label as "{ a | data ... }"; tolerate whitespace variations.
    assert 'label="{a' in dot.source.replace(" ", "") or 'label="{ a' in dot.source


def test_scalar_operations() -> None:
    a = Value(4.0)
    b = a / 2 + 1
    b.backward()
    assert b.data == 3.0
    assert abs(a.grad - 0.5) < 1e-6
