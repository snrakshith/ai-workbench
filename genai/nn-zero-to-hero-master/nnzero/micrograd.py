"""A tiny scalar-valued autograd engine.

This module implements the ``Value`` class from the micrograd lecture series.
It is intentionally minimal: every operation creates a node in a computation
graph, and calling ``.backward()`` on the final node applies the chain rule in
reverse topological order.

The implementation mirrors what is built by hand in the lectures, but adds
lightweight type hints and helper utilities so it can be imported into
notebooks and tests.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

from graphviz import Digraph


class Value:
    """A scalar value that remembers how it was computed and can backpropagate gradients.

    Attributes:
        data: The scalar numerical value.
        grad: The gradient of the final output with respect to this value.
            Initialized to ``0.0`` and populated by ``backward()``.
        label: Optional human-readable name for plotting/debugging.
        _prev: The parent ``Value`` nodes that produced this node.
        _op: The operator string (e.g. ``+``, ``*``, ``tanh``) used for visualization.
        _backward: A no-argument function that propagates ``out.grad`` backward
            to ``self`` (and optionally ``other``) using the local derivative.
    """

    def __init__(
        self,
        data: float,
        _children: Iterable[Value] = (),
        _op: str = "",
        label: str = "",
    ) -> None:
        self.data = float(data)
        self.grad = 0.0
        self._backward: Callable[[], None] = lambda: None
        self._prev: set[Value] = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self) -> str:
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def _to_value(self, other: float | Value) -> Value:
        """Convert a Python scalar to a leaf Value so we can mix types."""
        return other if isinstance(other, Value) else Value(other)

    def __add__(self, other: float | Value) -> Value:
        other = self._to_value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: float | Value) -> Value:
        return self + other

    def __neg__(self) -> Value:
        return self * -1

    def __sub__(self, other: float | Value) -> Value:
        return self + (-other)

    def __rsub__(self, other: float | Value) -> Value:
        return self._to_value(other) + (-self)

    def __mul__(self, other: float | Value) -> Value:
        other = self._to_value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: float | Value) -> Value:
        return self * other

    def __pow__(self, other: float) -> Value:
        """Power operation supporting scalar (int/float) exponents only."""
        if not isinstance(other, (int, float)):
            raise TypeError("only supporting int/float powers for now")
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward() -> None:
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other: float | Value) -> Value:
        return self * other**-1

    def tanh(self) -> Value:
        """Hyperbolic tangent activation: out = tanh(self)."""
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward() -> None:
            self.grad += (1.0 - t**2) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> Value:
        """Exponential activation: out = exp(self)."""
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward() -> None:
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def relu(self) -> Value:
        """ReLU activation: out = max(0, self)."""
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward() -> None:
            self.grad += (self.data > 0.0) * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        """Run reverse-mode autodiff from this node to all ancestors.

        Why topological sort first? We need to visit every node after all of
        its children have been visited, so that ``out.grad`` is fully populated
        before we use it to update parent gradients.
        """
        topo: list[Value] = []
        visited: set[Value] = set()

        def build_topo(v: Value) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


def trace(root: Value) -> tuple[set[Value], set[tuple[Value, Value]]]:
    """Build the set of all nodes and edges reachable from ``root``."""
    nodes: set[Value] = set()
    edges: set[tuple[Value, Value]] = set()

    def build(v: Value) -> None:
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root: Value, format: str = "svg", rankdir: str = "LR") -> Digraph:
    """Render a ``Value`` computation graph using Graphviz.

    Args:
        root: The output node of the graph.
        format: Graphviz output format (e.g. ``svg``, ``png``).
        rankdir: Graph direction, ``LR`` means left-to-right.

    Returns:
        A ``graphviz.Digraph`` object. In a Jupyter notebook this will render
        automatically; you can also call ``.render()`` to save to a file.
    """
    dot = Digraph(format=format, graph_attr={"rankdir": rankdir})
    nodes, edges = trace(root)

    for n in nodes:
        uid = str(id(n))
        dot.node(
            name=uid,
            label=f"{{ {n.label} | data {n.data:.4f} | grad {n.grad:.4f} }}",
            shape="record",
        )
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot


class Neuron:
    """A single neuron: tanh(sum(w_i * x_i) + b).

    This is the same object built in the second half of the micrograd lecture.
    Weights and bias are initialized uniformly in ``[-1, 1]``.
    """

    def __init__(self, nin: int) -> None:
        import random

        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x: Iterable[float | Value]) -> Value:
        act = sum((wi * xi for wi, xi in zip(self.w, x, strict=True)), self.b)
        return act.tanh()

    def parameters(self) -> list[Value]:
        return self.w + [self.b]


class Layer:
    """A layer of ``nout`` independent neurons, each receiving ``nin`` inputs."""

    def __init__(self, nin: int, nout: int) -> None:
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x: Iterable[float | Value]) -> Value | list[Value]:
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self) -> list[Value]:
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    """A multi-layer perceptron built from ``Layer`` objects.

    Args:
        nin: Number of input features.
        nouts: List of layer widths. The last entry is the output dimension.
    """

    def __init__(self, nin: int, nouts: list[int]) -> None:
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x: Iterable[float | Value]) -> Value | list[Value]:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]
