"""
Sections 3-4: Computation graphs and automatic differentiation.

Run with:
    uv run scripts/03_computation_graphs_and_autograd.py
"""
import torch
import torch.nn.functional as F
from torch.autograd import grad


def main():
    # Section 3: a logistic regression forward pass as a computation graph
    y = torch.tensor([1.0])
    x1 = torch.tensor([1.1])
    w1 = torch.tensor([2.2])
    b = torch.tensor([0.0])

    z = x1 * w1 + b
    a = torch.sigmoid(z)
    loss = F.binary_cross_entropy(a, y)
    print("loss (no grad tracking):", loss)

    # Section 4: automatic differentiation
    w1 = torch.tensor([2.2], requires_grad=True)
    b = torch.tensor([0.0], requires_grad=True)

    z = x1 * w1 + b
    a = torch.sigmoid(z)
    loss = F.binary_cross_entropy(a, y)

    grad_L_w1 = grad(loss, w1, retain_graph=True)
    grad_L_b = grad(loss, b, retain_graph=True)
    print("manual grad w.r.t. w1:", grad_L_w1)
    print("manual grad w.r.t. b:", grad_L_b)

    loss.backward()
    print("loss.backward() -> w1.grad:", w1.grad)
    print("loss.backward() -> b.grad:", b.grad)


if __name__ == "__main__":
    main()
