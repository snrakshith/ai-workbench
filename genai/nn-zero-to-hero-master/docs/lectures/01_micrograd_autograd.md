# Lecture 01: Building a Micrograd Autograd Engine

This lecture builds a tiny automatic-differentiation engine from scratch. Students start by approximating derivatives numerically, then implement a scalar `Value` class that records operations, builds a computation graph, and propagates gradients backward via the chain rule. The notebook connects the engine to a hand-built neuron (`tanh` activation) and shows how gradient descent nudges inputs to change the output.

## Key concepts

- **Numerical derivatives as a sanity check.** Before building autograd, we estimate `df/dx` with `(f(x + h) - f(x)) / h` for a tiny `h`. This gives an independent reference we can compare against the analytic gradients produced later.

- **`Value` objects as tiny tensors.** Each `Value` wraps a scalar, stores its gradient, remembers its parents (`_prev`), the operation that created it (`_op`), and a local backward closure (`_backward`). This is exactly the same information PyTorch keeps on tensors, just simplified to one number.

- **Reverse-mode autodiff with topological sort.** After the forward pass, `backward()` sorts the graph so every node is processed after all of its children. Reversing that order lets us apply the chain rule from the output back to the inputs without ever recomputing intermediate derivatives.

- **Local derivatives and the chain rule.** Each operation only knows its own local derivative (`+` contributes `1`, `*` contributes the other operand, `tanh` contributes `1 - tanh(x)^2`). During backprop, that local derivative is multiplied by the incoming gradient (`out.grad`) and passed to each parent.

- **Gradient accumulation for reused values.** A value used in multiple branches (e.g., `a + a` or `a` appearing in two terms) receives contributions from every branch. Using `+=` in `_backward` makes the engine automatically sum those partial derivatives, which is the multivariate chain rule in action.

## What to watch for

- **Forgetting to zero gradients.** `Value.grad` is initialized once and accumulated. If a student runs `backward()` twice on the same graph without resetting, gradients will be doubled.

- **Confusing `data` and `grad`.** `data` is the forward value; `grad` is how much the final output changes when this value changes. Remind students that `x.grad` is not a slope of `x` alone, but a sensitivity relative to the chosen output.

- **Wrong order in backprop.** Without topological sorting, a node might be processed before its children, giving stale `out.grad` values. The recursive `build_topo` in the notebook prevents this by visiting parents before appending the current node.

- **Perturbing the wrong variable in finite differences.** When verifying `do/dw1`, only `w1.data` should change between the two forward passes. Changing anything else makes the numeric estimate invalid.

## Tensor shapes / notation

This notebook is intentionally scalar: every `Value` holds a single real number, not a vector or matrix. Expressions like `x1*w1 + x2*w2 + b` are written out node by node so students can see each `+`, `*`, and `tanh` as a distinct graph vertex. Shapes are therefore trivial (`()` for every value). The later lectures lift this same logic to tensors.

## Extension ideas

- **Add more elementary operations.** Extend `Value` with `__pow__`, `__sub__`, `__truediv__`, or `log()`. For each, derive the local derivative and verify it with finite differences.

- **Build a tiny multi-layer perceptron.** Use only `Value` and the operations already implemented to create a small network with two inputs, one hidden layer, and one output. Train it on a single example by nudging weights opposite to their gradients.

- **Compare with PyTorch.** Re-implement the single-neuron forward pass using `torch.Tensor` with `requires_grad=True`. Call `.backward()` and check that PyTorch's gradients match the hand-rolled engine to high precision.
