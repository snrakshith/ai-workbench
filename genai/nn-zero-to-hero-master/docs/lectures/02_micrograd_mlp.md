# Lecture 02: Micrograd — Building an MLP from Scratch

This lecture takes students from scalar derivatives to a working multi-layer perceptron, all built on a tiny autograd engine. By implementing `Value`, `Neuron`, `Layer`, and `MLP` by hand and training the network on a toy binary-classification task, students see exactly how backpropagation chains simple local derivatives through a computation graph.

## Key concepts

- **Numerical derivatives as a sanity check.** Approximating a derivative with a small perturbation `(f(x + h) - f(x)) / h` is slow but exact enough to verify an analytical gradient implementation. Students should use it whenever they add a new operation to `Value`.

- **Reverse-mode autograd on scalars.** The `Value` object stores its parents, the operation that produced it, and a local `_backward` closure. Calling `backward()` performs a topological sort, seeds the output gradient at `1.0`, and walks the graph in reverse so the chain rule accumulates gradients at every leaf.

- **A neuron is a structured graph of `Value`s.** A single neuron computes `tanh(w·x + b)`, which is just addition, multiplication, and activation composed together. There is no magic: every weight, bias, and intermediate result is an explicit node with a `.grad`.

- **Why gradients must be zeroed before `backward()`.** Each `_backward` adds to `grad` rather than assigning, so repeated passes without zeroing accumulate stale gradients. This is the same behavior PyTorch requires with `optimizer.zero_grad()`.

- **Gradient descent on parameters.** After backprop, every parameter is nudged by `-learning_rate * p.grad`. Repeating forward → loss → zero → backward → update drives the MLP’s predictions toward the targets.

## What to watch for

- **Forgetting `self.grad +=` in `_backward`.** New implementations often write `self.grad = ...`, which overwrites gradients when a node is used more than once. Addition and multiplication both accumulate because a leaf can feed into multiple outputs.

- **Misunderstanding `o.grad = 1.0`.** Students sometimes think this number is learned or guessed. It is simply the derivative of the output with respect to itself, which starts the chain rule.

- **Confusing scalar `Value` with vectorized tensors.** Everything here is a scalar object. The MLP works because Python loops and lists of `Value`s replicate what frameworks later do with matrix operations. This is intentional pedagogical scaffolding.

- **Initializing with too-large weights or a too-large learning rate.** The toy network uses `random.uniform(-1, 1)` and a learning rate of `0.1`. If training diverges or oscillates, students should suspect these hyperparameters first.

## Tensor shapes / notation

This notebook deliberately stays scalar, but it is helpful to keep the implied shapes in mind:

- Input `x`: list of `nin` scalars, e.g. `[2.0, 3.0, -1.0]` → shape `(3,)`.
- `Neuron(nin)`: holds `nin` weights plus one bias → `nin + 1` parameters.
- `Layer(nin, nout)`: `nout` neurons, each with `nin + 1` parameters → `nout * (nin + 1)` parameters total.
- `MLP(nin, nouts)`: `nouts` is a list such as `[4, 4, 1]`. The full architecture is `3 → 4 → 4 → 1`. The final `Layer` returns a single `Value` when `nout == 1`, otherwise a list.
- Loss: `sum((yout - ygt)**2)` over the four examples; the result is a single scalar `Value` whose `.backward()` fills every parameter `.grad`.

## Extension ideas

- **Add ReLU to `Value`.** Implement `relu()`, check it against a numerical derivative, and train the MLP with ReLU activations instead of `tanh`. Notice how the loss landscape and convergence change.

- **Swap MSE for a binary classification loss.** Replace the mean-squared-error loss with a simple hinge loss or a hand-rolled sigmoid + log loss, and compare final predictions and gradients against PyTorch.

- **Implement mini-batch gradient descent.** Instead of iterating one example at a time, average the loss over several examples before each parameter update. Observe whether training becomes more stable and how the gradient magnitudes scale with batch size.
