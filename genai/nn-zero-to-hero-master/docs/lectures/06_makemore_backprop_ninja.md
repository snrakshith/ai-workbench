# Lecture 6: Makemore — Becoming a Backprop Ninja

This notebook takes apart PyTorch's automatic differentiation. Starting from a fully-expanded forward pass through a character-level MLP with BatchNorm and cross-entropy loss, you derive every gradient by hand and verify each one against `torch.autograd`. Once the explicit version works, you collapse it into compact forms—softmax-cross-entropy in one line, BatchNorm in one line—and use those to train the model inside `torch.no_grad()`.

## Key concepts

- **Forward pass as simple operations.** Backprop requires the forward pass to be split into tiny, differentiable steps: embedding lookup, matrix multiply, mean, variance, reciprocal square-root, tanh, and softmax. Each step has an easy local gradient; together they compose into a full language model.

- **Backpropagation is the chain rule in reverse.** For every tensor `x` you compute `dL/dx` by multiplying the upstream gradient against the local Jacobian. Process variables in reverse topological order—children before parents—because a parent’s gradient depends on all its children’s gradients.

- **Numerical stability in softmax.** Subtract the per-row maximum from logits before exponentiating. This does not change the probabilities, but it prevents overflow in `exp()`. `F.cross_entropy` applies the same trick, which is why the library call is preferred once you understand the derivation.

- **Compact cross-entropy gradient.** Cross-entropy followed by softmax simplifies to `softmax(logits) - one_hot(target)`, scaled by `1/n`. This replaces nine intermediate variables (`logprobs`, `probs`, `counts`, etc.) and is the form you actually use in production code.

- **Compact BatchNorm gradient.** BatchNorm centers and whitens a batch, then rescales with learnable `gamma` and `beta`. Its input gradient can be written in one line using only `dhpreact`, `bnraw`, and `bnvar_inv`, avoiding every intermediate mean and variance step.

## What to watch for

- **Forgetting `keepdim=True`.** `bnmeani`, `bnvar_inv`, and `logit_maxes` keep an extra dimension so broadcasting works. Preserve those shapes when writing gradients, or reductions will silently produce wrong dimensions.

- **Dropping the `1/n` loss scale.** The final loss is the mean over the batch, so every gradient carries a factor of `1/n`. Omitting it makes updates 32× too large and training diverges.

- **Confusing `hprebn` with `hpreact`.** `hprebn` is the linear output before BatchNorm; `hpreact` is after scaling and shifting. The tanh derivative applies to `hpreact`, and the BatchNorm backward pass bridges the two.

- **Using `1 / counts_sum` instead of `counts_sum**-1`.** The notebook uses the power form so manual backprop can be bit-exact with PyTorch. Small choices like this matter when checking gradients with `torch.allclose`.

- **Forgetting to accumulate embedding gradients.** `C[Xb]` indexes the embedding matrix, so the same character can appear multiple times in a batch. Use `dC[ix] += demb[k, j]`, otherwise earlier occurrences are overwritten.

## Tensor shapes / notation

The batch size is `n = 32` and `block_size = 3`.

| Tensor | Shape | Meaning |
|--------|-------|---------|
| `Xb` | `(n, 3)` | Input character indices |
| `C` | `(vocab_size, n_embd)` | Embedding matrix |
| `emb` | `(n, 3, n_embd)` | Character embeddings |
| `embcat` | `(n, 3*n_embd)` | Flattened embeddings |
| `hprebn` | `(n, n_hidden)` | Hidden pre-activations before BatchNorm |
| `bnmeani`, `bnvar_inv` | `(1, n_hidden)` | Batch statistics |
| `bnraw`, `hpreact`, `h` | `(n, n_hidden)` | Normalized, scaled, and activated hidden state |
| `W2`, `b2` | `(n_hidden, vocab_size)`, `(vocab_size,)` | Output layer |
| `logits` | `(n, vocab_size)` | Model scores |

Matrix multiplies introduce the usual transpose rules: if `logits = h @ W2`, then `dW2 = h.T @ dlogits` and `dh = dlogits @ W2.T`.

## Extension ideas

1. **Derive the backward pass for a different non-linearity.** Replace `tanh` with `ReLU` or `GELU` and re-derive the manual gradients. Compare training speed and final loss.

2. **Add a second hidden layer.** Extend the forward pass and derive the extra set of gradients. Pay attention to how Kaiming initialization scales change with depth.

3. **Implement inference-time BatchNorm from scratch.** The notebook calibrates running mean and variance over the training set. Write your own `BatchNorm` class that tracks exponential moving averages during training and uses them at test time.
