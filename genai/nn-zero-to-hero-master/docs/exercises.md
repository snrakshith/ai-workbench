# Exercise Index

A master list of exercises across the notebooks, grouped by topic and rated by
difficulty.

## Legend

- ⭐ Easy: mostly a one-line change or a short verification.
- ⭐⭐ Medium: requires understanding and modifying a few cells.
- ⭐⭐⭐ Hard: requires combining multiple concepts or writing non-trivial code.

---

## 01 — Micrograd Autograd

| # | Exercise | Difficulty | Key concept |
|---|----------|------------|-------------|
| 1 | Add an `exp()` operation to the manual `Value` class and verify its gradient numerically. | ⭐⭐ | Chain rule, exponential derivative |
| 2 | Verify the gradient of the hand-built neuron using finite differences. | ⭐⭐ | Numerical gradient check |

**Extensions**

- Add `sin`, `cos`, or `log` operations to `Value`.
- Visualize the computation graph for a small MLP using `draw_dot`.

---

## 02 — Micrograd MLP

| # | Exercise | Difficulty | Key concept |
|---|----------|------------|-------------|
| 1 | Add a `relu()` method to `Value` and train a small MLP with it. | ⭐⭐ | Custom activation, gradient clipping at 0 |
| 2 | Experiment with architecture, learning rate, and forgetting to zero gradients. | ⭐⭐ | Hyperparameters, gradient accumulation |

**Extensions**

- Implement L2 regularization by adding a penalty to the loss.
- Train on a tiny real dataset (e.g., first letters of names).

---

## 03 — Makemore Bigrams

| # | Exercise | Difficulty | Key concept |
|---|----------|------------|-------------|
| 1 | Tune the smoothing constant (`alpha`) and observe its effect on loss and samples. | ⭐ | Add-one smoothing, regularization |
| 2 | Vectorize the bigram loss using `F.cross_entropy` instead of the manual softmax loop. | ⭐⭐ | Softmax + cross-entropy equivalence |

**Extensions**

- Implement a trigram model (condition on the previous two characters).
- Compare sampling quality for different smoothing values.

---

## 04 — Makemore MLP

| # | Exercise | Difficulty | Key concept |
|---|----------|------------|-------------|
| 1 | Vary `block_size` and observe train/validation loss. | ⭐ | Context length vs. capacity |
| 2 | Visualize the embedding matrix in 2-D after training and interpret clusters. | ⭐⭐ | Embeddings, geometry |

**Extensions**

- Add a second hidden layer.
- Implement a simple learning-rate finder.

---

## 05 — Makemore BatchNorm

| # | Exercise | Difficulty | Key concept |
|---|----------|------------|-------------|
| 1 | Remove Kaiming init and observe activation/gradient statistics. | ⭐⭐ | Initialization importance |
| 2 | Remove BatchNorm from the deep network and observe training instability. | ⭐⭐ | BatchNorm stabilizing effect |

**Extensions**

- Plot activation histograms at multiple training stages.
- Implement LayerNorm instead of BatchNorm.

---

## 06 — Makemore Backprop Ninja

| # | Exercise | Difficulty | Key concept |
|---|----------|------------|-------------|
| 1 | Derive and implement the manual backward pass for cross-entropy in one line. | ⭐⭐⭐ | Softmax Jacobian simplification |
| 2 | Derive and implement the manual backward pass through BatchNorm. | ⭐⭐⭐ | BatchNorm backward formula |

**Extensions**

- Write the full training loop using only manual gradients (no `.backward()`).
- Compare the speed of manual backprop vs. PyTorch autograd.

---

## 07 — Makemore WaveNet

| # | Exercise | Difficulty | Key concept |
|---|----------|------------|-------------|
| 1 | Trace intermediate tensor shapes through the hierarchical model. | ⭐⭐ | Receptive field, `FlattenConsecutive` |
| 2 | Tune `block_size`, `n_embd`, `n_hidden`, or add an extra stage and measure validation loss. | ⭐⭐⭐ | Architecture search, overfitting |

**Extensions**

- Replace `FlattenConsecutive` with a 1-D convolution and compare results.
- Train on a different dataset (e.g., surnames, city names).

---

## Cross-cutting challenges

1. **Reproducibility audit:** Pick one notebook, restart the kernel, and confirm
   that training loss matches a previously saved value. Why might it differ on
   GPU vs. CPU?
2. **Visualization:** Pick a model and create a plot of train/val loss curves,
   activation histograms, and gradient histograms. What do healthy curves look
   like?
3. **Ablation study:** Remove one component (BatchNorm, Kaiming init, bias term,
   embedding table) and measure the impact on loss.
