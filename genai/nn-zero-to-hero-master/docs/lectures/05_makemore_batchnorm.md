# Makemore Part 3: Batch Normalization

This lecture trains a deeper character-level MLP on the `names.txt` dataset and introduces **Batch Normalization** to keep optimization stable. You will refactor the previous notebook's raw tensor code into reusable `Linear`, `BatchNorm1d`, and `Tanh` layers, then use activation and gradient histograms to diagnose whether training is healthy.

## Key concepts

- **Kaiming (He) initialization.** Weights are sampled from $\mathcal{N}(0, 1/\text{fan\_in})$ and multiplied by a gain (`5/3` for `tanh`). This keeps the variance of pre-activations close to one, so `tanh` stays in its linear region and gradients do not vanish.

- **Batch Normalization.** Before the non-linearity, subtract the batch mean and divide by the batch standard deviation, then apply a learned scale (`gamma`) and shift (`beta`). BatchNorm fixes the distribution feeding each layer, which reduces sensitivity to initialization and allows larger learning rates.

- **Running statistics for inference.** During training, BatchNorm uses statistics from the current mini-batch. During evaluation, it uses exponential moving averages accumulated during training, so predictions become deterministic and no longer depend on batch composition.

- **Modular layers mimicking PyTorch.** `Linear`, `BatchNorm1d`, and `Tanh` classes expose `__call__` for the forward pass and `parameters()` for optimization. This pattern is exactly what `nnzero.layers` provides and is the same API as PyTorch `nn.Module`.

- **Activation and gradient diagnostics.** Histograms of `tanh` outputs, per-layer gradients, and update-to-parameter ratios (`std(update) / std(parameter)`) reveal whether the network is saturated, whether gradients are exploding or vanishing, and whether all layers learn at a similar speed.

## What to watch for

- **Forgetting to set `training = False` at inference time.** BatchNorm must use running mean and variance when sampling or evaluating; otherwise the output depends on the current batch and predictions become unstable.

- **Confusing batch statistics with running statistics.** Batch stats are computed from the current mini-batch; running stats are updated with a momentum term (`0.001` in the manual code, `0.1` in the layer class) and used only at evaluation time.

- **Removing BatchNorm without fixing initialization.** Disabling BatchNorm while keeping large weight scales pushes `tanh` into saturation, causing gradients to vanish and training to plateau.

- **Treating BatchNorm as a substitute for initialization.** BatchNorm makes deep networks much easier to train, but you still need reasonable weight initialization; the two techniques work together.

- **Coupling examples inside a batch.** Because BatchNorm normalizes across the batch, changing one training example slightly affects the normalized value of every other example in that batch. This is fine during training but is why inference must switch to running stats.

## Tensor shapes / notation

| Tensor | Shape | Meaning |
|--------|-------|---------|
| `Xb` | `(batch_size, block_size)` | Integer indices for the input context |
| `emb = C[Xb]` | `(batch_size, block_size, n_embd)` | Character embedding lookup |
| `embcat` | `(batch_size, block_size * n_embd)` | Flattened embeddings fed into the MLP |
| `hpreact = embcat @ W1` | `(batch_size, n_hidden)` | Hidden-layer pre-activation |
| `bnmeani`, `bnstdi` | `(1, n_hidden)` | Batch mean and standard deviation over the batch dimension |
| `h = tanh(hpreact)` | `(batch_size, n_hidden)` | Hidden activation |
| `logits = h @ W2 + b2` | `(batch_size, vocab_size)` | Output logits for next-character prediction |

## Extension ideas

1. **Compare normalization schemes.** Implement LayerNorm or GroupNorm and train the same deep architecture. Which one gives the best validation loss, and which is easiest to use at inference time?

2. **Depth without BatchNorm.** Remove all `BatchNorm1d` layers and see how deep you can make the network using only Kaiming initialization. Plot the saturation percentage per layer to find the limit.

3. **Visualize the embedding space.** After training, plot the 2-D character embeddings with PCA or t-SNE. Do vowels cluster together? Do rare characters end up near the origin?
