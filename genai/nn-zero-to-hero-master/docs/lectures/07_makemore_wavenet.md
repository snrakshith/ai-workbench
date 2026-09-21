# Lecture 7: WaveNet-Style Hierarchical Character Models

This lecture turns the flat multilayer perceptron from earlier into a hierarchical, WaveNet-style character-level language model. Instead of flattening the whole context window into a single large vector, the network processes characters in stages: pairs of embeddings are merged, then pairs of pairs, and so on. This design lets the model grow its receptive field exponentially while keeping early-layer weight matrices small.

## Key concepts

- **Hierarchical receptive field.** Each `FlattenConsecutive(2)` layer groups neighboring time steps and merges them, so the first layer sees 2 characters, the second sees 4, and the third sees 8. The network learns local patterns first and composes them into longer-range structure, matching how sequences have structure at many scales.

- **`FlattenConsecutive`.** This layer reshapes a tensor so that consecutive positions are concatenated along the feature dimension. With `block_size=8`, it turns `(batch, 8, emb)` into `(batch, 4, emb*2)` so the following `Linear` operates on pairs of characters.

- **Kaiming initialization.** Deep networks can suffer from exploding or vanishing activations. Kaiming init scales weights by `1/sqrt(fan_in)`, which roughly preserves unit variance across layers and makes deep stacks trainable.

- **Batch normalization.** `BatchNorm1d` normalizes each mini-batch to zero mean and unit variance, then learns a per-channel scale (`gamma`) and shift (`beta`). This reduces internal covariate shift, allows higher learning rates, and makes evaluation rely on running statistics rather than batch statistics.

- **Cross-entropy and SGD with decay.** The model outputs logits and is trained with `F.cross_entropy` against the true next-character index. Plain SGD with a learning rate drop from `0.1` to `0.01` after most progress is made lets the model settle into a flatter minimum.

## What to watch for

- **Forgetting to set eval mode.** BatchNorm layers use running mean and variance at test time. If `layer.training` stays `True` during validation, the reported loss will be unreliable.

- **Mismatched `block_size` and number of stages.** Each `FlattenConsecutive(2)` doubles the receptive field; three stages cover exactly 8 characters. Adding or removing stages without changing `block_size` will cause shape errors or leave context unused.

- **Confusing flattening with convolutions.** `FlattenConsecutive` is applied globally across the batch and sequence, but the same linear weights are reused for every local patch. The final cells preview how this "for loop" becomes a 1-D convolution.

## Tensor shapes / notation

With `block_size = 8` and `batch_size = B`:

- Input: `(B, 8)` integer character indices
- Embedding: `(B, 8, 24)` — 24-dimensional embeddings
- After 1st `FlattenConsecutive(2)`: `(B, 4, 48)`
- After 1st `Linear`: `(B, 4, 128)`
- After 2nd `FlattenConsecutive(2)`: `(B, 2, 256)`
- After 2nd `Linear`: `(B, 2, 128)`
- After 3rd `FlattenConsecutive(2)`: `(B, 1, 256)`
- After 3rd `Linear`: `(B, 1, 128)`
- Final `Linear`: `(B, 1, 27)` logits over the vocabulary

(The leading singleton dimension is typically squeezed implicitly by `F.cross_entropy`, which accepts logits of shape `(B, C)` or `(B, C, ...)`.)

## Extension ideas

- **Vary the context length.** Try `block_size = 16` and add a fourth `FlattenConsecutive(2)` stage, or reduce to `block_size = 4`. Measure validation loss to see whether longer context actually helps for this dataset.

- **Replace flattening with `torch.nn.Conv1d`.** Implement the same hierarchical model using 1-D convolutions over the embedding dimension. Convolutions make the weight sharing explicit and are much more efficient than looping over positions.

- **Add residual connections or dilated convolutions.** Once the model is deep, residual links can stabilize training, and dilated convolutions can expand the receptive field without adding layers — both are core ingredients of the full WaveNet architecture.
