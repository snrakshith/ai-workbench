# Makemore Part 2: Multi-Layer Perceptron (MLP)

This lecture upgrades the bigram character-level language model into a multi-layer perceptron. Using a fixed window of previous characters as context, we look up learned embeddings, pass them through a hidden `tanh` layer, and predict the next character with a softmax output. The notebook walks through the full forward pass in raw PyTorch before relying on higher-level APIs, then trains the model with mini-batch stochastic gradient descent and samples new names from the learned distribution.

## Key concepts

- **Embedding lookup table.** Each of the 27 characters is represented by a learned vector stored in a matrix `C`. Indexing `C[X]` replaces every character index in a batch with its embedding vector, turning discrete symbols into continuous vectors that the network can learn to arrange by similarity.

- **Context window and flattening.** A `block_size` of 3 means each training example feeds the previous three characters into the model. After embedding, the shape `(batch, 3, emb_dim)` is flattened to `(batch, 3 * emb_dim)` so it can be multiplied by the first weight matrix `W1`.

- **Hidden non-linearity.** The first layer computes `h = tanh(emb.view(-1, 30) @ W1 + b1)`. The `tanh` introduces a non-linear transformation; without it, stacking linear layers would collapse back into a single linear model.

- **Softmax and cross-entropy loss.** The output logits are converted to a probability distribution over characters. Minimizing `F.cross_entropy(logits, targets)` is equivalent to maximizing the log-likelihood of the correct next character and is computed in a numerically stable way.

- **Mini-batch SGD and learning-rate decay.** Gradients are estimated on small random batches of 32 examples rather than the full dataset. The learning rate starts at `0.1` and drops to `0.01` after 100,000 steps so the optimizer can take large steps early and fine-tune later.

## What to watch for

- **Shape mismatches after `view`.** With `block_size=3` and `emb_dim=10`, the flattened input must be `(batch, 30)`, so `W1` must have shape `(30, hidden_size)`. A different embedding dimension requires resizing `W1` accordingly.

- **Forgetting to zero gradients.** If `p.grad = None` is omitted before `loss.backward()`, gradients accumulate across iterations and the optimizer will diverge or oscillate.

- **Confusing train, validation, and test loss.** The notebook tracks all three. Hyperparameters such as embedding size or hidden size should be tuned on the validation set; the test set is reserved only for final evaluation.

- **Sampling versus argmax generation.** At generation time, characters are drawn from the softmax distribution with `torch.multinomial`. Always choosing the most likely next character produces boring, repetitive output, whereas sampling gives diverse, name-like results.

## Tensor shapes / notation

| Tensor | Shape | Meaning |
|--------|-------|---------|
| `Xtr` | `(N, 3)` | Training indices for 3-character contexts |
| `C` | `(27, 10)` | Embedding matrix: 27 characters, 10 dimensions each |
| `emb` | `(32, 3, 10)` | Batch of 32 contexts after embedding |
| `emb.view(-1, 30)` | `(32, 30)` | Flattened context vectors |
| `W1`, `b1` | `(30, 200)`, `(200,)` | First layer weights and biases |
| `h` | `(32, 200)` | Hidden activations after `tanh` |
| `W2`, `b2` | `(200, 27)`, `(27,)` | Output layer weights and biases |
| `logits` | `(32, 27)` | Scores for each character in the batch |
| `loss` | scalar | Mean cross-entropy over the mini-batch |

## Extension ideas

- **Vary the context window and hidden size.** Try `block_size=5` or change the hidden layer width. Measure how train and validation loss change, and look for signs of overfitting when the model has too many parameters.

- **Explore temperature sampling.** Divide the logits by a `temperature` value before the softmax. Values below `1.0` make the distribution sharper and the output more conservative; values above `1.0` make it more random and creative.

- **Inspect the embedding space.** Plot the trained `C` matrix in 2-D. Characters that play similar roles in names (for example vowels or common consonants) often cluster together, revealing what the model has learned about character similarity.
