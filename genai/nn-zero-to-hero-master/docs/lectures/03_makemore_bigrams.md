# Makemore Part 1: Bigram Models

This notebook introduces the simplest character-level language model: a **bigram model** that predicts the next character from only the current character. Students first estimate a probability table by counting character pairs, then re-implement the same model as a one-layer neural network trained with gradient descent. Seeing that both approaches converge to the same math is the central insight—neural networks generalize the simple counting idea to far richer settings.

## Key concepts

- **Bigram model.** A bigram model predicts each character conditioned only on the single character before it. The notebook frames this as `P(next | current)` and stores all pair counts in a 27 × 27 matrix (26 letters plus a `.` start/end token).

- **Counting and normalizing.** The model builds a count matrix `N` and turns it into probabilities by dividing each row by its sum. This is the simplest way to estimate a conditional distribution directly from data.

- **Laplace smoothing.** Adding a constant such as `+1` to every count before normalizing prevents zero probabilities. Without smoothing, unseen bigrams would produce `-inf` log-likelihoods and could break sampling.

- **Negative log-likelihood.** A good model assigns high probability to the real data. Maximizing the product of probabilities is equivalent to minimizing the average negative log-likelihood, which is the standard cross-entropy loss for classification.

- **Bigram as a neural network.** The same model can be written as `one_hot(x) @ W`, followed by `softmax`. With enough training, the learned weight matrix `W` becomes equivalent to the log-count table, but this framework scales to deeper models later in the course.

## What to watch for

- **Rows vs. columns in the count matrix.** The notebook uses rows for the current character and columns for the next character. Mixing this up means sampling from the wrong conditional distribution.

- **Forgetting `keepdim=True`.** `P.sum(1, keepdim=True)` returns shape `(27, 1)` and broadcasts correctly across columns. Without `keepdim=True`, the result has shape `(27,)` and PyTorch can silently broadcast the wrong way, producing an invalid probability matrix.

- **Zero probabilities without smoothing.** Students should run the `alpha=0` case in the smoothing experiment to see `-inf` loss and nonsensical sampling. This motivates why even a tiny smoothing constant matters.

- **One-hot encoding is not "wasteful" here.** Each one-hot vector selects exactly one row of `W`, which is precisely the log-count table. This makes the equivalence between counting and neural-network approaches explicit.

## Tensor shapes / notation

| Tensor | Shape | Meaning |
|--------|-------|---------|
| `N` | `(27, 27)` | Raw bigram counts |
| `P` | `(27, 27)` | Row-normalized probabilities |
| `P.sum(1, keepdim=True)` | `(27, 1)` | Normalizing constants that broadcast over columns |
| `xs`, `ys` | `(num_bigrams,)` | Input and target character indices |
| `xenc = F.one_hot(xs, 27)` | `(num_bigrams, 27)` | One-hot input matrix |
| `W` | `(27, 27)` | Learned weight matrix (log-counts) |
| `logits = xenc @ W` | `(num_bigrams, 27)` | Output scores for next character |
| `probs = softmax(logits)` | `(num_bigrams, 27)` | Predicted next-character distribution |

## Extension ideas

- **Higher-order n-grams.** Try a trigram model that conditions on the previous two characters. How quickly does the count matrix become sparse, and how much smoothing do you need?

- **Smoothing grid search.** Sweep several smoothing constants (e.g., `alpha` in `[0, 0.1, 1, 10]`) and plot average negative log-likelihood on the training data. Notice the bias-variance trade-off.

- **Compare manual softmax to `F.cross_entropy`.** The notebook asks students to replace the Python loss loop with `F.cross_entropy(logits, ys)`. Verify numerically that both give the same loss and that training still converges.
