# Glossary

A concise reference for terms used across the course.

## A

**Activation function**
A non-linear function applied after a linear transformation (e.g., `tanh`,
`ReLU`). Without non-linearities, stacking multiple layers would be equivalent
to a single linear layer.

**Autograd**
Automatic differentiation. A system that records operations and applies the
chain rule to compute gradients automatically. PyTorch's autograd is built on
a tensor-level graph; micrograd builds the same idea for scalars.

## B

**Backpropagation**
The algorithm that computes gradients of a loss with respect to every
parameter by applying the chain rule backward through the computation graph.

**Batch normalization**
A layer that normalizes the inputs to the next layer using the mean and
variance of the current mini-batch, then rescales with learned `gamma` and
`beta` parameters. It makes deep networks easier to train.

**Bias**
An additive parameter in a linear layer: `out = x @ W + b`.

**Bigram model**
A language model that predicts the next token based only on the immediately
preceding token.

## C

**Chain rule**
The calculus rule that lets us compute the derivative of a composition of
functions: `dL/dx = dL/dy * dy/dx`.

**Computation graph**
A directed graph where nodes are values/operations and edges show how values
flow forward and gradients flow backward.

**Context length**
The number of previous tokens a language model looks at to predict the next
token. Also called `block_size` in the notebooks.

**Cross-entropy loss**
A loss function for classification that measures how surprised the model is by
the correct class. For a single example: `-log(p_correct)`.

## D

**Derivative**
The instantaneous rate of change of a function with respect to one of its
inputs. We approximate it with `(f(x+h) - f(x)) / h` for tiny `h`.

## E

**Embedding**
A learned lookup table that maps discrete tokens (characters, words) to dense
vectors. In the notebooks: `C[ix]` gives the embedding vector for character
`ix`.

## F

**Forward pass**
Computing the output of a model given an input.

## G

**Gradient**
The vector of partial derivatives of a loss with respect to parameters. It
points in the direction that increases the loss, so we move in the opposite
direction to minimize loss.

**Gradient descent**
An optimization algorithm that updates parameters using `p = p - lr * grad`.

## H

**Hidden layer**
A layer of neurons between the input and output. "Hidden" because its values
are not directly observed in the data.

## K

**Kaiming initialization**
A weight initialization scheme that scales random weights by `1 / sqrt(fan_in)`
(or a gain factor) so that activations keep a stable variance through deep
layers.

## L

**Language model**
A model that assigns probabilities to sequences of tokens and can generate new
sequences by sampling one token at a time.

**Learning rate**
A hyperparameter that controls the step size during gradient descent. Too
large: unstable. Too small: slow.

**Logits**
The raw, unnormalized scores output by the last linear layer before applying
softmax.

**Loss function**
A scalar that measures how bad the model's predictions are. Training minimizes
it.

## M

**Mini-batch gradient descent**
Computing the loss and gradient on a small random subset (mini-batch) of the
data, then updating parameters. Faster and noisier than full-batch descent.

**MLP (Multi-Layer Perceptron)**
A feed-forward neural network with one or more hidden layers.

## N

**Negative log-likelihood (NLL)**
The loss obtained by taking `-log(probability_assigned_to_correct_token)`. Averaging
over examples gives the cross-entropy loss.

**Neuron**
A computation unit: weighted sum of inputs plus bias, passed through an
activation function.

## O

**One-hot encoding**
A vector representation of a category where exactly one element is `1` and the
rest are `0`. Used in the bigram notebook to feed character indices into a
linear layer.

**Overfitting**
When a model learns the training data too well and performs poorly on unseen
data. Detected by a growing gap between training and validation loss.

## P

**Parameter**
A tensor that the model learns during training (weights, biases, embeddings,
BatchNorm `gamma`/`beta`).

**Perceptron**
Another name for a single artificial neuron.

## R

**ReLU**
Rectified Linear Unit: `f(x) = max(0, x)`.

**Receptive field**
The range of input positions a particular output can "see." In WaveNet-style
models, the receptive field grows exponentially with depth.

## S

**Sampling**
Generating new tokens by repeatedly picking from the model's predicted
probability distribution.

**Softmax**
A function that turns logits into a probability distribution: probabilities are
positive and sum to 1.

**Stochastic gradient descent (SGD)**
Gradient descent using mini-batches and (optionally) shuffling.

## T

**tanh**
The hyperbolic tangent activation: outputs in `(-1, 1)`, S-shaped, smooth.

**Tensor**
A multi-dimensional array. PyTorch tensors can track gradients via
`requires_grad=True`.

**Token**
A discrete unit of text. In these notebooks, characters are tokens.

**Topological sort**
An ordering of graph nodes such that every node appears after its children.
Used in backprop to ensure gradients flow in the correct order.

## U

**Underfitting**
When a model is too simple to capture the patterns in the data. Both training
and validation loss remain high.

## V

**Validation set**
A held-out portion of the data used to tune hyperparameters and detect
overfitting. Distinct from the test set, which is used only at the end.

**Vanishing/exploding gradients**
When gradients become extremely small or large in deep networks, making
learning unstable. Kaiming init and BatchNorm help prevent this.

## W

**WaveNet**
A deep generative model for audio (and later text) that uses hierarchical,
dilated convolutions. The final makemore notebook builds a character-level
analog with `FlattenConsecutive` layers.

**Weight**
A multiplicative parameter in a linear layer: `out = x @ W + b`.

**Weight initialization**
Choosing the starting values of weights. Good initialization helps activations
and gradients stay at healthy scales.
