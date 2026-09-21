"""
Section 5: Implementing multilayer neural networks.

Run with:
    uv run scripts/04_building_neural_networks.py
"""
import torch


class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),
            # 2nd hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),
            # output layer
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        return self.layers(x)


def main():
    torch.manual_seed(123)
    model = NeuralNetwork(50, 3)
    print(model)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of trainable model parameters:", num_params)

    print("layer[0].weight.shape:", model.layers[0].weight.shape)

    torch.manual_seed(123)
    X = torch.rand((1, 50))
    out = model(X)
    print("forward pass (with grad):", out)

    with torch.no_grad():
        out = model(X)
    print("forward pass (no_grad):", out)

    with torch.no_grad():
        probas = torch.softmax(model(X), dim=1)
    print("class-membership probabilities:", probas)


if __name__ == "__main__":
    main()
