"""
Section 8: Saving and loading models.

Run with:
    uv run scripts/07_saving_and_loading_models.py
"""
import torch


class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        return self.layers(x)


def main():
    torch.manual_seed(123)
    model = NeuralNetwork(2, 2)

    torch.save(model.state_dict(), "model.pth")
    print("Saved model.state_dict() to model.pth")

    model2 = NeuralNetwork(2, 2)  # must match the original architecture
    model2.load_state_dict(torch.load("model.pth", weights_only=True))
    print("Restored model2 from model.pth (weights_only=True)")


if __name__ == "__main__":
    main()
