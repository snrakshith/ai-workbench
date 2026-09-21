"""
Section 9.1-9.2: PyTorch computations on GPU devices + single-device training.

Uses utils/device.py to auto-detect CUDA / Apple Silicon (MPS) / CPU, so this
script runs unmodified on any machine.

Run with:
    uv run scripts/08_gpu_training.py
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from utils.device import print_hardware_report


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


class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return self.labels.shape[0]


def main():
    device = print_hardware_report()

    # 9.1: moving tensors to a device
    tensor_1 = torch.tensor([1., 2., 3.]).to(device)
    tensor_2 = torch.tensor([4., 5., 6.]).to(device)
    print("\ntensor_1 + tensor_2 on", device, "->", tensor_1 + tensor_2)

    # 9.2: single-device training loop (identical to scripts/06_training_loop.py
    # plus the 3 device-related changes noted below)
    X_train = torch.tensor([
        [-1.2, 3.1], [-0.9, 2.9], [-0.5, 2.6], [2.3, -1.1], [2.7, -1.5]
    ])
    y_train = torch.tensor([0, 0, 0, 1, 1])
    train_loader = DataLoader(
        ToyDataset(X_train, y_train), batch_size=2, shuffle=True,
        num_workers=0, drop_last=True,
    )

    torch.manual_seed(123)
    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    model.to(device)  # Change 1 + 2: device variable + move model to it

    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    num_epochs = 3

    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)  # Change 3

            logits = model(features)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch: {epoch + 1:03d}/{num_epochs:03d}"
                  f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
                  f" | Train/Val Loss: {loss:.2f}")
        model.eval()


if __name__ == "__main__":
    main()
