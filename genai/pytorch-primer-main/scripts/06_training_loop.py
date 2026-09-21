"""
Section 7: A typical training loop.

Run with:
    uv run scripts/06_training_loop.py
"""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


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


def compute_accuracy(model, dataloader):
    model.eval()
    correct = 0.0
    total_examples = 0
    for features, labels in dataloader:
        with torch.no_grad():
            logits = model(features)
        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)
    return (correct / total_examples).item()


def main():
    X_train = torch.tensor([
        [-1.2, 3.1], [-0.9, 2.9], [-0.5, 2.6], [2.3, -1.1], [2.7, -1.5]
    ])
    y_train = torch.tensor([0, 0, 0, 1, 1])
    X_test = torch.tensor([[-0.8, 2.8], [2.6, -1.6]])
    y_test = torch.tensor([0, 1])

    train_loader = DataLoader(
        ToyDataset(X_train, y_train), batch_size=2, shuffle=True,
        num_workers=0, drop_last=True,
    )
    test_loader = DataLoader(
        ToyDataset(X_test, y_test), batch_size=2, shuffle=False, num_workers=0
    )

    torch.manual_seed(123)
    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (features, labels) in enumerate(train_loader):
            logits = model(features)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch: {epoch + 1:03d}/{num_epochs:03d}"
                  f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
                  f" | Train/Val Loss: {loss:.2f}")
        model.eval()

    print("\nTrain accuracy:", compute_accuracy(model, train_loader))
    print("Test accuracy:", compute_accuracy(model, test_loader))


if __name__ == "__main__":
    main()
