"""
Section 6: Setting up efficient data loaders.

Run with:
    uv run scripts/05_data_loaders.py
"""
import torch
from torch.utils.data import Dataset, DataLoader


class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return self.labels.shape[0]


def main():
    X_train = torch.tensor([
        [-1.2, 3.1], [-0.9, 2.9], [-0.5, 2.6], [2.3, -1.1], [2.7, -1.5]
    ])
    y_train = torch.tensor([0, 0, 0, 1, 1])
    X_test = torch.tensor([[-0.8, 2.8], [2.6, -1.6]])
    y_test = torch.tensor([0, 1])

    train_ds = ToyDataset(X_train, y_train)
    test_ds = ToyDataset(X_test, y_test)
    print("len(train_ds):", len(train_ds))

    torch.manual_seed(123)
    train_loader = DataLoader(
        dataset=train_ds, batch_size=2, shuffle=True, num_workers=0, drop_last=True
    )
    test_loader = DataLoader(
        dataset=test_ds, batch_size=2, shuffle=False, num_workers=0
    )

    print("\nTraining batches (drop_last=True):")
    for idx, (x, y) in enumerate(train_loader):
        print(f"Batch {idx + 1}:", x, y)

    print("\nTest batches:")
    for idx, (x, y) in enumerate(test_loader):
        print(f"Batch {idx + 1}:", x, y)


if __name__ == "__main__":
    main()
