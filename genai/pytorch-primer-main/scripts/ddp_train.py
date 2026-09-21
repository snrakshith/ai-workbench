"""
Multi-GPU training with PyTorch's DistributedDataParallel (DDP).

Blog reference: "PyTorch in One Hour", Section 9.3, Training with multiple GPUs.

IMPORTANT:
- This script requires 2+ CUDA GPUs to demonstrate real distributed training
  (it also runs, degenerately, on 1 GPU or CPU via gloo, mostly to sanity-check
  the code path).
- It MUST be launched via `torchrun`, not `python ddp_train.py` directly, and
  NOT from a Jupyter/IPython kernel. DDP spawns one independent OS process
  per GPU; notebooks don't support that process model.

Usage:
    # Run on exactly 2 GPUs:
    torchrun --nproc_per_node=2 scripts/ddp_train.py

    # Run on every GPU available on this machine:
    torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) scripts/ddp_train.py

See notebooks/09_multi_gpu_ddp.ipynb for the full conceptual walkthrough of
every piece of this script.
"""

import os
import platform

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


def ddp_setup(rank: int, world_size: int) -> None:
    """
    Initialize the distributed process group (one process per GPU) so the
    processes can communicate (e.g. to synchronize gradients).

    Arguments:
        rank: this process's unique ID (0, 1, 2, ...)
        world_size: total number of processes in the group
    """
    # Only set MASTER_ADDR / MASTER_PORT if torchrun hasn't already set them.
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12345"

    if platform.system() == "Windows":
        # PyTorch for Windows isn't built with libuv support.
        os.environ["USE_LIBUV"] = "0"
        # gloo: Facebook Collective Communication Library (portable fallback)
        init_process_group(backend="gloo", rank=rank, world_size=world_size)
    else:
        # nccl: NVIDIA Collective Communication Library (fast GPU-to-GPU comms)
        init_process_group(backend="nccl", rank=rank, world_size=world_size)

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)


class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return self.labels.shape[0]


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


def prepare_dataset():
    X_train = torch.tensor([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5],
    ])
    y_train = torch.tensor([0, 0, 0, 1, 1])

    X_test = torch.tensor([
        [-0.8, 2.8],
        [2.6, -1.6],
    ])
    y_test = torch.tensor([0, 1])

    # Uncomment to scale the toy dataset up so this script can run on up to 8 GPUs:
    # factor = 4
    # X_train = torch.cat([X_train + torch.randn_like(X_train) * 0.1 for _ in range(factor)])
    # y_train = y_train.repeat(factor)
    # X_test = torch.cat([X_test + torch.randn_like(X_test) * 0.1 for _ in range(factor)])
    # y_test = y_test.repeat(factor)

    train_ds = ToyDataset(X_train, y_train)
    test_ds = ToyDataset(X_test, y_test)

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=False,          # False: DistributedSampler handles shuffling
        pin_memory=True,
        drop_last=True,
        sampler=DistributedSampler(train_ds),  # chunk batches across GPUs, no overlap
    )
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=2,
        shuffle=False,
    )
    return train_loader, test_loader


def compute_accuracy(model, dataloader, device):
    model = model.eval()
    correct = 0.0
    total_examples = 0

    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)
        with torch.no_grad():
            logits = model(features)
        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)
    return (correct / total_examples).item()


def main(rank: int, world_size: int, num_epochs: int) -> None:
    ddp_setup(rank, world_size)

    train_loader, test_loader = prepare_dataset()
    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    model.to(rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)
    # The unwrapped model is accessible as model.module if you need it.

    for epoch in range(num_epochs):
        # Ensure each epoch reshuffles differently across processes.
        train_loader.sampler.set_epoch(epoch)

        model.train()
        for features, labels in train_loader:
            features, labels = features.to(rank), labels.to(rank)
            logits = model(features)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"[GPU{rank}] Epoch: {epoch+1:03d}/{num_epochs:03d}"
                  f" | Batchsize {labels.shape[0]:03d}"
                  f" | Train/Val Loss: {loss:.2f}")

    model.eval()

    try:
        train_acc = compute_accuracy(model, train_loader, device=rank)
        print(f"[GPU{rank}] Training accuracy", train_acc)
        test_acc = compute_accuracy(model, test_loader, device=rank)
        print(f"[GPU{rank}] Test accuracy", test_acc)
    except ZeroDivisionError as e:
        raise ZeroDivisionError(
            f"{e}\n\nThis script is designed for 2 GPUs. Run it as:\n"
            "torchrun --nproc_per_node=2 scripts/ddp_train.py\n"
            f"Or, to run it on {torch.cuda.device_count()} GPUs, uncomment the "
            "dataset-scaling lines in prepare_dataset()."
        )

    destroy_process_group()


if __name__ == "__main__":
    # torchrun sets these environment variables for us.
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))

    if rank == 0:
        print("PyTorch version:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        print("Number of GPUs available:", torch.cuda.device_count())

    torch.manual_seed(123)
    main(rank, world_size, num_epochs=3)
