# DDP Script Walkthrough

!!! abstract "Script"
    [`scripts/ddp_train.py`](https://github.com/sourangshupal/pytorch-primer/blob/main/scripts/ddp_train.py)

This is the full, runnable version of the concepts covered in
[9.3 Multi-GPU training (DDP)](track1-raschka/09-multi-gpu-ddp.md). Unlike every other script in
this repo, it **cannot** be run with plain `python` or inside a notebook — it needs 2+ CUDA GPUs
and must be launched with `torchrun`, which spawns one independent OS process per GPU.

```bash
torchrun --nproc_per_node=2 scripts/ddp_train.py
```

## `ddp_setup` — one-time process-group setup

```python
def ddp_setup(rank: int, world_size: int) -> None:
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
```

Every process calls this once, at the very start. `rank` is this process's unique ID (`0`, `1`,
`2`, ...); `world_size` is the total process/GPU count. `nccl` is used on Linux/CUDA for
fast GPU-to-GPU communication; `gloo` is the portable fallback used on Windows.

## The model and dataset — identical to Notebook 6/8

```python
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
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        return self.layers(x)
```

Nothing new here — this is exactly the toy `ToyDataset`/`NeuralNetwork` pair from
[Notebook 6 — Training Loop](track1-raschka/06-training-loop.md), reused so this script is a
minimal, self-contained DDP example.

## `prepare_dataset` — the one DDP-specific change to `DataLoader`

```python
def prepare_dataset():
    X_train = torch.tensor([
        [-1.2, 3.1], [-0.9, 2.9], [-0.5, 2.6], [2.3, -1.1], [2.7, -1.5],
    ])
    y_train = torch.tensor([0, 0, 0, 1, 1])

    X_test = torch.tensor([[-0.8, 2.8], [2.6, -1.6]])
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
```

Note `shuffle=False` on the training loader — shuffling is now the `DistributedSampler`'s job, not
the `DataLoader`'s. Passing both would conflict.

!!! note "Toy dataset size"
    With only 5 training examples, this script is designed for exactly 2 GPUs. The commented-out
    `factor` lines scale the dataset up (with small random jitter) so it can be run on up to 8 GPUs
    without every GPU getting an empty shard.

## `compute_accuracy` — device-parameterized version

```python
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
```

Same shape as [Notebook 6's `compute_accuracy`](track1-raschka/06-training-loop.md), with an
explicit `device` argument since each DDP process runs on a different GPU (`rank`).

## `main` — the full training loop

```python
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
```

Key points:

- `model.to(rank)` moves the model to *this process's* GPU — `rank` doubles as the device index.
- `DDP(model, device_ids=[rank])` wraps the model **after** moving it to the device, so gradient
  synchronization happens across GPUs transparently inside `loss.backward()`.
- `train_loader.sampler.set_epoch(epoch)` must be called every epoch — without it, the
  `DistributedSampler` would produce the same shuffle order every epoch, which hurts training.
- Every print is prefixed `[GPU{rank}]` since all processes print independently — see the
  ["duplicated print output" gotcha](track1-raschka/09-multi-gpu-ddp.md#one-gotcha-duplicated-print-output)
  from the concept page for the `if rank == 0` pattern used to avoid this in other examples.
- `destroy_process_group()` cleanly tears down inter-process communication at the end.

## Entry point

```python
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
```

`torchrun` injects `WORLD_SIZE`, `RANK`, and `LOCAL_RANK` into every spawned process's environment
— the script reads them here rather than taking them as command-line arguments.
