# 9. Optimizing training performance with GPUs

!!! abstract "Notebook / Script"
    [`notebooks/08_gpu_training.ipynb`](https://github.com/sourangshupal/pytorch-primer/blob/main/notebooks/08_gpu_training.ipynb) ·
    [`scripts/08_gpu_training.py`](https://github.com/sourangshupal/pytorch-primer/blob/main/scripts/08_gpu_training.py) ·
    See also: [Hardware Detection](../hardware-detection.md)

## 9.1 PyTorch computations on GPU devices

Every PyTorch tensor lives on a **device** (CPU, CUDA GPU, or Apple Silicon MPS), and operations on
a tensor run on whatever device it lives on. All tensors participating in an operation must be on
the *same* device.

We use the shared `utils/device.py` helper (see [Notebook 1](01-what-is-pytorch.md) and the
[Hardware Detection](../hardware-detection.md) page) so this code runs unmodified whether you
have an NVIDIA GPU, an Apple Silicon chip, or neither.

`get_device()` picks the best available device using a simple priority order:

```mermaid
flowchart TD
    A["get_device()"] --> B{CUDA available?}
    B -->|Yes| C["return 'cuda'"]
    B -->|No| D{MPS available?<br/>(Apple Silicon)}
    D -->|Yes| E["return 'mps'"]
    D -->|No| F["return 'cpu'"]
```

```python
import sys, os
sys.path.append(os.path.abspath(".."))

import torch
from utils.device import get_device, print_hardware_report

device = print_hardware_report()
```

```python
tensor_1 = torch.tensor([1., 2., 3.])
tensor_2 = torch.tensor([4., 5., 6.])

# Runs on CPU by default:
print(tensor_1 + tensor_2)

# Move both tensors to the detected device and repeat the computation there:
tensor_1 = tensor_1.to(device)
tensor_2 = tensor_2.to(device)
print(tensor_1 + tensor_2)
```

If your device printed as `cuda`, the result above shows `device='cuda:0'`. Mixing devices (e.g.
adding a CPU tensor to a CUDA tensor) raises a `RuntimeError` — PyTorch refuses to guess which
device you meant.

## 9.2 Single-device training

Moving the full training loop from [Notebook 6](06-training-loop.md) onto whatever device is
available takes exactly **three changes**: create a `device` variable, move the `model` to it, and
move each batch of data to it inside the loop.

```python
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


X_train = torch.tensor([[-1.2, 3.1], [-0.9, 2.9], [-0.5, 2.6], [2.3, -1.1], [2.7, -1.5]])
y_train = torch.tensor([0, 0, 0, 1, 1])
train_loader = DataLoader(ToyDataset(X_train, y_train), batch_size=2, shuffle=True,
                           num_workers=0, drop_last=True)

torch.manual_seed(123)
model = NeuralNetwork(num_inputs=2, num_outputs=2)

# --- Change 1: device variable (from utils.device.get_device -> CUDA > MPS > CPU) ---
# --- Change 2: move the model to that device ---
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):

        # --- Change 3: move each batch to the same device as the model ---
        features, labels = features.to(device), labels.to(device)

        logits = model(features)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
              f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
              f" | Train/Val Loss: {loss:.2f}")

    model.eval()
```

On this toy dataset you won't see a speedup from GPU training (the data transfer itself costs
more than the tiny computation) — but for real models, especially LLM-scale ones, the speedup is
often 10-100x.

!!! question "Why not hardcode `\"cuda\"`?"
    Writing `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` misses Apple
    Silicon entirely. What our `get_device()` helper does (plus an MPS check, plus the modern
    `torch.accelerator` API) is considered best practice: the exact same script then runs
    correctly on a GPU workstation, an Apple laptop, or a CPU-only CI machine, with zero changes.

## 9.3 A workload where the speedup is actually visible

The toy dataset above is too small to show a GPU/MPS advantage — data-transfer overhead dominates.
Here's a bigger matmul workload where, if you have a GPU/MPS device, the difference becomes
visible:

```python
import time

size = 4096
a_cpu, b_cpu = torch.randn(size, size), torch.randn(size, size)

start = time.perf_counter()
c_cpu = a_cpu @ b_cpu
cpu_time = time.perf_counter() - start

if device.type != "cpu":
    a_dev, b_dev = a_cpu.to(device), b_cpu.to(device)
    _ = a_dev @ b_dev  # warm-up: first accelerator call pays a one-time init cost

    start = time.perf_counter()
    c_dev = a_dev @ b_dev
    dev_time = time.perf_counter() - start
    print(f"{device.type.upper()} matmul: {dev_time*1000:.1f} ms  ({cpu_time / dev_time:.1f}x faster)")
```

## 9.4 The most common GPU bug: a tensor left on the wrong device

`RuntimeError: Expected all tensors to be on the same device` is probably the single most common
error message you'll hit once you start using an accelerator — usually from moving the *model* to
`device` but forgetting to move some *newly created* tensor:

```python
model_on_device = NeuralNetwork(num_inputs=2, num_outputs=2).to(device)
forgotten_input = torch.tensor([[1.0, 2.0]])  # NOT moved to device — the classic mistake
model_on_device(forgotten_input)
# RuntimeError: Tensor for argument input is on cpu but expected on mps
```

**Next:** what if one GPU isn't enough? [9.3 Multi-GPU training (DDP)](09-multi-gpu-ddp.md)
