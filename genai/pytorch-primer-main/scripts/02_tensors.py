"""
Section 2: Understanding tensors.

Run with:
    uv run scripts/02_tensors.py
"""
import torch


def main():
    # 2.1 Scalars, vectors, matrices, and tensors
    tensor0d = torch.tensor(1)
    tensor1d = torch.tensor([1, 2, 3])
    tensor2d = torch.tensor([[1, 2], [3, 4]])
    tensor3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    print("0D:", tensor0d)
    print("1D:", tensor1d)
    print("2D:", tensor2d)
    print("3D:", tensor3d)

    # 2.2 Tensor data types
    print("\nint tensor dtype:", tensor1d.dtype)          # torch.int64
    floatvec = torch.tensor([1.0, 2.0, 3.0])
    print("float tensor dtype:", floatvec.dtype)           # torch.float32
    floatvec = tensor1d.to(torch.float32)
    print("converted dtype:", floatvec.dtype)              # torch.float32

    # 2.3 Common PyTorch tensor operations
    tensor2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print("\ntensor2d:\n", tensor2d)
    print("shape:", tensor2d.shape)
    print("reshape(3, 2):\n", tensor2d.reshape(3, 2))
    print("view(3, 2):\n", tensor2d.view(3, 2))
    print("transpose (.T):\n", tensor2d.T)
    print("matmul:\n", tensor2d.matmul(tensor2d.T))
    print("@ operator:\n", tensor2d @ tensor2d.T)


if __name__ == "__main__":
    main()
