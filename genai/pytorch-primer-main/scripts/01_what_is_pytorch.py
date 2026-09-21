"""
Section 1: What is PyTorch - environment/hardware check.

Run with:
    uv run scripts/01_what_is_pytorch.py
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from utils.device import print_hardware_report


def main():
    print("torch.__version__:", torch.__version__)
    print_hardware_report()


if __name__ == "__main__":
    main()
