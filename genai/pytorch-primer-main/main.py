"""
Quick smoke test for the pytorch-primer environment.

Run with:
    uv run main.py

If this prints a PyTorch version and a detected device, your environment
is ready for notebooks/ and scripts/.
"""
from utils.device import print_hardware_report


def main():
    print("pytorch-primer: environment check\n" + "-" * 34)
    print_hardware_report()


if __name__ == "__main__":
    main()
