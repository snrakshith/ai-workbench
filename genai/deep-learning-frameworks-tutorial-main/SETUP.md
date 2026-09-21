# Setup and Installation Guide

This guide will help you set up your environment for the Deep Learning Frameworks Tutorial.

## Prerequisites

- **Operating System**: Windows, macOS, or Linux
- **Python**: Version 3.11 or higher (3.11, 3.12, or 3.13 recommended)
- **Disk Space**: At least 5 GB free (for packages and datasets)
- **RAM**: Minimum 8 GB recommended

## Step 1: Check Python Version

Open a terminal (Command Prompt on Windows, Terminal on macOS/Linux) and check your Python version:

```bash
python3 --version
```

You should see something like `Python 3.11.x` or higher. If not, proceed to install Python.

### Installing Python

#### macOS
```bash
# Using Homebrew
brew install python@3.11
```

#### Ubuntu/Debian Linux
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip
```

#### Windows
1. Download Python 3.11+ from [python.org](https://www.python.org/downloads/)
2. Run the installer
3. **Important**: Check "Add Python to PATH" during installation
4. Verify installation: `python --version` or `python3 --version`

## Step 2: Navigate to Project Directory

```bash
cd ~/Downloads/dl-frameworks-tutorials
```

On Windows:
```cmd
cd C:\Users\YourUsername\Downloads\dl-frameworks-tutorials
```

## Step 3: Create Virtual Environment

A virtual environment isolates project dependencies from your system Python installation.

### Using venv (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows (Command Prompt)
venv\Scripts\activate

# Activate on Windows (PowerShell)
venv\Scripts\Activate.ps1
```

When activated, you should see `(venv)` at the beginning of your terminal prompt.

### Using Conda (Alternative)

If you prefer Conda:

```bash
# Create conda environment
conda create -n dl-tutorial python=3.11

# Activate environment
conda activate dl-tutorial
```

## Step 4: Install Dependencies

With your virtual environment activated, install all required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- TensorFlow (>= 2.15.0)
- Keras (>= 3.13.0)
- PyTorch (>= 2.9.0)
- torchvision (>= 0.20.0)
- NumPy, Pandas, Matplotlib, Seaborn
- scikit-learn
- Jupyter Notebook

**Note**: Installation may take 5-15 minutes depending on your internet speed.

### Platform-Specific PyTorch Installation

If you want GPU support for PyTorch (optional), visit [pytorch.org](https://pytorch.org/get-started/locally/) and use their installation command instead.

For CPU-only (default from requirements.txt):
```bash
pip install torch torchvision
```

For NVIDIA GPU support (Linux/Windows):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

For Apple Silicon (M1/M2/M3) Mac:
```bash
# PyTorch has Metal Performance Shaders (MPS) support
# The default installation from requirements.txt already includes this
```

## Step 5: Verify Installation

Run this Python command to verify all packages are installed correctly:

```bash
python3 -c "import tensorflow as tf; import keras; import torch; import torchvision; print(f'TensorFlow: {tf.__version__}'); print(f'Keras: {keras.__version__}'); print(f'PyTorch: {torch.__version__}'); print('All packages installed successfully!')"
```

Expected output:
```
TensorFlow: 2.15.x (or higher)
Keras: 3.13.x (or higher)
PyTorch: 2.9.x (or higher)
All packages installed successfully!
```

## Step 6: Launch Jupyter Notebook

```bash
jupyter notebook
```

This will:
1. Start the Jupyter server
2. Open your default web browser
3. Display the Jupyter file browser

You should see the tutorial notebooks:
- `keras_ann_tutorial.ipynb`
- `pytorch_ann_tutorial.ipynb`

Click on either notebook to open it and start learning!

## Step 7: Test Notebook

Open `keras_ann_tutorial.ipynb` and run the first cell to ensure everything works:

1. Click on the first code cell
2. Press `Shift + Enter` to run it
3. You should see version information without errors

## GPU Setup (Optional)

### For NVIDIA GPUs (Linux/Windows)

1. **Install NVIDIA Drivers**: [Download here](https://www.nvidia.com/download/index.aspx)

2. **Install CUDA Toolkit 12.1**: [Download here](https://developer.nvidia.com/cuda-downloads)

3. **Install cuDNN**: [Download here](https://developer.nvidia.com/cudnn)

4. **Install GPU-enabled PyTorch**:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

5. **Verify GPU access**:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

### For Apple Silicon Macs (M1/M2/M3)

PyTorch automatically uses Metal Performance Shaders (MPS) for GPU acceleration:

```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
```

TensorFlow on Apple Silicon:
```bash
pip install tensorflow-metal  # For GPU acceleration
```

**Note**: GPU support is **optional** for these tutorials. All code works on CPU.

## Troubleshooting

### Issue: `pip install` fails with permission error

**Solution**: Use `--user` flag or ensure virtual environment is activated
```bash
pip install --user -r requirements.txt
```

### Issue: Jupyter notebook doesn't open in browser

**Solution**: Copy the URL from terminal and paste in browser manually
```
http://localhost:8888/?token=...
```

### Issue: Module not found error in notebook

**Solution**: Install Jupyter in the same environment
```bash
pip install jupyter ipykernel
python3 -m ipykernel install --user --name=dl-tutorial
```

Then select the `dl-tutorial` kernel in Jupyter: Kernel → Change Kernel → dl-tutorial

### Issue: TensorFlow/Keras import warnings

**Solution**: These are usually just informational messages. You can ignore them or suppress:
```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
import tensorflow as tf
```

### Issue: PyTorch installation is very large

**Solution**: This is normal. PyTorch with dependencies can be 1-2 GB.

### Issue: Slow training on Mac

**Solution**: For M1/M2/M3 Macs, ensure you're using the native arm64 Python, not x86_64 via Rosetta:
```bash
python3 -c "import platform; print(platform.machine())"
```
Should output `arm64`, not `x86_64`.

### Issue: Kernel keeps dying

**Solution**: Increase available RAM or reduce batch sizes in the code:
```python
# Instead of batch_size=128
batch_size = 32  # Smaller batch size
```

## Updating Packages

To update to the latest versions:

```bash
pip install --upgrade tensorflow keras torch torchvision
```

## Deactivating Virtual Environment

When you're done:

```bash
# For venv
deactivate

# For conda
conda deactivate
```

## Uninstalling

To completely remove the environment:

### venv
```bash
rm -rf venv  # macOS/Linux
rmdir /s venv  # Windows
```

### conda
```bash
conda env remove -n dl-tutorial
```

## Next Steps

1. Open `README.md` to understand the learning objectives
2. Launch Jupyter Notebook: `jupyter notebook`
3. Start with `keras_ann_tutorial.ipynb` or `pytorch_ann_tutorial.ipynb`
4. Run cells sequentially (Shift + Enter)
5. Read explanations carefully
6. Experiment with the code!

## Getting Help

- **Jupyter Shortcuts**: Press `H` in Jupyter to see keyboard shortcuts
- **Package Documentation**: Links provided in README.md
- **Common Errors**: Check the final section of each notebook

---

**You're all set! Happy learning! 🎓**
