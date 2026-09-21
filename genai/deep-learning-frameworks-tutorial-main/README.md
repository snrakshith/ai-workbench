# Deep Learning Frameworks Tutorial

A comprehensive educational resource for learning **Keras** and **PyTorch** frameworks, focused on Artificial Neural Networks (ANNs) for beginner to intermediate students.

## Overview

This repository contains two detailed Jupyter notebooks that teach the fundamentals of building, training, and evaluating neural networks using Keras and PyTorch. Both notebooks cover equivalent concepts, allowing students to understand the syntax and patterns of each framework.

## What's Included

- **`keras_ann_tutorial.ipynb`** - Complete Keras tutorial with TensorFlow backend
- **`pytorch_ann_tutorial.ipynb`** - Complete PyTorch tutorial
- **`requirements.txt`** - All required Python packages
- **`SETUP.md`** - Detailed installation and setup instructions

## Learning Objectives

By completing both notebooks, you will be able to:

1. Build Artificial Neural Networks from scratch in both frameworks
2. Understand the equivalent concepts and syntax patterns across Keras and PyTorch
3. Perform binary classification, multi-class classification, and regression tasks
4. Preprocess data appropriately for neural networks
5. Train models with proper validation strategies
6. Evaluate model performance using various metrics
7. Save and load trained models for reuse
8. Apply regularization techniques (Dropout, BatchNormalization)
9. Debug common errors and issues
10. Choose the right framework for your projects

## Topics Covered

### Core Concepts (Both Frameworks)
- Data loading and preprocessing
- Model architecture design
- Layer types and parameters
- Activation functions
- Loss functions and optimizers
- Training and validation
- Model evaluation and predictions
- Model persistence (saving/loading)
- Regularization techniques
- Best practices and debugging

### Three Complete Projects
1. **Binary Classification** - Iris dataset (2 classes)
2. **Multi-Class Classification** - MNIST/Fashion-MNIST (10 classes)
3. **Regression** - California Housing price prediction

## Prerequisites

- **Python Basics**: Variables, functions, loops, basic data structures
- **Basic Machine Learning Concepts**: Understanding of supervised learning, training/testing splits
- **NumPy Fundamentals**: Basic array operations (helpful but not required)
- **Python 3.11+** installed on your system

## Quick Start

### 1. Clone or Download

```bash
cd ~/Downloads/dl-frameworks-tutorials
```

### 2. Set Up Environment

See [SETUP.md](SETUP.md) for detailed instructions, or quick start:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 4. Start Learning

Open either `keras_ann_tutorial.ipynb` or `pytorch_ann_tutorial.ipynb` and start learning!

## Recommended Learning Path

### Option 1: Framework-by-Framework
1. Complete the entire Keras notebook first
2. Then complete the entire PyTorch notebook
3. Compare the approaches and syntax

### Option 2: Concept-by-Concept
1. Learn basic model building in both frameworks
2. Learn training in both frameworks
3. Learn evaluation in both frameworks
4. Continue comparing concepts side-by-side

### Option 3: Project-Based
1. Complete binary classification project in both frameworks
2. Complete multi-class classification project in both frameworks
3. Complete regression project in both frameworks

**Recommendation**: Start with Option 1 if you're completely new to deep learning. Use Option 2 or 3 if you have some experience and want to compare frameworks.

## Notebook Structure

### Keras Notebook (~60-75 cells)
1. Introduction & Setup
2. Data Fundamentals
3. Sequential API Model Building
4. Model Compilation
5. Training Models
6. Evaluation & Prediction
7. Functional API
8. Common Layers & Techniques
9. Multi-Class Classification Project (MNIST)
10. Regression Project (California Housing)
11. Saving & Loading Models
12. Best Practices & Tips

### PyTorch Notebook (~70-85 cells)
1. Introduction & Setup
2. Tensor Basics
3. Data Handling (Dataset & DataLoader)
4. Building Your First Model (nn.Module)
5. Loss Functions & Optimizers
6. The Training Loop
7. Model Evaluation
8. Common Layers & Techniques
9. Multi-Class Classification Project (Fashion-MNIST)
10. Regression Project (Housing Prices)
11. Advanced Training Techniques
12. Saving & Loading Models
13. Best Practices & Tips

## Framework Versions

This tutorial uses the latest versions as of December 2025:
- **Keras**: 3.13.0
- **PyTorch**: 2.9.0
- **TensorFlow**: 2.15.0+ (backend for Keras)

## Datasets Used

All datasets are built-in and will download automatically:
- **Iris** (scikit-learn) - Binary classification
- **MNIST** (Keras) - Digit recognition (0-9)
- **Fashion-MNIST** (PyTorch) - Clothing classification (10 classes)
- **California Housing** (scikit-learn) - Price regression

No manual dataset downloads required!

## Additional Resources

### Official Documentation
- [Keras Documentation](https://keras.io/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [TensorFlow Documentation](https://www.tensorflow.org/)

### Further Learning
- [Deep Learning Specialization (Coursera)](https://www.coursera.org/specializations/deep-learning)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Keras Examples](https://keras.io/examples/)

## Troubleshooting

If you encounter issues:
1. Check [SETUP.md](SETUP.md) for detailed installation steps
2. Ensure you're using Python 3.11 or higher
3. Verify all packages are installed: `pip list`
4. Make sure you're in the virtual environment

Common issues and solutions are documented in the notebooks' final sections.

## Contributing

This is an educational resource. If you find errors or have suggestions for improvements, please feel free to contribute!

## License

This educational content is provided as-is for learning purposes.

## Acknowledgments

- Keras and TensorFlow teams for excellent documentation
- PyTorch team for comprehensive tutorials
- scikit-learn for easy-to-use datasets

---

**Happy Learning! 🚀**

Start with `keras_ann_tutorial.ipynb` or `pytorch_ann_tutorial.ipynb` and begin your deep learning journey!
