# Neural Network Project 🧠

🧠 Dynamic Multi-Label Neural Network built from scratch using a Perceptron-based architecture. Includes a flexible NeuralNetwork class with backpropagation and a CrossValidator for K-Fold cross-validation. Designed for training models with multiple binary outputs (e.g. predicting 3 possible diseases from patient features).

A small, minimal neural network project. This README explains the repository layout, how to run the project, and quick notes for development — all in English with emojis 🎉.

## Overview 🚀

This repository contains a simple Python entry script and a dependency manifest. It's intended as a starting point for experiments, training, or inference workflows.

## Project Structure 📁

- `index.py` — Main entry point. Run this script to start training, evaluation, or inference. It typically contains dataset loading and the model loop.
- `requirements.txt` — Lists Python packages required to run the project. Install with `pip install -r requirements.txt`.
- `README.md` — This file (explains the structure and how to run).

If you add more modules, consider placing them in a `src/` or `models/` folder and updating this section accordingly.

## Prerequisites ✅

- Python 3.8+ recommended
- `pip` available in your PATH

## Installation & Run 🛠️

Follow these steps to create an environment, install dependencies, and run the project:

```bash
# Create virtual environment (cross-platform)
python -m venv env

# Windows (PowerShell)
env\Scripts\Activate.ps1

# macOS / Linux
source env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the main script
python index.py
```

If `requirements.txt` is empty or missing packages, open `index.py` to see which libraries are imported and add them to `requirements.txt`.

## Tips & Notes 💡

- If you need GPU support (for example with PyTorch), install the appropriate package variant for your CUDA version.
- Keep hyperparameters and dataset paths configurable inside `index.py` for easier experimentation.
