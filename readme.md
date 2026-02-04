# Dynamic Training Engine (DTE) - v0.1 (beta)

This repository provides a **generic, block-based engine for training neural networks with adaptive and recursive execution**. 
It is designed as a reusable control layer for iteration, stopping, and unrolling during training, without hard-coding a specific algorithm.

The engine includes a **Tiny Recursive Model (TRM)** implementation as a **baseline reference**.

## Project status

**v0.1 (beta)**

This release provides a stable core training engine and a reference TRM baseline.
The API and configuration structure may still evolve.

---

## Requirements

- Linux
- Python ≥ 3.10
- Conda (recommended)
- CUDA (optional, for GPU training)

---

## Quick Installation

```bash
# Load CUDA
module load cuda/12.6.0

# IMPORTANT:
# Do NOT load a system Python module when using conda.
# (e.g. do not write module load python/3.10)
# Conda must own the Python interpreter.

# Initialize conda
source ~/miniconda3/etc/profile.d/conda.sh

# Create and activate environment
conda create -n dte-env python=3.10 -y
conda activate dte-env

# Upgrade tooling inside the conda env
python -m pip install -U pip setuptools wheel

# Install PyTorch
pip install torch

# Clone and install the package
git clone https://github.com/windows7lover/DTE-DynamicTrainingEngine.git
cd DTE-DynamicTrainingEngine
pip install -e .
```

Check the installation by writing in your console

```bash
python -c "import dte; print('DTE installed')"
```

### Simple pretraining of TRM on a sorting task

The following command launches pretraining of the TRM model on a simple
vector sorting task.

This run uses the **simplest configuration**:
- single GPU or CPU
- no compilation
- no Weights & Biases logging

The task consists of sorting input vectors of length 2–100, with integer values
in the range [1, 64].

```bash
# From the DTE-DynamicTrainingEngine repository
cd example
python pretrain.py \
  --config-name=config \
  torch_compile.enabled=false \
  wandb_config.enabled=false
```

By default, the pretraining script loads `config/config.yaml`.

To enable compilation, either remove the override or explicitly set:
```bash
torch_compile.enabled=true
```

Enabling compilation typically yields a **~2.5× speedup**, depending on hardware.

### Optimized multi-GPU training

The DTE package and the provided `pretrain.py` script are compatible with
**Distributed Data Parallel (DDP)**.

The following command launches pretraining on 4 GPUs, with
compilation enabled and Weights & Biases logging active:

```bash
# From the DTE-DynamicTrainingEngine/example folder
torchrun --nproc-per-node=4 pretrain.py --config-name=config
```

---

## Reference implementation

The TRM baseline follows the original implementation and design described in:

https://github.com/SamsungSAILMontreal/TinyRecursiveModels

The provided TRM baseline reproduces the reported performance on Sudoku (~4h00 on a 
single A100, using compilation)

If you use the **TRM baseline** provided in this repository, please consider citing:

```bibtex
@article{jolicoeur2025less,
  title   = {Less is More: Recursive Reasoning with Tiny Networks},
  author  = {Jolicoeur-Martineau, Alexia},
  journal = {arXiv preprint arXiv:2510.04871},
  year    = {2025}
}
```

## Acknowledgements

Special thanks to **Alexia Jolicoeur-Martineau** for her help and guidance in
implementing this system properly and ensuring reproducibility of the original TRM
performance.

