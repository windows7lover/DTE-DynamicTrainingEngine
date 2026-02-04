# Dynamic Training Engine (DTE)

This repository provides a **generic, block-based engine for training neural networks with adaptive and recursive execution**. 
It is designed as a reusable control layer for iteration, stopping, and unrolling during training, without hard-coding a specific algorithm.

The engine includes a **Tiny Recursive Model (TRM)** implementation as a **baseline reference**.

---

## Requirements

- Linux
- Python ≥ 3.10
- Conda (recommended)
- CUDA (optional, for GPU training)

---

## Quick Installation

```bash
# (Optional) load CUDA if you use GPUs
module load cuda/12.6.0

# Initialize conda
source ~/miniconda3/etc/profile.d/conda.sh

# Create and activate environment
conda create -n dte python=3.10 -y
conda activate dte

# Upgrade tooling inside the env
python -m pip install -U pip setuptools wheel

# Install PyTorch (CPU or CUDA build as appropriate)
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

## Quick launch

### Pretrain TRM on sorting problem

Write this line to launch the pretrain of the TRM model on a simple sorting problem.
This is with the simplest setting: no multi gpu, no co,pilation, no wandb, no resume

```bash
# Once in the folder DTE-DynamicTrainingEngine
cd example
python pretrain.py \
torch_compile.enabled=false \
checkpoint.auto_resume=false \
checkpoint.save.train=false \
checkpoint.save.model=false \
checkpoint.save.ema=false \
wandb_config.enabled=false \
```

## Reference implementation

The TRM baseline follows the original implementation and design described in:

https://github.com/SamsungSAILMontreal/TinyRecursiveModels

The provided TRM baseline reproduces the reported performance on Sudoku (~4h00 on a 
single A100)

## Acknowledgements

Special thanks to **Alexia Jolicoeur-Martineau** for her help and guidance in
implementing this system properly and ensuring reproducibility of the original TRM
performance.
