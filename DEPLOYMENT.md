# Factored Subspace Compression for Edge Pruning — Implementation & Deployment Guide

## Table of Contents

1. [What Was Implemented](#1-what-was-implemented)
2. [Compute Requirements](#2-compute-requirements)
3. [Deployment on Princeton Adroit](#3-deployment-on-princeton-adroit)
4. [Deployment on Google Colab Pro](#4-deployment-on-google-colab-pro)

---

## 1. What Was Implemented

### 1.1 Core Idea

The standard Edge Pruning algorithm stores disentangled residual stream activations in a `(num_writers, batch, seq, d)` tensor — the "outbox" — where `d` is the model's hidden dimension. Every edge-masking operation (the O(N^2) routing) runs in this full `d`-dimensional space, and PyTorch's autograd retains these large tensors for the backward pass.

**Factored Subspace Compression** projects the outbox into a low-rank latent space of dimension `k << d`:

```
Source Compression:    c_j = P * y_j          (d -> k)
Latent Routing:        S = sum( z_ji * c_j )  (all masking in R^k)
Dest. Decompression:   h_i = U * S            (k -> d)
```

This reduces the outbox from `(writers, batch, seq, d)` to `(writers, batch, seq, k)`, and — critically — the custom autograd functions avoid saving the large `d`-dimensional tensors in the backward graph.

### 1.2 Files Created

All original files are **untouched**. Every compressed variant lives in a separate file for clean comparison.

#### Modeling Files (3)

| File | Model | Lines Changed vs Original |
|------|-------|--------------------------|
| `src/modeling/modeling_fpt2_compressed.py` | GPT-2 | +150 lines (autograd functions, compression in read/write/forward) |
| `src/modeling/modeling_erazr_compressed.py` | Tracr | +140 lines (same pattern, adapted for `apply_inv_proj`) |
| `src/modeling/modeling_fllama_compressed.py` | CodeLlama | +160 lines (same pattern, adapted for single-embedding RoPE arch) |

#### Pruning Scripts (3)

| File | Based On |
|------|----------|
| `src/prune/fpt2_ioi_compressed.py` | `fpt2_ioi.py` |
| `src/prune/erazr_reverse_compressed.py` | `erazr_reverse.py` |
| `src/prune/fllama_boolean_expressions_ip_compressed.py` | `fllama_boolean_expressions_ip.py` |

#### Run Scripts (3)

| File | Model | Default k |
|------|-------|-----------|
| `run_scripts/ioi_sweep_compressed.sh` | GPT-2 / IOI | k=64 |
| `run_scripts/tracr_reverse_compressed.sh` | Tracr / Reverse | k=16 |
| `run_scripts/launch_fllama_instr_prune_compressed.sh` | CodeLlama / Boolean Expressions | k=256 |

#### Test Suite (1)

| File | Tests |
|------|-------|
| `test_edge_pruning_compression.py` | 8 tests across 4 classes (VRAM comparison, OOM wall, large model, gradient flow) |

### 1.3 Key Technical Decisions

**Custom Autograd Functions.** `CompressProjection` and `DecompressProjection` override `torch.autograd.Function` to save only the small projection matrix `P` (k x d) during backward — not the d-dimensional input. This is the core memory optimization. Standard `x @ P.t()` would save `x` (d-dim) for backward, but since `grad_x = grad_output @ P`, we only need `P`.

**Dual corr_x.** Corrupted activations are split at the top of each model's `forward()`:
- `corr_x_k` (k-dim) — used in read methods for edge-level masked routing
- `corr_x_d` (d-dim) — used in write methods for node-level ablation

Both are detached tensors (from `torch.no_grad()`), so they add no autograd overhead.

**Fallback.** When `compression_dim=0`, all projection references are `None` and every `if P is not None` branch is skipped, producing behavior identical to the original code.

### 1.4 New CLI Arguments

All three pruning scripts accept:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--compression_dim` | int | 0 | Latent dimension k. 0 disables compression. |
| `--compression_init` | str | `"gaussian"` | `"gaussian"` (JL sketching), `"orthogonal"`, or file path to pretrained weights |
| `--freeze_projections` | bool | True | Freeze P and U during mask optimization |

---

## 2. Compute Requirements

### 2.1 Per-Experiment Summary

| Experiment | GPUs | VRAM/GPU | Wall Time | Disk (data) | Disk (output) |
|-----------|------|----------|-----------|-------------|---------------|
| **GPT-2 / IOI** (1 sparsity level) | 1 | 8–12 GB | ~30 min | 5 MB | ~1 GB |
| **GPT-2 / IOI** (full 17-level sweep) | 1 | 8–12 GB | ~8 hrs | 5 MB | ~17 GB |
| **GPT-2 / IOI Compressed** (k=64, 1 level) | 1 | 6–9 GB | ~30 min | 5 MB | ~1 GB |
| **Tracr / Reverse** | 1 | 2–4 GB | ~20 min | 15 MB | ~500 MB |
| **Tracr / Reverse Compressed** (k=16) | 1 | 2–3 GB | ~20 min | 15 MB | ~500 MB |
| **CodeLlama 13B** (standard) | 32 (4 nodes x 8) | 40–80 GB/GPU | ~25 hrs | 100 MB | ~2 GB |
| **CodeLlama 13B Compressed** (k=256) | 32 (4 nodes x 8) | 30–60 GB/GPU | ~22 hrs | 100 MB | ~1.5 GB |
| **Test suite** | 1 | 16–24 GB | ~10 min | 0 | 0 |

### 2.2 Software Dependencies

**Standard environment** (`requirements.txt`):
```
Python          >= 3.9 (recommended 3.10–3.11)
PyTorch         2.5.1 + CUDA 12.1
transformers    (latest compatible)
accelerate      (for multi-GPU)
datasets        (HuggingFace)
flash-attn      2.7.2
jax             0.5.2        (Tracr only)
tensorflow      2.18.0       (Tracr only)
networkx, graphviz, scipy, numpy, tqdm, einops
```

**Updated environment** (`requirements-experimental.txt`) — recommended for GPT-2:
```
PyTorch         2.7.1
transformers    4.54.0       (fixes log_alpha init bugs)
flash-attn      2.8.2
```

### 2.3 GPU Compatibility

| GPU | VRAM | GPT-2 | Tracr | CodeLlama | Test Suite |
|-----|------|-------|-------|-----------|------------|
| T4 | 16 GB | Standard + Compressed | Yes | No | Exp 1 + 4 only |
| V100 | 16–32 GB | Yes | Yes | FSDP only (32 GB) | Yes |
| A100 40 GB | 40 GB | Yes | Yes | FSDP (8+ GPUs) | Yes |
| A100 80 GB | 80 GB | Yes | Yes | FSDP (4+ GPUs) | Yes |
| RTX 4090 | 24 GB | Yes | Yes | No | Yes |
| L4 (Colab) | 22.5 GB | Yes | Yes | No | Yes |

### 2.4 Data Preparation

Before any experiment, extract the data archive:
```bash
cd /path/to/Edge-Pruning
unzip data.zip
```

This creates `data/datasets/`, `data/tracr_models/`, and `data/runs/`. Total size: ~500 MB.

---

## 3. Deployment on Princeton Adroit

### 3.1 Cluster Overview

Princeton's Adroit cluster provides:
- NVIDIA A100 80 GB GPUs (`gpu80` constraint) and V100 32 GB GPUs
- SLURM job scheduler
- Scratch storage at `/scratch/gpfs/<netid>/`
- Module system for CUDA, conda, etc.

### 3.2 One-Time Setup

```bash
# 1. SSH into Adroit
ssh <netid>@adroit.princeton.edu

# 2. Clone the repository to scratch (faster I/O than home)
cd /scratch/gpfs/<netid>
git clone <repo-url> Edge-Pruning
cd Edge-Pruning

# 3. Extract data
unzip data.zip

# 4. Create conda environment
module load anaconda3/2024.2
conda create -n edgeprune python=3.11 -y
conda activate edgeprune

# 5. Install PyTorch with CUDA
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# 6. Install remaining dependencies
pip install -r requirements-experimental.txt

# 7. Verify GPU access
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### 3.3 Running GPT-2 Experiments (Single GPU)

**Interactive (for debugging):**
```bash
salloc --nodes=1 --gres=gpu:1 --constraint=gpu80 --mem=30G --time=02:00:00
conda activate edgeprune
cd /scratch/gpfs/<netid>/Edge-Pruning

# Standard (uncompressed)
bash run_scripts/ioi_sweep.sh

# Compressed (k=64)
bash run_scripts/ioi_sweep_compressed.sh
```

**Batch submission:**
```bash
#!/bin/bash -l
#SBATCH --job-name=ioi-compressed
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --mem=30G
#SBATCH --time=10:00:00
#SBATCH --output=./joblog/%x-%j.out

module load anaconda3/2024.2
conda activate edgeprune
cd /scratch/gpfs/<netid>/Edge-Pruning

bash run_scripts/ioi_sweep_compressed.sh
```

Save as `submit_ioi_compressed.sh` and run:
```bash
mkdir -p joblog
sbatch submit_ioi_compressed.sh
```

### 3.4 Running Tracr Experiments (Single GPU)

```bash
salloc --nodes=1 --gres=gpu:1 --mem=16G --time=01:00:00
conda activate edgeprune
bash run_scripts/tracr_reverse_compressed.sh
```

### 3.5 Running CodeLlama Experiments (Multi-Node)

The CodeLlama experiments require 4 nodes with 8 GPUs each (32 GPUs total) using FSDP.

```bash
# Submit directly — the script already has SBATCH directives
cd /scratch/gpfs/<netid>/Edge-Pruning
mkdir -p joblog

# Standard
sbatch run_scripts/launch_fllama_instr_prune.sh

# Compressed (k=256)
sbatch run_scripts/launch_fllama_instr_prune_compressed.sh
```

Monitor:
```bash
squeue -u <netid>              # Check job status
tail -f joblog/instr_prune-*   # Watch logs
```

### 3.6 Running the Test Suite

```bash
salloc --nodes=1 --gres=gpu:1 --constraint=gpu80 --mem=30G --time=00:30:00
conda activate edgeprune
cd /scratch/gpfs/<netid>/Edge-Pruning

# Run all tests
pytest test_edge_pruning_compression.py -v -s

# Run only the fast gradient tests
pytest test_edge_pruning_compression.py -v -s -k "TestGradientFlow"

# Run only the VRAM comparison
pytest test_edge_pruning_compression.py -v -s -k "TestApplestoApples"
```

### 3.7 Evaluating a Pruned Circuit

After training completes:
```bash
# Extract circuit edges
python src/modeling/vis_fpt2.py -i ./data/runs/<run_name>/ -w

# Draw the circuit
python src/modeling/draw_fpt2.py -i ./data/runs/<run_name>/edges.json

# Evaluate
python src/eval/ioi.py -m ./data/runs/<run_name>/ -w
```

---

## 4. Deployment on Google Colab Pro

### 4.1 What You Can Run on Colab

| Experiment | Colab Free (T4 16GB) | Colab Pro (A100 40GB) | Colab Pro+ (A100 40GB) |
|-----------|---------------------|----------------------|----------------------|
| GPT-2 Standard | Tight (batch<=16) | Yes | Yes |
| GPT-2 Compressed (k=64) | Yes (batch=32) | Yes | Yes |
| Tracr | Yes | Yes | Yes |
| CodeLlama 13B | No | No | No |
| Test Suite (Exp 1, 3, 4) | Partial | Yes | Yes |
| Test Suite (Exp 2 stress) | Yes | Yes | Yes |

CodeLlama 13B requires 32 GPUs and cannot run on Colab.

### 4.2 Colab Setup Notebook

Create a new Colab notebook and run the following cells:

**Cell 1 — Clone and install:**
```python
# Select GPU runtime: Runtime > Change runtime type > T4 or A100

!git clone <repo-url> Edge-Pruning
%cd Edge-Pruning

# Extract data
!unzip -q data.zip

# Install dependencies (skip JAX/TF to save space — only needed for Tracr)
!pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
!pip install -q transformers==4.54.0 accelerate datasets einops scipy tqdm
!pip install -q flash-attn --no-build-isolation

# For Tracr experiments, also install:
# !pip install -q jax[cuda12]==0.5.2 dm-haiku==0.0.13 tensorflow==2.18.0
```

**Cell 2 — Verify GPU:**
```python
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

**Cell 3 — Run a single GPT-2 compressed experiment:**
```python
# Run one sparsity level (faster than the full sweep)
!WANDB_MODE=disabled python src/prune/fpt2_ioi_compressed.py \
    --report_to wandb \
    --do_train \
    --do_eval \
    --dataset_path ./data/datasets/ioi/ \
    --train_split train \
    --initialize_from gpt2 \
    --max_seq_length 64 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --eval_accumulation_steps 16 \
    --edge_learning_rate 0.8 \
    --layer_learning_rate 0.8 \
    --reg_edge_learning_rate 0.8 \
    --reg_layer_learning_rate 0.8 \
    --max_steps 3000 \
    --warmup_steps 200 \
    --eval_strategy steps \
    --eval_steps 64 \
    --save_steps 64 \
    --logging_steps 8 \
    --save_total_limit 1 \
    --start_edge_sparsity 0.00 \
    --target_edge_sparsity 0.97 \
    --start_layer_sparsity 0.00 \
    --target_layer_sparsity 0.72 \
    --num_sparsity_warmup_steps 2500 \
    --max_train_samples 1000000 \
    --max_eval_samples 200 \
    --output_dir ./data/runs/ioi-compressed-k64-es0.97/ \
    --remove_unused_columns false \
    --dataloader_num_workers 0 \
    --warmup_type linear \
    --with_embedding_nodes \
    --compression_dim 64 \
    --compression_init gaussian \
    --freeze_projections \
    --disable_node_loss
```

**Cell 4 — Run the test suite:**
```python
!pip install -q pytest
!pytest test_edge_pruning_compression.py -v -s -k "not TestMemoryWall"
# Note: TestMemoryWall intentionally OOMs the GPU, which can crash the Colab
# kernel. Run it separately if desired:
# !pytest test_edge_pruning_compression.py -v -s -k "TestMemoryWall"
```

**Cell 5 — Visualize the circuit:**
```python
!python src/modeling/vis_fpt2.py -i ./data/runs/ioi-compressed-k64-es0.97/ -w
!python src/modeling/draw_fpt2.py -i ./data/runs/ioi-compressed-k64-es0.97/edges.json

# Display in notebook
from IPython.display import display, Image
# If graphviz rendered a PNG:
# display(Image(filename="./data/runs/ioi-compressed-k64-es0.97/edges.png"))
```

**Cell 6 — Evaluate the circuit:**
```python
!python src/eval/ioi.py -m ./data/runs/ioi-compressed-k64-es0.97/ -w
```

### 4.3 Colab Tips

- **Runtime disconnects:** Colab kills idle sessions after ~30–90 min. Keep a cell running or use browser extensions to stay alive. GPT-2 experiments take ~30 min per sparsity level, so they fit within a single session.
- **Disk space:** Colab provides ~100 GB. The repo + data + one run uses ~3 GB. Multiple sweep runs can fill disk — delete old checkpoints with `!rm -rf data/runs/old-run/`.
- **Batch size on T4:** With 16 GB VRAM, use `--per_device_train_batch_size 16` for standard pruning. The compressed version can handle `32`.
- **Saving results:** Mount Google Drive to persist outputs:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  !cp -r ./data/runs/ioi-compressed-k64-es0.97/ /content/drive/MyDrive/edge-pruning-results/
  ```

### 4.4 Comparing Standard vs Compressed on Colab

Run both variants and compare:
```python
# Standard
!WANDB_MODE=disabled python src/prune/fpt2_ioi.py \
    --output_dir ./data/runs/ioi-standard-es0.97/ \
    # ... (same args as Cell 3 but without compression flags, using fpt2_ioi.py)

# Compressed
!WANDB_MODE=disabled python src/prune/fpt2_ioi_compressed.py \
    --output_dir ./data/runs/ioi-compressed-k64-es0.97/ \
    # ... (same args as Cell 3)

# Compare
!python src/eval/ioi.py -m ./data/runs/ioi-standard-es0.97/ -w
!python src/eval/ioi.py -m ./data/runs/ioi-compressed-k64-es0.97/ -w
```
