# 📖 BitMC-SSM End-to-End Training & Deployment Guide

This guide provides complete instructions for training, scaling, checkpointing, and deploying **BitMC-SSM** models across different hardware setups — from free Kaggle/Colab GPUs to multi-GPU enterprise clusters.

---

## 📐 1. Model Sizing & Architecture Matrix

BitMC-SSM scales gracefully from lightweight nano-models up to edge-pro models:

| Tier | Parameters | `d_model` | `n_layers` | `d_state` | Target Dataset Tokens | Estimated Dual-T4 Training Time | Capability Level |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Nano** | **~30M - 50M** | 384 | 8 | 32 | 25M tokens | **~25 - 35 mins** | Grammar, simple Q&A, pattern completion |
| **Micro** | **~60M - 70M** | 576 | 16 | 48 | 100M tokens | **~1.2 - 1.5 hours** | Natural English/Japanese dialog, basic reasoning |
| **Small** *(Recommended)* | **~135M** | 768 | 24 | 64 | 500M tokens | **~9 - 10 hours** | Instruction following, Python scripts, logical flow |
| **Edge-Pro** | **~350M - 500M** | 1024 | 32 | 64 | 1B+ tokens | **~35 - 40 hours** (Multi-Session) | Deep context comprehension, edge deployment |

---

## ⚡ 2. Quickstart Training Workflows

### A. Dual Tesla T4 GPUs (Kaggle / Free Tier)
Kaggle provides 2x Tesla T4 GPUs (16GB VRAM each) with PyTorch 2.0+ and Triton pre-installed:

```bash
# 1. Clone repository & install dependencies
git clone https://github.com/fukayatti/BitMC-SSM.git
cd BitMC-SSM
pip install -q triton transformers datasets

# 2. Launch Dual-GPU DDP Training with Triton acceleration
torchrun --nproc_per_node=2 python/train.py \
    --dataset smollm \
    --dataset_subset cosmopedia-v2 \
    --num_samples 50000 \
    --d_model 384 \
    --n_layers 8 \
    --d_state 32 \
    --batch_size 32 \
    --amp \
    --galore_rank 32 \
    --epochs 8 \
    --save_ckpt_dir /kaggle/working/checkpoints
```

### B. Single GPU (Google Colab / Local RTX GPU)
```bash
python python/train.py \
    --dataset smollm \
    --d_model 384 \
    --n_layers 8 \
    --batch_size 32 \
    --amp \
    --device cuda
```

### C. CPU-Only Debug Mode
```bash
python python/train.py \
    --dataset synthetic \
    --d_model 128 \
    --n_layers 2 \
    --d_state 16 \
    --num_samples 100 \
    --batch_size 4 \
    --device cpu
```

---

## 🔧 3. Key CLI Arguments & Optimization Flags

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--dataset` | `str` | `smollm` | Dataset source (`smollm` or `synthetic`). |
| `--dataset_subset` | `str` | `cosmopedia-v2` | SmolLM subset (`cosmopedia-v2`, `fineweb-edu-dedup`, etc.). |
| `--num_samples` | `int` | `50000` | Total samples streamed from HuggingFace. |
| `--batch_size` | `int` | `32` | Batch size per GPU rank (Effective batch = `batch_size * world_size`). |
| `--amp` | `flag` | `False` | Enables FP16 Automatic Mixed Precision (Tensor Core acceleration). |
| `--galore_rank` | `int` | `32` | GaLore low-rank projection rank for 70% optimizer memory savings. |
| `--grad_accum_steps`| `int` | `1` | Gradient accumulation steps for larger virtual batches. |
| `--save_every_epochs`| `int` | `1` | Checkpoint saving frequency. |
| `--save_ckpt_dir` | `str` | `checkpoints` | Directory to save `.pt` checkpoints. |
| `--resume_from` | `str` | `None` | Path to a checkpoint `.pt` file to resume training from. |

---

## 🔄 4. Checkpoint Resuming & Multi-Session Training

For long training runs across Kaggle/Colab sessions, save intermediate checkpoints and resume seamlessly:

```bash
# Resume training from Epoch 4
torchrun --nproc_per_node=2 python/train.py \
    --dataset smollm \
    --d_model 384 --n_layers 8 \
    --resume_from /kaggle/working/checkpoints/bit_mc_ssm_epoch4.pt \
    --epochs 12 \
    --save_ckpt_dir /kaggle/working/checkpoints
```

---

## 📦 5. Export to C++ 2-Bit Binary & Zero-GEMM Inference

Once training completes, export the model weights to the compact packed 2-bit binary format (`model.bin`):

```python
import torch
from train import BitMCSSM, export_to_binary

# 1. Load trained PyTorch checkpoint
model = BitMCSSM(vocab_size=50257, d_model=384, n_layers=8, d_state=32)
ckpt = torch.load("checkpoints/bit_mc_ssm_epoch8.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)

# 2. Export 1.58-bit packed weights to binary
export_to_binary(model, "model_final.bin")
print("✅ Export complete! Model size:", os.path.getsize("model_final.bin") / (1024 * 1024), "MB")
```

### Run Realtime CPU Inference (100+ tokens/sec):
```bash
# Build native C++20 engine
make

# Run streaming generation
./infer model_final.bin 60
```
