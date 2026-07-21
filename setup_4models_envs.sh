#!/bin/bash
# ============================================================================
# setup_4models_envs.sh — 为 PixArt 和 SD 模型创建 conda 环境
#
# 创建:
#   - pixart: PixArt-α / PixArt-Σ (DiT + cross-attention)
#   - sd15:   Stable Diffusion 1.5 / SD XL (UNet + cross-attention)
#
# 用法:
#   bash setup_4models_envs.sh
# ============================================================================
set -e

echo "============================================"
echo "Setting up environments for 4 new models"
echo "============================================"

# --- pixart env ---
echo ""
echo "[1/2] Creating 'pixart' environment..."
conda create -n pixart python=3.10 -y
echo "  Installing PyTorch + dependencies..."
conda run -n pixart pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
conda run -n pixart pip install diffusers transformers accelerate timm einops numpy tqdm
echo "  pixart env done."

# --- sd15 env ---
echo ""
echo "[2/2] Creating 'sd15' environment..."
conda create -n sd15 python=3.10 -y
echo "  Installing PyTorch + dependencies..."
conda run -n sd15 pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
conda run -n sd15 pip install diffusers transformers accelerate numpy tqdm
echo "  sd15 env done."

echo ""
echo "============================================"
echo "All environments created successfully!"
echo ""
echo "Verify:"
echo "  conda activate pixart && python -c 'import torch; print(torch.__version__)'"
echo "  conda activate sd15   && python -c 'import torch; print(torch.__version__)'"
echo "============================================"
