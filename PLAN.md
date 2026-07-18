# pixelflow_na — 实现计划

> 2026-07-18 更新 · Pivot: pixel space 从零训 → latent space SiT pretrained

---

## 硬件情况

| GPU | 型号 | 显存 | 当前空闲 | 用途 |
|-----|------|------|----------|------|
| 0 | RTX 3080 Ti | 12 GB | **全空闲** | 备用 / 小实验 |
| 1 | RTX 3080 Ti | 12 GB | **全空闲** | 备用 / 小实验 |
| 2 | RTX 3090 | 24 GB | 18 GB | **主力训练卡** |
| 3 | RTX 4080 | 16 GB | 13 GB | SiT-B/2 训练 / 测量 |

- RAM: 125 GB total, 95 GB 可用
- CPU: Xeon Silver 4210, 40 cores
- 环境: conda `natten`, Python 3.10, PyTorch 2.3+cu121, NATTEN 0.17.4

### 显存估算

| 模型 | Params | 推理 VRAM | 训练 VRAM (bs=4) |
|------|--------|-----------|-------------------|
| SiT-XL/2 | 675M | ~10 GB (bf16) | ~18 GB |
| SiT-B/2 | 130M | ~3 GB | ~8 GB |

---

## Step 0: 环境准备（5 分钟）

```bash
pip install timm diffusers  # SiT 依赖
```

验证：
```bash
python3 -c "from diffusers.models import AutoencoderKL; vae=AutoencoderKL.from_pretrained('vae/'); print('VAE OK')"
python3 -c "from SiT.models import SiT_XL_2; print('SiT OK')"
```

---

## Step 1: `measure_sit.py` — pretrained SiT 零训练 ERF 测量

**目标**：用 SiT-XL/2 pretrained 直接测 per-t ERF。零训练成本，这是决定整篇文章走向的关键数据。

### 为什么先做这一步

ROADMAP.md 的核心假设：Flow Matching 下不同 t 需要不同 receptive field → 如果 per-t ERF 差异 ≥ 2× → story 牢靠。如果差异小 → 需要重新评估方向。这张图出来之前，不做任何训练。

### 实现

**新建文件**：`pixelflow_na/measure_sit.py`

- 加载 `SiT/models.py` 的 `SiT_XL_2()`，load `SiT/checkpoints/SiT-XL-2-256.pt`
- 使用 `SiT.get_attention_weights()` 提取 attention（已实现）
- 复用 `measure.py` 的 ERF/distance 计算逻辑
- SiT latent: 32×32 grid, patch=2 → 256 tokens → full attention 测量极快
- 输入数据：随机噪声 latent（`randn(B,4,32,32)`），ERF 测量不依赖真实图像
- **t 采样点**：t = 0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0（7 个点）

### 测量指标

| 指标 | 方法来源 | 输出 |
|------|---------|------|
| Per-t ERF | ΔConvFusion §3 | ERF 值随 t 变化曲线 |
| Per-layer ERF | ΔConvFusion §3 | 每层 ERF |
| Per-t Distance | PiT §3 | mean/P99 distance 随 t 变化 |
| Distance cumulative | PiT §3 | P(d<k) 累计分布 |

### 输出

```
outputs/sit_measure/
├── sit_erf_full.json
├── erf_vs_t.png          # 🔑 关键：ERF 随 t 变吗？
├── erf_per_layer.png
├── distance_vs_t.png
└── distance_cumulative.png
```

### 运行

```bash
python measure_sit.py --device cuda:0 --n_samples 32
# 预期：<5 分钟完成
```

---

## Step 2: `finetune_sit_na.py` — NA Fine-tune + 验证图保存

### 策略

先用 SiT-B/2 (130M) on GPU 3 (4080) 验证 pipeline 能跑通（快 4x），然后切 SiT-XL/2 on GPU 2 (3090) 正式跑。

### 关键功能

1. **VAE 预提取 latents**：一次性把 160K ImageNet 图像 encode 成 latent（~1.3GB fp16），训练时直接加载
2. **NA 替换**：加载 pretrained token → 替换所有 block.attn 为 NA → fine-tune
3. **每 500 steps 保存验证图**：固定 8 个 class labels，生成 sample + VAE decode → 保存 PNG
4. **全参数 fine-tune**，LR=1e-5（比从头训低 10x），EMA decay=0.9999

### 训练配置

| 参数 | SiT-B/2 | SiT-XL/2 |
|------|---------|----------|
| GPU | 4080 (GPU 3) | 3090 (GPU 2) |
| Batch size | 4-8 | 1-2 |
| LR | 1e-5 | 1e-5 |
| Fine-tune steps | ~5000 | ~5000 |
| Sample every | 500 steps | 500 steps |

### 验证样本

每 500 steps 生成 8 张样本图（固定 class labels: 207, 360, 387, 974, 88, 979, 417, 279）：

```
outputs/sit_finetune_na{k}/
├── samples/
│   ├── step_0500.png
│   ├── step_1000.png
│   └── ...
├── checkpoint.pt
└── measure_results.json
```

### 运行

```bash
# 先预提取 latents
python precompute_latents.py --device cuda:0

# 快速验证 pipeline（500 steps）
python finetune_sit_na.py --kernel_size 7 --max_steps 500 --device cuda:3

# 正式 fine-tune（5000 steps）
python finetune_sit_na.py --kernel_size 7 --max_steps 5000 --device cuda:2
```

---

## Step 3: `sweep_sit.py` — 批量 NA Sweep

对 NA k = 3, 5, 7, 11, 15 批量 fine-tune + 自动测量。

### 实验矩阵

| Variant | Attention | Kernel | GPU | 测量 |
|---------|-----------|--------|-----|------|
| baseline | full | — | — | Step 1 已测 |
| na3 | NA | 3 | 3090 | ERF + Dist + FID + GFLOPs |
| na5 | NA | 5 | 3090 | ↑ |
| na7 | NA | 7 | 3090 | ↑ |
| na11 | NA | 11 | 3090 | ↑ |
| na15 | NA | 15 | 3090 | ↑ |

### 输出

`outputs/sit_sweep_results.json` — 兼容 `analyze.py` 直接可视化。

---

## Step 4: 可选 — 跨模型 post-hoc 验证

对开源 Flow Matching 模型（如 SD3/Flux DiT backbone）做 hook-based ERF 测量，零训练成本提供跨模型 validation。

---

## 文件变更

| 文件 | 操作 | 说明 |
|------|------|------|
| `PLAN.md` | 更新 | 本文件 |
| `measure_sit.py` | **新建** | SiT ERF/distance 测量入口 |
| `finetune_sit_na.py` | **新建** | SiT NA fine-tune + 验证图保存 |
| `sweep_sit.py` | **新建** | SiT NA sweep 自动化 |
| `precompute_latents.py` | **新建** | 一次性预提取 ImageNet latents |
| `measure.py` | 不改 | 提取公共函数到 module 级别供复用 |
| `SiT/` | 不改 | 官方代码，直接 import |
| `analyze.py` | 不改 | JSON 格式兼容 |

---

## 验证

| 阶段 | 命令 | 预期 |
|------|------|------|
| Step 1 | `python measure_sit.py --device cuda:0` | <5min, 产出 erf_vs_t.png |
| Step 2 | `python finetune_sit_na.py -k 7 --max_steps 500 --device cuda:3` | loss 下降, sample 图正常 |
| Step 3 | `python sweep_sit.py --kernels na7 na15 --device cuda:2` | FID + ERF 数据一致 |

---

*Updated 2026-07-18 — pivot 到 SiT latent space，具体 GPU 分配 + 验证图策略*
