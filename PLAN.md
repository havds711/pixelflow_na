# pixelflow_na — 实现计划

> 2026-07-18 更新 · Pivot: post-hoc ERF 分析，零训练

---

## 硬件情况

| GPU | 型号 | 显存 | 当前空闲 | 用途 |
|-----|------|------|----------|------|
| 0 | RTX 3080 Ti | 12 GB | **全空闲** | ERF 测量 |
| 1 | RTX 3080 Ti | 12 GB | **全空闲** | ERF 测量 |
| 2 | RTX 3090 | 24 GB | 18 GB | 大模型推理 |
| 3 | RTX 4080 | 16 GB | 13 GB | 备用 |

- RAM: 125 GB total, 95 GB 可用
- CPU: Xeon Silver 4210, 40 cores
- 环境: conda `natten`, Python 3.10, PyTorch 2.3+cu121

---

## 核心思路（修正后）

**研究问题**：Flow Matching 下 self-attention 是否和 DDPM 下一样局部？

**方法**：ΔConvFusion (ICCV 2025) 的 post-hoc ERF 分析方法，搬到 FM 模型上。**不需要训练任何模型。**

**为什么之前的思路错了**：
- ΔConvFusion 分析的是 DDPM 模型（SD1.5, SDXL, PixArt），没有 FM 模型
- 他们的方法是纯 post-hoc 的：加载权重 → 推理 → 抓 attention map → 算 ERF
- 零训练成本就能回答核心研究问题
- 我们之前却在做 NA fine-tune、t-adaptive kernel——这些是"方法创新"而非"行为验证"

---

## Step 1: 实现 ΔConvFusion 原版 ERF 方法

### 目标

把 ΔConvFusion §3.2 的 ERF 测量公式实现为可复用的 PyTorch 代码。

### 方法流程

```
输入: pretrained SiT-XL/2, 随机 latent x ~ N(0,I), timestep t

1. 提取 attention maps
   attn_list = model.get_attention_weights(x, t, y)
   # 每层产出 [B, heads, N, N] post-softmax attention

2. 对每个 query 位置 (x_i, y_j)，取其 2D attention map
   A = attn[q_idx].reshape(H, W)  # H=W=16 for SiT

3. DFT → 高通 Butterworth → IDFT
   F = torch.fft.fft2(A)
   H = butterworth_highpass(H, W, cutoff, order)  # 高通滤波器
   Lambda = torch.fft.ifft2(F * H).real  # 高频 attention map

4. 计算 ASM(K) 曲线
   for K in range(1, max_k+1):
       asm = sum of Lambda within K×K window around query
       记录 asm / total_asm

5. 确定 ERF = 最小 K 使 ASM 占比 ≥ 80%
   erf = min{K | ASM(K) / ASM_total >= 0.8}
```

### 参考实现

- **Butterworth 高通滤波器**：scikit-image `skimage/filters/_fft_based.py` 的 `_get_nd_butterworth_filter`
  ```python
  # scikit-image 核心逻辑（翻译成 PyTorch）:
  def butterworth_highpass(shape, cutoff, order=2):
      # 构建频率网格
      rows = torch.fft.fftfreq(H)
      cols = torch.fft.fftfreq(W)
      grid = torch.sqrt(rows[:,None]**2 + cols[None,:]**2)
      # Butterworth 高通: H = 1 / (1 + (cutoff/D)**(2*order))
      H = 1.0 / (1.0 + (cutoff / (grid + 1e-8)) ** (2 * order))
      return H  # 高通（低频抑制）
  ```

### 文件

**新建**：`pixelflow_na/erf_deltaconv.py` — ΔConvFusion ERF 方法实现

### 验证

```bash
python erf_deltaconv.py --device cuda:0 --n_samples 16
# 预期：< 10 分钟完成，产出 per-layer ERF + ASM 曲线数据
```

---

## Step 2: SiT-XL/2 ERF 测量 + 与文献对比

### 运行

```bash
python erf_deltaconv.py --model SiT-XL/2 --device cuda:2 --n_samples 32
```

### 输出

```
results/sit_erf_deltaconv/
├── sit_erf_full.json        # 完整数据（per-layer ERF, ASM曲线, per-t）
├── erf_per_layer.png        # 每层 ERF bar chart
├── asm_vs_k.png             # ASM 随 K 增长（关键：二次增长=局部性证据）
├── erf_comparison.png       # SiT(FM) vs PixArt(DDPM) vs SD1.5(DDPM) ERF 对比
└── attention_locality.png   # 高频 ASM 占比可视化
```

### 对比基线

| 模型 | 训练框架 | 架构 | 报告 ERF | 来源 |
|------|---------|------|----------|------|
| SiT-XL/2 | **Flow Matching** | DiT | **待测** | 我们的数据 |
| PixArt | DDPM | DiT | < 15×15 | ΔConvFusion Table/Fig |
| SD1.5 | DDPM | U-Net | < 20×20 | ΔConvFusion Table/Fig |

---

## Step 3: 多模型 post-hoc 验证（Phase 1 有结论后）

对能找到的其他 FM 开源模型做同样分析：

| 模型 | 架构 | 可行性 |
|------|------|--------|
| SiT-B/2, SiT-S/2 | DiT + FM | ✅ 已有 weights |
| Flux.1-dev/schnell | MMDiT + FM | 需适配 MMDiT attention 提取 |
| SD3-medium | MMDiT + FM | 同上 |

**成本**：每个模型加载权重 + 跑几十张图推理 = 几分钟到几十分钟，零训练。

---

## 已废弃的步骤

| 步骤 | 说明 | 废弃原因 |
|------|------|----------|
| ~~Step 2: finetune_sit_na.py~~ | NA fine-tune | 不需要训练 |
| ~~Step 3: sweep_sit.py~~ | NA kernel sweep | 同上 |
| ~~precompute_latents.py~~ | VAE 预编码 | 同上 |
| ~~Procrustes latent→pixel~~ | 空间转换 | 同上 |

`finetune_sit_na.py` 和 `precompute_latents.py` 保留在仓库中但不作为当前计划的一部分。

---

## 文件变更

| 文件 | 操作 | 说明 |
|------|------|------|
| `ROADMAP.md` | 更新 | 新路线：post-hoc ERF 分析 |
| `PLAN.md` | **重写** | 本文件 |
| `erf_deltaconv.py` | **新建** | ΔConvFusion ERF 方法实现 |
| `measure_sit.py` | 保留 | 现有简化版 ERF（备用参考） |
| `finetune_sit_na.py` | 保留不删 | 已废弃，后续可能有用 |
| `precompute_latents.py` | 保留不删 | 同上 |
| `README.md` | 更新 | 修正研究 gap 描述 |
| `BUILD.md` | 更新 | 修正技术选型 |

---

## 验证清单

| # | 命令 | 预期 |
|---|------|------|
| 1 | `python erf_deltaconv.py --device cuda:0 --n_samples 8` | <5min, 产出 per-layer ERF |
| 2 | 对比产出的 ERF 与 ΔConvFusion 报告的 DDPM ERF | 数值在合理范围（~5-15 grid cells） |
| 3 | ASM vs K 曲线呈二次增长 | 确认 attention 局部性 |
| 4 | 高通滤波后 80% ASM 集中在小窗口 | 核心结论支撑 |

---

*Updated 2026-07-18 — 完全重写，废弃训练路线，改为纯 post-hoc ERF 分析*
