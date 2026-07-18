# pixelflow_na — 研究路线图

> 2026-07-18 更新 · Pivot: post-hoc ERF 分析，零训练验证 FM 下 attention 局部性

---

## 🎯 核心判断（修正后）

**研究问题不需要训练任何模型。** ΔConvFusion (ICCV 2025) 的 ERF 分析方法是纯 post-hoc 的——加载已训好的模型，跑推理抓 attention map，DFT+高通滤波+ASM 算 ERF。一步训练都没有。

他们分析的是 **DDPM 模型**（SD1.5, SDXL, PixArt）。**没有任何人在 Flow Matching 模型上做过同样的分析。**

我们的贡献：**把 ΔConvFusion 的 ERF 测量方法搬到 Flow Matching 模型上**，回答"FM 下 self-attention 也是局部的吗？"

```
ΔConvFusion 方法          同样的方法                     发现相同/不同
在 DDPM 上证明了：    →   搬到 Flow Matching 上    →   解释为什么
self-attention 是局部的    重新测一遍                  （直线路径→更局部的注意力？）
```

---

## 📐 新路线总览

```
Phase 1: FM 模型 post-hoc ERF 分析（1-3 天，零训练成本）
  ├─ 实现 ΔConvFusion 原版 ERF 方法（DFT + Butterworth 高通 + ASM 80%）
  ├─ SiT-XL/2 full attention ERF 测量
  ├─ 与文献 DDPM ERF 值对比（PixArt <15, SD1.5 <20）
  └─ 产出：FM vs DDPM attention 局部性对比

         ↓  决策点：FM attention 比 DDPM 更局部？

  YES ──→ Story: "FM 直线路径不需要全局 attention，局部性更强"
            → 短文/workshop，结论+数据直接发表
  NO  ──→ Story: "FM 和 DDPM 的 attention 局部性一致"
            → 同样有发表价值（第一个在 FM 上验证的），但贡献稍薄
            → 考虑多模型验证增强说服力

         ↓

Phase 2: 跨模型验证（Phase 1 有结论后，零训练成本）
  ├─ Flux/SD3 等 FM 模型的 ERF hook 测量
  ├─ 验证 Phase 1 结论的跨模型泛化性
  └─ 完整论文 → ECCV/ICCV 短文

         ↓

Phase 3: 如果发现差异——深入分析
  ├─ 为什么 FM 和 DDPM attention 不同？
  ├─ ODE 直线路径 vs SDE 弯曲路径对 attention 的影响
  └─ 理论分析 + 实验验证
```

---

## Phase 1 详细：ΔConvFusion 方法复现 + SiT ERF 测量

### ΔConvFusion ERF 测量方法（Section 3.2）

```
Step 1: 提取 Attention Map
  对第 l 层 self-attention，取 softmax 后的 attention weights
  A_l(x_i, y_j) ∈ R^{H'×W'}（每个 query 位置一个 2D attention map）

Step 2: DFT → 高通 Butterworth 滤波 → IDFT
  对每个 attention map 做 2D DFT
  用高通 Butterworth 滤波器抑制低频分量
  逆变换得到高频 attention map Λ_l

Step 3: 计算 ASM（Attention Score Mass）
  在高频 attention map Λ_l 上，以 query 为中心、边长 K 的方形窗口内累加：
  ASM_l(x_i, y_j) = Σ_{(x_m, y_n) ∈ d∞ < K/2} Λ_l(x_i, y_j)(x_m, y_n)

Step 4: 确定 ERF（80% 阈值）
  k̂_l = min{ k ∈ {0,...,K} | ASM_Λ^l ≥ 0.8 }
  即：高频 ASM 占比 ≥ 80% 的最小窗口大小
```

### 参考实现

- **scikit-image Butterworth 滤波器**：`skimage/filters/_fft_based.py` 中 `_get_nd_butterworth_filter` — NumPy 实现，直接翻译成 PyTorch
- **我们已有的资源**：`SiT/models.py` 的 `get_attention_weights()` 已经能提取 attention map

### 测量对象

| 模型 | 训练框架 | Token 数 | 来源 |
|------|---------|----------|------|
| **SiT-XL/2** | Flow Matching | 256 (16×16) | 已有 pretrained weights |
| Flux/SD3 (可选) | Flow Matching | 待定 | 需下载 |

### 对比基线（文献值）

| 模型 | 训练框架 | 报告 ERF | 来源 |
|------|---------|----------|------|
| PixArt (DiT) | DDPM | < 15×15 | ΔConvFusion §3.2 |
| SD1.5 (U-Net) | DDPM | < 20×20 | ΔConvFusion §3.2 |

### 输出

```
results/
├── sit_erf_deltaconv/              # ΔConvFusion 方法测量
│   ├── sit_erf_full.json           # 完整数据
│   ├── erf_per_layer.png           # 每层 ERF
│   ├── asm_vs_k.png                # ASM 随 K 增长曲线
│   └── comparison_ddpm_fm.png      # FM vs DDPM 对比
```

---

## 🔀 决策点（修正后）

```
Phase 1 数据出来后：

FM attention ERF 明显小于 DDPM（差 ≥ 30%）
  → Story: "FM 直线路径下 self-attention 更局部"
  → 不需要 NA——convolution 就够覆盖有效感受野
  → 短文即可发表

FM attention ERF 与 DDPM 相近（差 < 30%）
  → Story: "FM 和 DDPM 下 attention 局部性一致"
  → 第一手 FM 验证数据仍有发表价值
  → 补跨模型验证 + 理论解释增强说服力

FM attention ERF 明显大于 DDPM
  → Story: "FM 直线路径反而需要更大感受野"
  → 最有意思的发现！追下去
```

---

## ⚠️ 已废弃的思路

| 思路 | 为什么废弃 |
|------|-----------|
| **t-adaptive kernel** | 数据已证伪：per-t ERF 变化 < 8%（full）/< 0.1%（NA） |
| **NA fine-tune 5 variants** | 研究问题不需要训练模型——post-hoc 分析就能回答 |
| **pixel space 从头训 DiT** | 成本高、结论弱、pretrained FM 模型直接用 |
| **Procrustes latent→pixel 迁移** | 同上，不需要 |

**保留价值**：`models/` 和 `SiT/` 的 attention/measure 代码作为工具库，Phase 3 或后续需要训练时再用。

---

## 📁 代码复用计划

| 模块 | 用途 | 状态 |
|------|------|------|
| `SiT/models.py` `get_attention_weights()` | 提取 attention map | ✅ 已有 |
| `SiT/checkpoints/SiT-XL-2-256.pt` | 预训练 FM 模型 | ✅ 已下载 |
| `measure_sit.py` | 需改写为 ΔConvFusion 原版方法 | 🔄 待修改 |
| `finetune_sit_na.py` | 不再需要 | ❌ 废弃 |
| `precompute_latents.py` | 不再需要 | ❌ 废弃 |
| `analyze.py` | 可视化，可复用 | ✅ 保留 |

---

## ⏱️ 时间线（修正后）

| 阶段 | 内容 | 预计 |
|------|------|------|
| Day 1 | 实现 ΔConvFusion DFT+Butterworth+ASM 方法 | 半天 |
| Day 1 | SiT-XL/2 ERF 测量 + 可视化 | 半天 |
| Day 2 | 与文献 DDPM ERF 对比 + 分析 | 半天 |
| Day 2-3 | 根据结果决定：多模型验证 or 深入分析 | 1-2 天 |

---

*Updated 2026-07-18 — pivot 到 ΔConvFusion post-hoc ERF 分析，废弃所有训练相关路线*
