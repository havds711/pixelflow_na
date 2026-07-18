# Attention Locality in Flow Matching Transformers — 分析 & 后续计划

> 数据: `outputs/attention_locality_sit_xl/attention_locality_sit_xl.pt` (SiT), `outputs/attention_locality_pixeldit/attention_locality_pixeldit.pt` (PixDiT)
> 实验脚本: `measure/measure_locality_sit.py`, `measure/measure_locality_pixeldit.py`
> 分析脚本: `analysis/analyze_sit.py`, `analysis/analyze_pixeldit.py`
> 日期: 2026-07-18

---

## 1. 实验设置

| 参数 | SiT-XL/2 | PixDiT |
|---|---|---|
| 参数量 | 675M | ~675M |
| 层数 | 28 | 30 (26 patch + 4 pixel) |
| 头数 | 16 | 16 |
| Tokens | 256 (16×16) | 256 (16×16) |
| 推理 | ODE Euler, 20步, t=1→0 | ODE Euler, 25步, t=0→1 |
| 输入 | 4张 ImageNet | 4张 ImageNet |
| k values | [1,3,5,7,9,11,13,15] | [1,3,5,7,9,11,13,15] |
| 总测量点 | 9,175,040 | 12,288,000 |

---

## 2. SiT-XL/2 核心发现

### 2.1 Attention Locality 极弱

**k=15 才能让 93.7% 的 query 达到 >80% attention mass。**

| k | 累计mass均值 | >50% | >80% | >90% | >95% | >99% |
|---|---|---|---|---|---|---|
| k=1 | 0.018 | 0.4% | 0.2% | 0.1% | 0.1% | 0.02% |
| k=3 | 0.170 | 11.3% | 8.7% | 7.6% | 6.8% | 5.3% |
| k=5 | 0.263 | 15.9% | 10.7% | 9.3% | 8.3% | 6.2% |
| k=7 | 0.368 | 22.9% | 13.0% | 10.7% | 9.2% | 6.7% |
| k=9 | 0.484 | 34.7% | 16.0% | 12.5% | 10.4% | 7.3% |
| k=11 | 0.610 | 64.0% | 20.6% | 15.0% | 12.2% | 8.4% |
| k=13 | 0.750 | 95.9% | 34.5% | 20.1% | 15.2% | 10.2% |
| **k=15** | **0.908** | **99.6%** | **93.7%** | **59.1%** | **30.8%** | **14.3%** |

### 2.2 Reverse Locality: 浅层更全局, 深层更局部

| 深度分组 | >80%@k=7 |
|---|---|
| 浅层 L0-L8 | 3.6% |
| 中层 L9-L18 | 13.0% |
| 深层 L19-L27 | 17.9% |

与 CNN "浅层局部→深层全局" 完全相反。假设: SiT 浅层做全局语义对齐, 深层做局部细节精炼。

### 2.3 Head 多样性: 同层内差异可达 5×

L13: H0/H2/H4/H9/H10/H14 的 mean min_k≈4, 其余10个head=15。只有 **11% (50/448)** 的 head mean min_k ≤ 7。

### 2.4 ODE 时间步几乎无影响

Early/mid/late 的 >80%@k=7 分别为 13.3%/12.9%/13.0%, std=0.0035。

### 2.5 空间位置 / 输入图像几乎无影响

Corner/edge/interior 差异 < 0.2; 4图 max-min spread = 0.0038。

### 2.6 Per-Head NA 结论: ❌ 没搞头

| 策略 | FLOPs节省 |
|---|---|
| 统一 k=15 NA | 13% |
| Per-head 覆盖80% query | 25% |
| Per-head 覆盖50% query | 29% |

---

## 3. PixDiT 核心发现

### 3.1 Attention 显著更局部 🔥

**PixDiT 在每个 k 值上 attention mass 都比 SiT 高得多：**

| k | SiT >80% | PixDiT >80% | 差距 |
|---|---|---|---|
| k=1 | 0.2% | **3.2%** | 16× |
| k=3 | 8.7% | **11.9%** | +37% |
| k=5 | 10.7% | **18.8%** | +76% |
| k=7 | 13.0% | **24.7%** | **+90%** |
| k=9 | 16.0% | **31.9%** | +99% |
| k=11 | 20.6% | **41.5%** | **+101%** |
| k=13 | 34.5% | **57.4%** | +66% |
| k=15 | 93.7% | 92.1% | -1.7% |

**k=1 单点 attention mass 均值: SiT 1.8% vs PixDiT 9.1% — 5倍差距。**

**猜测原因**: PixDiT 的 pixel stream 负责高频局部细节, patch-stream attention 不需要覆盖所有像素, 可以更聚焦。dual-stream 设计自然产生了 attention 分工。

### 3.2 没有 Reverse Locality

| 深度 | SiT >80%@k=7 | PixDiT >80%@k=7 |
|---|---|---|
| 浅层 | 3.6% | **20.1%** |
| 中层 | 13.0% | **29.8%** |
| 深层 | 17.9% | **27.6%** |

PixDiT: 中层最局部, 浅层次之。与 SiT 完全不同的模式。

### 3.3 Pixel Blocks 两极分化

| Pixel Layer | >80%@k=7 | 特征 |
|---|---|---|
| L26 | 18.2% | 与 patch blocks 类似 |
| **L27** | **2.1%** | **极其全局 — 94.4% query 需要 k=15** |
| L28 | 34.0% | 非常局部 |
| L29 | 16.9% | 中等 |

### 3.4 ODE 步间趋势: 随时间变得更局部

- 早期 t≈0: >80%@k=7 = 15.8%
- 晚期 t≈0.96: >80%@k=7 = **29.0%**

SiT 没有这个趋势。PixDiT 的 attention 随去噪进行逐步聚焦局部。

### 3.5 图像间有显著差异

| Image | >80%@k=7 |
|---|---|
| golden_retriever | 0.272 |
| otter | 0.260 |
| lesser_panda | 0.237 |
| geyser | 0.219 |

max-min spread = **0.053** (SiT 仅 0.004, 差 14 倍)。PixDiT 的 locality 对输入内容敏感。

### 3.6 NA-Friendly Head 更多: 19% vs 11%

PixDiT: **91/480 (19.0%)** head mean min_k ≤ 7。分布更均匀, 不集中少数层。

Per-head NA 宽松策略 (50% query): k_avg=11.3, FLOPs节省 **39%** (SiT: 29%)。

---

## 4. SiT vs PixDiT 总结对比

| 属性 | SiT-XL/2 | PixDiT |
|---|---|---|
| Attention Locality | 极弱 (全局性) | 中等 (比SiT局部~2×) |
| k=1 mean mass | 1.8% | 9.1% |
| 深度模式 | Reverse (浅全局→深局部) | 中层最局部, 无reverse |
| ODE时间依赖 | 几乎无 | 明显: 晚期更局部 |
| 图像依赖 | 几乎无 | 有显著差异 (spread 5.3pp) |
| NA-friendly heads | 11% | 19% |
| Per-head NA 最优 FLOPs节省 | 29% | 39% |

**关键结论: Attention Locality 不是 FM 模型的通用属性 — 它高度依赖架构设计。** PixDiT 的 dual-stream (patch + pixel) 设计使其 attention 系统性地更局部且行为更丰富。这验证了多模型跑同一流程的价值。

---

## 5. 已验证 / 待验证假设

| # | 假设 | SiT | PixDiT | 状态 |
|---|---|---|---|---|
| 1 | FM模型attention偏向全局 | ✅ 极全局 | ⚠️ 中等局部 | **推翻: 架构依赖** |
| 2 | Reverse Locality 跨模型一致 | ✅ reverse | ❌ 无reverse | **推翻** |
| 3 | Head多样性5×普遍存在 | ✅ 存在 | ✅ 存在 | **确认** |
| 4 | 规模效应 (更大→更局部?) | - | - | 待测 (SiT-B/L) |
| 5 | ODE轨迹上locality稳定 | ✅ 稳定 | ❌ 不稳定 | **推翻** |
| 6 | 内容独立性 | ✅ 独立 | ❌ 依赖输入 | **推翻** |

---

## 6. Per-Head NA 可行性 (更新)

在 pretrained 模型上直接替换仍然不够 (PixDiT 最多省 39% FLOPs)，但：
- **从零训练 Per-Head Adaptive NA 的论据增强了** — PixDiT 证明不同架构可以学到更局部的 attention
- **PixDiT 的 dual-stream 可以作为 NA 训练的参考设计** — pixel stream + local patch attention
- **可学习窗口大小的思路得到支持** — attention locality 可以被训练塑造

---

## 7. 后续计划: 多模型验证

**核心问题升级: Attention Locality 在 FM 模型中的变化规律是什么? 与架构/训练方式/规模的关系?**

### 7.1 候选模型 & 状态

| 模型 | 状态 | 架构 | 参数量 | 优先级 |
|---|---|---|---|---|
| **SiT-XL/2** | ✅ done | DiT + FM | 675M | — |
| **PixDiT** | ✅ done | Dual-stream DiT + FM | ~675M | — |
| DiT-XL/2 | 待测 | DiT + DDPM | 675M | 高 — 对比FM vs DDPM |
| SiT-B/2 | 待测 | DiT + FM | 130M | 高 — 规模效应 |
| SiT-L/2 | 待测 | DiT + FM | 450M | 中 |
| MDTv2-XL/2 | 待测 | Masked DiT + FM | ~675M | 中 — 不同attention变体 |
| MMDiT / SD3 | 待测 | MMDiT + FM | 2B-8B | 中 — 文本条件+双流 |
| FlowAR / MAR | 待测 | AR + FM | ? | 低 — 范式差异大 |

### 7.2 每个模型的测量要点

1. **DiT-XL/2**: 是否 DDPM 训练让 attention 更局部/全局? FM vs DDPM 的关键对比
2. **SiT-B/L**: 规模是否系统性地改变 locality?
3. **MDTv2**: Masked attention 训练是否让 unmasked attention 更局部?
4. **MMDiT/SD3**: 双流 (text+image) 的 attention 分工? prompt 依赖性?
5. **FlowAR**: 自回归 + FM 混合范式的 locality 特征?

---

## 8. 环境规划

按代码源分环境，同源模型复用：

```
SiT 系列 (SiT-B/L/XL)    →  conda env: natten       (已有 environment.yml)
DiT 系列 (DiT-B/L/XL)    →  conda env: dit       (Meta 原版, 依赖类似但独立)
PixDiT                   →  conda env: pixel     (已配置)
MDTv2                    →  conda env: mdt       (独立 repo)
MMDiT / SD3              →  conda env: mmdit     (diffusers >= 0.31, transformers 新)
FlowAR / MAR             →  conda env: flowar    (完全不同范式)
```

**为什么必须隔离:**
- `diffusers` 版本差异最大 — SD3 需要 ≥0.31，SiT 用的是旧版
- `transformers` — SD3 依赖新版，别的可能不兼容
- `torch` — 不同 model checkpoint 对不同 PyTorch 版本敏感（尤其是 torch.compile）
- CUDA 版本 — 有些模型绑定了特定 CUDA toolkit

**实际做法:**
```bash
# SiT (已有 natten 环境, 不用动)
# conda env create -f SiT/environment.yml  # 已装好

# 其他的按这个模板
conda create -n dit python=3.10 -y
conda activate dit
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
git clone <DiT repo> && cd DiT && pip install -e .
```

**注意**: 当前网络代理不通, 建议:
1. 先搞定网络/代理配置
2. 或者在联网机器上用 `pip download` 下好 wheel, 拷过来离线安装

---

## 9. 开放问题

- [ ] DiT (DDPM) 的 locality 和 SiT/PixDiT (FM) 有没有本质区别?
- [ ] PixDiT 的 image-dependence — 不同内容导致 5pp 差异的原因?
- [ ] 文本条件 (SD3) 的 attention locality 会因 prompt 变化吗?
- [ ] 更大的 token 数 (如 32×32=1024) 下 locality 怎么变化?
- [ ] PixDiT pixel block L27 为什么几乎是纯全局 attention?
- [ ] Fine-tuned model (如经过 NA fine-tune) 的 locality 会变化吗?
- [ ] 为什么 PixDiT 比 SiT 局部 2×, 具体是哪个设计差异导致的?
- [ ] 从零训练 NA 后, attention locality 会如何重新分布?
