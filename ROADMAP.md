# pixelflow_na — 后续研究思路

> 2026-07-17 · 跑完 Phase 1 后可以继续做的方向

---

## 当前

DiT + ImageNet-64 + 6 种 kernel size sweep，per-t ERF 和 distance distribution 分析。

---

## 一条主线：Receptive Field Dynamics in Flow Matching

同一个 research direction 可以滚雪球出多篇论文，每篇在前一篇基础上新增一个核心贡献：

```
论文 1: 发现 ────→ 论文 2: 发现+方法 ────→ 论文 3: 理论+系统 ────→ 论文 4: 期刊长文
```

---

## 一、近期的直接扩展

不需要新 idea，从当前代码框架直接延伸：

### 1. Latent Space 验证

拿 pretrained model 做 inference 时 hook attention，零训练成本：

| 模型 | 架构 | 来源 |
|------|------|------|
| SD1.5 | UNet | HF diffusers |
| SDXL | UNet | HF diffusers |
| PixArt-α | DiT | HF diffusers |
| Flux | DiT (MMDiT) | black-forest-labs |

Hook 不同 denoising step 的 attention weights，画 per-t ERF 曲线。**如果结论跨架构一致，说明这不只是 DiT 的 artifacts。**

### 2. 更高分辨率实验

256×256 或 512×512 上跑 full / na7 / na15 三个 variant，验证最优 kernel size 是否随分辨率变化。

### 3. t-adaptive Kernel 方法

如果 per-t ERF 差异显著，设计 kernel size 随 t 变化的方法：

- **离散调度**：前一半采样步数大窗口，后一半小窗口，零额外成本
- **连续调度**：`k(t) = k_min + (k_max - k_min) * t^α`
- **可学习调度**：一个轻量 MLP 根据 (t, layer_index) 预测最优 kernel size

对比 baseline：固定 k vs t-adaptive k vs full attention (upper bound)。

---

## 二、中期深化

### 1. 理论分析

从 flow ODE 角度解释为什么 t 影响 optimal receptive field：

```
t→1: x_t 接近纯噪声，velocity field 是"从噪声到数据分布"的全局映射
      → 需要 global context

t→0: x_t 接近数据流形，velocity 近似"流形上的精细调整"
      → 局部即可
```

可能的分析工具：
- Score function 的 Lipschitz 常数随 t 的变化
- 最优传输耦合矩阵在不同 t 下的 support size
- 一维 toy case 的闭式解（Gaussian → Gaussian mixture）

### 2. 跨架构验证

在 UNet-based（SD系列）、DiT-based（PixArt/SD3）、Autoregressive（LlamaGen）上系统测量 per-t attention 模式，证明现象的普适性。

### 3. 扩展到视频生成

时间维度的 attention receptive field 可能有类似模式：
- 早期 step → 需要看远帧（全局运动）
- 后期 step → 只需要近帧（局部细节）

---

## 三、长期方向

### 方向 A：动态计算生成模型

核心 idea：生成过程的不同阶段需要不同计算量，为什么要 uniform？

```
传统: step 1→20 用同一个模型

动态:
  step 1-5:   重 block — global attention, 全层激活
  step 6-15:  常规 block — 中等窗口 NA
  step 16-20: 轻量 block — 小窗口 NA 甚至 MLP-only
```

不只是 kernel size，是 **t-adaptive architecture**：
- Token pruning：早期全量，后期逐步剪枝
- Layer skipping：某些层在某 t 区间直接跳过 attention
- 宽度动态调整：MLP ratio 随 t 变化

### 方向 B：Attention 可解释性工具

将 ERF + distance distribution 测量方法做成通用分析框架，系统研究 generative models 的 attention 行为模式。类似 Anthropic 的 Transformer Circuits 但针对生成模型，这个方向几乎是空白。

### 方向 C：高效生成架构设计

从 NA analysis 出发设计更好的 attention variant：
- Learned sparsity：让模型自己学每个 token 该看谁
- Multi-scale attention heads：不同 head 用不同窗口大小
- Dilated NA 在生成中的系统研究（目前没有）
- Timestep-conditional routing：某些层在某些 t 完全 skip attention

---

## 关键决策点

```
Phase 1 跑完后:

per-t ERF 差异 ≥ 2×  →  Story 牢靠，全速推进 t-adaptive
per-t ERF 差异中等    →  存在但不惊艳，补 latent space 实验再看
per-t ERF 差异很小    →  不往 t-adaptive 走，转向 NA 在 FM 下的系统对比
                        （作为 HDiT 在 Flow Matching 下的复现+扩展仍有价值）
```

---

## 参考工作

- HDiT (Crowson et al., ICML 2024): pixel-space NA + DDPM
- ΔConvFusion: ERF measurement methodology
- PiT (Heo et al., ICCV 2021): distance distribution methodology
- SiT (Ma et al., ECCV 2024): Flow Matching for diffusion
- DiT (Peebles & Xie, ICCV 2023): transformer backbone
- DynamicViT (Rao et al., NeurIPS 2021): token pruning → 动态计算参考
- Round and Round: step-adaptive diffusion → 动态采样参考
