# pixelflow_na — 未来研究路线图

> 2026-07-17 · 基于当前项目方向的完整演化路径

---

## 当前项目定位

**核心问题**：DDPM 上 NA 的 receptive field 结论（ERF < 15, 99% token 交互 ≤ 6）在 Flow Matching 下还成立吗？

**当前工作（Phase 1）**：DiT + ImageNet-64 + 6 种 kernel size (full / na3~na15) 的 sweep 实验，per-t ERF 和 distance distribution 分析。

---

## 同一 Research Direction 的产出演化

一个方向不是一篇论文，而是一组论文的雪球：

```
Research Direction: Understanding & Exploiting
Receptive Field Dynamics in Generative Models

论文 1: 发现 ────→ 论文 2: 发现+方法 ────→ 论文 3: 理论+系统 ────→ 论文 4: 期刊综述
```

每篇在前一篇基础上新增一个核心贡献，不是灌水。

---

## 硕士三年时间线

```
研0 暑假（2026.7 — 2026.9）
  ├─ Phase 1: 6 variants sweep → per-t ERF 数据
  └─ → arXiv preprint / 顶会 workshop

研一上（2026.9 — 2027.1）
  ├─ + latent space 实验（SD/Flux pretrained）
  ├─ + t-adaptive kernel 方法设计
  ├─ + 256×256 扩展实验
  └─ → CVPR 2027 投稿（11月 deadline）

  寒假: 实习面试准备

研一下（2027.2 — 2027.8）
  ├─ 论文结果出来后分支:
  │   ├─ 中 → 扩展版投 TPAMI/TMLR
  │   └─ 没中 → 补实验转投 NeurIPS 2027（5月 deadline）
  └─ 暑假: 第一段实习

研二上（2027.9 — 2028.1）
  ├─ 方向深化（三选一，见下文）
  └─ → CVPR/ICLR 2028 投稿

研二下（2028.2 — 2028.8）
  ├─ 第二段实习 + 秋招准备
  └─ 论文 revision / 第三篇推进

研三（2028.9 — 2029.6）
  ├─ 毕业论文收尾
  └─ 期刊长文投稿（TPAMI/IJCV）
```

---

## 短期扩展（研0~研一上，2-3个月工作量）

直接从当前代码框架延伸，不需要新 idea：

### 1. Latent Space 验证（性价比最高）

拿 pretrained model 做 inference 时 hook attention，零训练成本：

| 模型 | 架构 | latent 分辨率 | 来源 |
|------|------|-------------|------|
| SD1.5 | UNet | 64×64 | HuggingFace diffusers |
| SDXL | UNet | 128×128 | HuggingFace diffusers |
| PixArt-α | DiT | 64×64 | HuggingFace diffusers |
| Flux | DiT (MMDiT) | latent packing | black-forest-labs |

- Hook 不同 denoising step 上的 attention weights
- 画 per-t ERF 曲线
- **如果结论跨架构一致 → story 强度翻倍，不只是 DiT 的 artifacts**

### 2. 256×256 扩展实验

- 只跑 full / na7 / na15 三个 variant
- 短训练（50 epochs）验证可扩展性
- 关注：更高分辨率下最优 kernel size 是否变化

### 3. t-adaptive Kernel 方法设计

不止简单的 `k(t) = k_max * t`：

- **方案 A — 离散调度**：前 50% 采样步数用大 k，后 50% 用小 k，零额外成本
- **方案 B — 连续调度**：`k(t) = k_min + (k_max - k_min) * t^α`，α 可调或可学
- **方案 C — Per-layer + per-t predictor**：`k(l, t) = f_θ(layer_embed, t_embed)`，轻量 MLP 预测每层每个 t 的最优 kernel size
- **对比 baseline**：统一固定 k，t-adaptive k，full attention（upper bound）

---

## 中期深化（研一~研二，6-12个月）

这里开始回答"为什么"而不只是"是什么"：

### 1. 理论分析 — 为什么 t 影响 Optimal Receptive Field

从 flow ODE 角度形式化：

```
dx/dt = v(x, t)

t→1: x_t 接近纯噪声
      velocity field 是一个"从噪声到数据分布"的全局映射
      → 需要 global context
      → loss landscape 的非局部曲率大

t→0: x_t 接近数据流形
      velocity 近似"流形上的精细调整"
      → 局部即可
      → loss landscape 接近凸，局部曲率主导
```

可尝试的分析工具：
- **Score function 的 Lipschitz 常数** 随 t 变化
- **最优传输 (OT)** 视角：不同 t 下的耦合矩阵需要不同的 support size
- 一维 toy case 的闭式解（Gaussian → Gaussian mixture）

### 2. 跨架构系统实验

证明现象的**普适性**：

| 架构 | 代表模型 | 分析要点 |
|------|---------|---------|
| UNet-based | SD1.5 / SDXL | cross-attention vs self-attention 的 ERF 差异 |
| DiT-based | PixArt-α / SD3 | 和本项目的 DiT 实验对照 |
| Autoregressive | LlamaGen | next-token 的 attention 是否也有时序上的 receptive field 变化 |
| Video | Open-Sora / CogVideo | temporal attention 的 per-step behavior |

### 3. 扩展到视频生成

时间维度的 receptive field 有类似模式：
- 早期 denoising step → 需要看远帧（全局运动、叙事结构）
- 后期 denoising step → 只需要看近帧（局部动作细节）

竞争者很少，且视频生成正火。

---

## 长期方向（研二~研三，1-2年）

### 方向 A：动态计算生成模型（最推荐）

**核心 idea**：生成过程的不同阶段需要不同的计算量，为什么要 uniform？

当前所有 diffusion/flow 模型都是 **uniform compute per step**。per-t ERF 分析提供了一个具体切口：

```
传统架构:
  step 1→20: 同一个 DiT，同一组参数，同一组层

动态架构:
  step 1-5:   重点 block — global attention, 全层激活
  step 6-15:  常规 block — medium 窗口 NA
  step 16-20: 轻量 block — 小窗口 NA 甚至 MLP-only
```

这比 t-adaptive kernel 更进一步 — **t-adaptive architecture**：
- **Token pruning**: 早期保留全量 token，后期逐步剪枝
- **Layer skipping**: 某些层在某 t 区间跳过 attention 直接 MLP
- **Head specialization**: 不同 head 在不同 t 区间激活
- **宽度动态调整**: MLP ratio 随 t 变化

相关 baseline：DynamicViT、DVT 的"动态 token 剪枝"思路 + 生成模型。

**优势**：可定义新方向、竞争者少、单卡可做。

### 方向 B：Attention 可解释性工具

将 ERF + distance distribution 测量方法做成 **通用分析工具**：

- 统一的 attention 行为分析框架
- 测量不同模型、不同任务、不同训练阶段的 attention 演化
- 发现"好的生成模型有什么 attention 特征"、"训练过程中 attention 如何收敛"
- 类似 Anthropic 的 Transformer Circuits，但针对 generative models 几乎是空白

**可投**：ICLR / NeurIPS interpretability track。

### 方向 C：高效生成架构设计

从 NA analysis 出发设计更好的 attention variant：

- **Learned sparsity**: 不是固定窗口，让模型学每个 token 该看谁
- **Multi-scale attention heads**: 不同 head 用不同 kernel size / dilation
- **Timestep-conditional routing**: 某些层在某些 t 时完全 skip attention
- **Dilated NA 在生成中的系统研究**: dilation 的影响目前没有任何系统性分析

竞争激烈但有实际价值 — 如果真能降 30% 算力同时保持 FID，工业界也会有引用。

---

## 投稿策略速查

| 会议/期刊 | 级别 | 适合什么 | Deadline 参考 |
|-----------|------|---------|--------------|
| **CVPR** | A | 方法+实验扎实 | 11月 |
| **ICCV** | A | 同上，奇数年 | 3月 |
| **NeurIPS** | A+ | 有理论 or 有惊喜 | 5月 |
| **ICLR** | A+ | 有理论/可解释性 | 10月 |
| **ICML** | A+ | 理论强 | 1月 |
| **BMVC** | B+ | 实验型工作 | 5月 |
| **ACCV** | B+ | 亚洲地区，门槛适中 | 5月 |
| **WACV** | B | 应用导向 | 7月 |
| **TMLR** | 期刊 | 分析型工作，审稿质量高 | 随时 |
| **TPAMI** | A 刊 | 系统性长文综述 | 随时 |
| **IJCV** | A 刊 | 同上 | 随时 |

### 策略建议

- **第一篇（研0）**：顶会 workshop → 快速曝光，低风险
- **第二篇（研一）**：BMVC / ACCV / CVPR → 核心贡献
- **第三篇（研二）**：CVPR / NeurIPS / ICLR → 冲刺
- **第四篇（研三）**：TPAMI / TMLR → 收官

---

## 关键决策点

```
Phase 1 跑完（2-4周后）
  │
  ├─ per-t ERF 差异 ≥ 2× ──→ Story 牢靠，全速推进
  │
  ├─ per-t ERF 差异 1.2-2× ──→ 存在但不惊艳，补 latent space 实验再看
  │
  └─ per-t ERF 差异 < 1.2× ──→ 现象不显著
       ├─ 及时止损，不往 t-adaptive 方向走
       └─ 转向 NA 本身在 Flow Matching 下的性能分析
           （作为 HDiT 在 FM 下的系统复现+扩展，仍有发表价值）
```

---

## 风险与应对

| 风险 | 概率 | 应对 |
|------|------|------|
| per-t ERF 差异不显著 | 中 | 转向 NA vs Full 的系统对比 + 架构分析 |
| 被类似工作 scoop | 中低 | 尽早 arXiv preprint |
| 256×256 单卡跑不动 | 中 | 用 gradient accumulation + 小 batch；或只测 pretrained |
| 理论分析做不动 | 高（短期） | 先用实验喂饱前两篇，理论留给第三篇 |
| NATTEN kernel 兼容问题 | 低 | PyTorch fallback 已就绪 |

---

## 参考工作

- **HDiT** (Crowson et al., ICML 2024): pixel-space NA + DDPM → 对比基准
- **ΔConvFusion** (anonymous, under review): ERF measurement methodology → 测量方法
- **PiT** (Heo et al., ICCV 2021): distance distribution methodology → 测量方法
- **SiT** (Ma et al., ECCV 2024): Flow Matching for diffusion → 训练框架
- **DiT** (Peebles & Xie, ICCV 2023): transformer backbone → 架构参考
- **Scalable Diffusion with Transformers**: 更大规模 DiT → 扩展方向
- **DynamicViT** (Rao et al., NeurIPS 2021): token pruning → 动态计算参考
- **Round and Round**: step-adaptive diffusion → 动态采样参考
