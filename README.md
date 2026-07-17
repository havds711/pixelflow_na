# Flow-NA: Flow Matching × Neighbor Attention for Pixel-Space Image Generation

> 论文阅读指南 · 2026-07-17 · 4 篇核心 + 2 篇参考 + 8 篇补充

---

## 🔄 方向 Pivot

| ❌ 旧方向（已放弃） | ✅ 新方向（当前） |
|---|---|
| DDPM 扩散 + 像素空间 + flat DiT + NA | **Flow Matching + NA + 像素空间** |
| 核心卖点：「flat 架构也能做 pixel-space 生成」 | 核心卖点：「更少采样步数 × 更低单步计算 = 双重加速」 |
| 问题：HDiT 已做 NA + pixel-space，flat vs hourglass 增量不够 | 空白：PixelFlow 做了 flow + pixel（全 attn），HDiT 做了扩散 + pixel + NA——**还没人把 flow 和 NA 拼在 pixel-space** |
| 28 篇阅读清单，先有锤子再找钉子 | 14 篇核心阅读，问题驱动——先找坑再填坑 |

```
Flow Matching          Neighbor Attention       Pixel Space
(采样快，步数少)   +   (单步计算省)        +   (无 VAE 压缩)
          ↓                    ↓                      ↓
          └────────────────────┴──────────────────────┘
                               ↓
                    双重加速：没人在 flat DiT 做过
```

---

## 🎯 Gap 验证：这个坑还空着吗？

经过系统性搜索（2026-07-17），确认 **Flow Matching × NA × Pixel Space × Flat DiT** 的组合至今无人做过。

| 论文 | Flow Match | Pixel Space | Local Attn | Flat DiT | 差距 |
|------|-----------|-------------|------------|----------|------|
| **🎯 你的 Flow-NA** | ✅ | ✅ | ✅ NA | ✅ | — |
| MPDiT (CVPR 2026) | ✅ | ❌ latent | 🟡 multi-patch | ✅ | latent space |
| AsymFlow (2026) | ✅ | ✅ | ❌ full | ✅ | 全 attention |
| HDiT (ICML 2024) | ❌ diffusion | ✅ | ✅ NA | ❌ hourglass | 扩散+hourglass |
| Graph Flow Matching (AAAI 2026) | ✅ | ❌ latent | 🟡 GNN | ❌ U-Net | latent+GNN |
| PixelFlow (2025) | ✅ | ✅ | ❌ full | — | 全 attention |

> **结论**：窗口期还在，但正在快速收窄。MPDiT 和 AsymFlow 是最强烈的信号——社区已经在往 flow + efficient attention 方向走，但还没人在 pixel-space flat DiT 上做 NA。

---

## 📋 阅读清单

### 🔴 核心 1-3：你的直接前驱 + 理论武器 + 框架选择

#### 1. PixelFlow: Pixel-Space Generative Models with Flow
- **Shoufa Chen 等 · 2025 · HKU + Adobe** · arxiv: 2504.07963
- 📄 `pixel_try/papers/2504.07963.pdf` | 📁 Zotero → 图像生成 / 像素空间生成
- 做了 flow + pixel-space，但是**全 attention**。你的直接前驱——把它的全 attention 换成 NA，就是你的方案。
- **读完要回答**：全 attention 瓶颈在哪层？换成 NA 预期省多少计算？

#### 2. ΔConvFusion: Can We Achieve Efficient Diffusion without Self-Attention?
- **Ziyi Dong 等 · ICCV 2025** · arxiv: 2504.21292
- 📄 `pixel_try/papers/2504.21292.pdf` | 📁 Zotero → 图像生成 / 像素空间生成
- **最强理论武器**。证明了扩散模型 attention 的有效感受野 < 15×15。NA 不是妥协，是理论支撑的设计。
- **读完要回答**：感受野 < 15×15 在 flow matching 下还成立吗？（flow 路径更直，去噪更快，感受野需求可能不同）

#### 3. SiT: Scalable Interpolant Transformers
- **Nanye Ma 等 · ECCV 2024** · arxiv: 2401.08740
- 📁 Zotero → 图像生成 / 扩散模型核心
- 统一扩散和 flow matching 的框架。让你能科学论证「为什么选 flow 不选扩散」。
- **读完要回答**：pixel-space + NA 场景下哪个 interpolant 最合适？

### 🟡 参考 4-5：代码工具箱

#### 4. NAT: Neighborhood Attention Transformer
- **Ali Hassani 等 · CVPR 2023** · arxiv: 2204.07143
- 📁 Zotero → 图像生成 / 高效注意力
- NA 数学定义和 CUDA 实现（NATTEN 库）。重点看 Section 3：滑动窗口 attention、kernel size 设计空间。

#### 5. StyleNAT: Efficient Image Generation with Variadic Attention Heads
- **Steven Walton 等 · CVPR 2025 Workshop** · arxiv: 2211.05770
- 📁 Zotero → 图像生成 / 高效注意力
- Hydra-NA：同一层不同 head 看不同 kernel size。如果单一 kernel size 效果不好，这是你的升级路线。

---

## ⚠️ 风险评估

### 🔴 高风险

1. **竞争窗口期很短** — MPDiT 已做 flow + global→local（latent），扩展到 pixel space 只是迁移成本。如果实验周期超 3-4 个月，可能被抢先。

2. **ΔConvFusion 的 ERF 结论在 flow 下需重新验证** — ERF < 15×15 在扩散上测得。Flow 路径更直、每步去噪量更大——可能意味着**需要更大而非更小的感受野**。这是最需要实验验证的关键假设。

3. **Consistency Model 从根本上挑战多步生成** — "There is No VAE"（ICLR 2026）单步 FID 1.58，CrossFlow one-step（2026）一步 FID 1.62。必须准备回答：「既然可以一步，为什么还需要多步 flow？」防御：flow 可在质量/速度间灵活权衡（1-50 步）；pixel-space consistency model 训练稳定性存疑。

### 🟡 中风险

4. **Flat DiT + Pixel Space 的固有矛盾** — Flat 架构每层全分辨率（64×64=4096 tokens），HDiT hourglass 瓶颈层仅 256 tokens。需论证「为什么 flat > hourglass 对 flow matching 特别重要」。

5. **AsymFlow / FreqFlow 已把 pixel-space flow FID 压到 1.38-1.57** — 建议定位为「效率优先」而非「SOTA FID」。你的卖点是 FLOPS/FID trade-off。

6. **频域方法竞争** — FREPix、FreqFlow、DeCo 的频域分解和 NA 直觉重叠。防御：NA 是更通用的方案，不改变架构设计，可按层定制感受野。

---

## 📋 补充论文（2026-07-17 新增）

### 🔴 必读补充：最危险的竞争者

#### 6. AsymFlow: Asymmetric Flow Models
- **Hansheng Chen 等 · 2026 · Stanford** · arxiv: 2605.12964
- **Pixel-space flow SOTA, FID 1.57**。核心：rank-asymmetric velocity——噪声项投影到低秩子空间。首次从 FLUX.2 fine-tune 到 pixel-space。
- **必读**：低秩投影机制、latent→pixel fine-tune 方法（Procrustes alignment）

#### 7. MPDiT: Multi-Patch Global-to-Local Transformer
- **Quan Dao, Dimitris Metaxas · CVPR 2026 · Rutgers** · arxiv: 2603.26357
- **最接近你的方案**：flow + global→local transformer。早期大 patch（全局）、后期小 patch（局部）。FNO time embedding (+4 FID)。ImageNet FID 2.05。
- **必读**：multi-patch 过渡机制、FNO embedding、跟 NA 方案的对比

#### 8. FreqFlow: Frequency-Aware Flow Matching
- **Rensu Sun 等 · CVPR 2026 · JHU + ByteDance** · arxiv: 2604.15521
- **Flow matching SOTA, FID 1.38**。双分支架构：频域分支 + 空间分支。需论证为什么 NA 比频域分解更好/更通用。
- **略读**：双分支分工、时间依赖权重调度

### 🟡 理论武器

#### 9. On Inductive Biases That Enable Generalization of Diffusion Transformers
- **Jie An 等 · NeurIPS 2025 · Apple** · arxiv: 2410.21273
- NA 的泛化理论支撑：DiT 泛化依赖局部 attention 模式。数据不足时全 attention 有害。NA 不仅是效率优化，还有正则化效果。

#### 10. PiT: Pseudo Shifted Window Attention for Diffusion Transformers
- **Wu 等 · 2025 · Tencent** · arxiv: 2505.13219
- ΔConvFusion 的补充证据：>99% token 交互 distance ≤ 6。54% FID improvement。Introduction 最佳引用——三组独立证据表明全 attention 是冗余的。

### 🔵 高频参考（Phase 1 跑通后再精读）

#### 11. FREPix: Frequency-Heterogeneous Flow Matching for Pixel-Space
- **Ming-Hung Lin 等 · 2026** · arxiv: 2605.06421
- 频域异构 flow + pixel space，FID 1.91。替代路线备选。

#### 12. Graph Flow Matching: Enhancing Image Generation with Neighbor-Aware Flow Fields
- **Siddiqui 等 · AAAI 2026** · arxiv: 2505.24434
- 唯一把「邻居信息」和「flow matching」放一起的论文。GNN 聚合邻居轨迹。Latent + U-Net。

#### 13. StraightFM: Straighter Flow Matching via Diffusion-Based Coupling Prior
- **Siyu Xing 等 · PRCV 2025** · arxiv: 2311.16507
- 扩散先验让 flow 路径更直，5 步高质量生成。

---

## 📍 讨论节点

### 🔴 讨论 #1：读完 PixelFlow + ΔConvFusion + AsymFlow 之后
**预计**：2-3 天

**带着这些问题来**：
- PixelFlow 的全 attention 瓶颈在哪？换成 NA 预期能省多少？
- ΔConvFusion 的 ERF 结论在 flow matching 下还成立吗？
- AsymFlow 的 rank-asymmetric velocity 能不能和 NA 结合？
- MPDiT 的 multi-patch 和 NA，哪个更适合 pixel space？
- **讨论目标**：决定「Flow + NA + pixel-space」是否值得深入

### 🟡 讨论 #2：读完 SiT + NAT + 理论武器之后
**预计**：再 2-3 天

**带着这些问题来**：
- pixel-space + NA 场景选什么 interpolant？
- NA 的 kernel size / dilation / multi-scale head，实验矩阵长什么样？
- Baseline 选什么？（全 attn flow？扩散 NA？）
- **讨论目标**：确定实验矩阵，开始写代码跑 Phase 1

### 🟢 讨论 #3：Phase 1 消融跑完第一批结果之后
**预计**：编码 + 训练 1-2 周

**带着这些数据来**：
- 64×64 上：全 attn flow vs NA flow，不同 kernel size 的 FID / 采样速度
- 哪个 kernel size 最好？dilation 有帮助吗？
- Flow 采样步数 sensitivity——步数少的时候 NA 的优势变大还是变小？
- **讨论目标**：判断实验结果是否足够支撑一篇论文

---

## 📐 路线总览

```
现在 → PixelFlow + ΔConvFusion + AsymFlow → 讨论#1(方向决策)
     → SiT + NAT + 理论武器(PiT, Inductive Bias) → 讨论#2(实验设计)
     → 写代码跑 Phase 1（先验证 ERF！） → 讨论#3(论文决策)
```

> **核心原则**：14 篇论文 + 3 次讨论 + 实验驱动的补充阅读。阅读应该被「我要回答哪个问题」驱动，不是被「我想掌握这个领域」驱动。掌握领域是写论文的副产品，不是前提。

### 🗺️ 阅读优先级

| 优先级 | 论文 | 时间 |
|--------|------|------|
| 🔵 第一优先（讨论#1前） | PixelFlow, ΔConvFusion, AsymFlow, MPDiT | 2-3天 |
| 🟣 第二优先（讨论#2前） | SiT, NAT, StyleNAT, On Inductive Biases, PiT, FreqFlow | 再2-3天 |
| ⚪ 后续参考 | FREPix, GFM, StraightFM, 旧指南 Round 1-2 | Phase 1跑通后 |

---

## 📁 相关资源

- **旧阅读指南**：[像素空间生成-论文阅读指南.html](像素空间生成-论文阅读指南.html) — Round 1-2（HDiT、NAT、DiNAT、NABLA 等）作为参考库
- **HDiT 架构拆解**：[HDiT-architecture-explained.html](HDiT-architecture-explained.html)
- **代码仓库**：`pixel_try/` — DiT + pixel space 实验代码
- **本地论文**：`pixel_try/papers/` — 已下载 PDF

---

*Generated 2026-07-16 · Updated 2026-07-17（新增竞争格局 + 风险评估 + 8篇补充论文）*
