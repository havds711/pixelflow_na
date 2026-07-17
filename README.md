# NA 在 Flow Matching 下的行为验证

> 论文阅读指南 · 2026-07-17 · 核心问题：DDPM 上 NA 的结论，搬到 Flow Matching 还成立吗？

---

## 🔄 方向 Pivot

| ❌ 旧方向 — 架构创新 | ✅ 新方向 — 行为验证 |
|---|---|
| DDPM 扩散 + 像素空间 + flat DiT + NA | **NA 在 Flow Matching 下的系统验证** |
| 核心卖点：「flat 架构也能做 pixel-space」 | 核心问题：「前人在 DDPM 上证明的结论，flow 下还成立吗？」 |
| 问题：HDiT 已做，flat vs hourglass 增量不够 | 方法：前人定义了测什么+怎么测 → flow 下重新测 → 解释异同 |
| 28 篇阅读，先有锤子再找钉子 | ~14 篇核心阅读，问题驱动 |

```
HDiT/ΔConvFusion/PiT        同样方法            发现相同/不同
在 DDPM 上证明了：     →    搬到 Flow Matching   →  解释为什么
① NA 有效               重新测一遍              （flow 路径更直→?）
② ERF < 15
③ 99% 交互 ≤ 6
```

---

## 🎯 实验逻辑：前人怎么做，你就怎么做

| 分析维度 | 方法来源 | 在 DDPM 上的结论 | 你要在 Flow 上做什么 |
|----------|----------|-----------------|-------------------|
| **NA vs 全 Attention FID 对比** | HDiT Table 1 | NA FID 略差但计算大降 | 全 attn flow vs NA flow，不同 kernel size，FID + GFLOPs |
| **有效感受野 (ERF) 测量** | ΔConvFusion §3 | ERF < 15×15 | 复现测量方法，看 flow 下 ERF 更大还是更小 |
| **Token 交互距离分布** | PiT §3 | >99% 交互 distance ≤ 6 | 统计 flow 模型各层 attention distance |
| **不同采样步数下的行为** | SiT interpolant 框架 | 扩散步数多，flow 步数少 | Flow 步数越少，NA 的优势变大还是变小？ |

### Baseline

最简单的 flow DiT：标准 DiT backbone + flow matching loss，固定分辨率（如 64×64），不做 cascade。唯一变量 = attention 类型。

### 论文标题思路

- **推荐**：*Revisiting Neighborhood Attention under Flow Matching: Is Local Enough for Straight Paths?*
- **备选**：*Do Straight Flows Need Less Context? A Systematic Study of Neighborhood Attention for Flow Matching*

---

## 🎯 Gap 验证

核心问题：**NA 在 DDPM 下的结论，搬到 Flow Matching 下还成立吗？没人回答过。**

| 论文 | 验证了 NA 有效？ | 测量了 ERF？ | 场景 | 没做的事 |
|------|:---:|:---:|------|------|
| **🎯 你的工作** | ✅ | ✅ | Flow Matching | — |
| HDiT (ICML 2024) | ✅ | ❌ | DDPM | 没在 flow 下验证；没测 ERF |
| ΔConvFusion (ICCV 2025) | 🟡 蒸馏成 conv | ✅ ERF<15 | DDPM | 没在 flow 下测 ERF |
| PiT (2025) | ✅ | ✅ 99%≤6 | DDPM | 没在 flow 下统计距离分布 |
| Graph Flow Matching (AAAI 2026) | 🟡 GNN neighbor | ❌ | Flow Matching | 用 GNN 不是标准 NA；latent space |

> **结论**：NA 在 DDPM 下被充分验证，但**没有任何人在 Flow Matching 下系统地重新验证这些结论**。不是发明新东西，是填补验证空白。

---

## 📋 阅读清单

### 🔴 核心 1-3：方法论 + 工具 + 场景

#### 1. HDiT (ICML 2024) — NA 在 DDPM 下怎么验证的
- **Katherine Crowson 等** · arxiv: 2401.11605
- 📄 `pixel_try/papers/2401.11605.pdf`
- 第一篇把 NA 用在 pixel-space 生成。关键不是它的 hourglass——是**消融方法论**（Table 1）。你的 Phase 1 照搬这个表格结构。
- **必读**：Table 1 消融路线、§3.3 NA 配置、NA vs Swin vs 全局 attention 对比

#### 2. ΔConvFusion (ICCV 2025) — ERF 怎么测
- **Ziyi Dong 等** · arxiv: 2504.21292
- 📄 `pixel_try/papers/2504.21292.pdf`
- 定义了 ERF 测量方法。你的关键实验就是用它同样的方法在 flow 上重测。
- **必读**：§3 感受野分析方法（可视化→定量统计→频域验证）
- **读后回答**：测量步骤能不能直接搬到 flow 模型？

#### 3. SiT (ECCV 2024) — Flow Matching 框架
- **Nanye Ma 等** · arxiv: 2401.08740
- 统一扩散和 flow 的框架。帮你选 interpolant，解释为什么 flow 下行为可能不同。
- **必读**：§3 Interpolant Framework、扩散 vs flow 对比实验

### 🟡 参考 4-5：NA 工具箱

#### 4. NAT (CVPR 2023) — NA 定义 + CUDA 实现
- **Ali Hassani 等** · arxiv: 2204.07143
- 重点看 §3 NA 数学定义、kernel size 设计空间。不需要精读。

#### 5. StyleNAT (CVPR 2025 Workshop) — 多尺度升级备选
- **Steven Walton 等** · arxiv: 2211.05770
- Hydra-NA：同一层不同 head 用不同 kernel size。Phase 2 升级路线。

---

## ⚠️ 风险评估

### 🔴 高风险

1. **Flow 下 NA 行为和 DDPM 下完全一样** — 那你的贡献就薄了。防御：(1) 第一手验证数据本身有发表价值；(2) 不做预设，发现任何差异就是完整论文种子。

2. **审稿人说纯验证不是 novel contribution** — 防御：不只是验证，还要解释机制 + 给出 actionable insight（如"flow 下用 kernel X 最划算"）。

### 🟡 中风险

3. **Flow Matching 本身在快速演进** — Consistency model、AsymFlow 一步生成。防御：NA 是通用 attention 设计，不依赖特定生成框架。

4. **4×3090 能不能跑 ERF 测量？** — 防御：64×64 + DiT-S 足够 Phase 1；或拿开源 flow 模型做 post-hoc 分析。

5. **Graph Flow Matching 已碰"邻居+flow"概念** — 防御：它用 GNN 不是标准 NA，且没做 ERF/distance 行为分析，你跟它正交。

---

## 📋 补充论文（2026-07-17 新增）

### 🔴 必读补充

| # | 论文 | 信息 | 为什么读 |
|---|------|------|----------|
| 6 | **AsymFlow** | Stanford 2026 · arxiv:2605.12964 | Pixel-space flow SOTA, FID 1.57。知道天花板在哪 |
| 7 | **MPDiT** | CVPR 2026 · arxiv:2603.26357 | flow + global→local，最近竞争者 |
| 8 | **FreqFlow** | CVPR 2026 · arxiv:2604.15521 | Flow SOTA, FID 1.38。知道频域路线优势 |

### 🟡 理论武器

| # | 论文 | 信息 | 为什么读 |
|---|------|------|----------|
| 9 | **On Inductive Biases of DiT** | NeurIPS 2025 · arxiv:2410.21273 | 局部 attention 改善 DiT 泛化——NA 不仅是效率优化 |
| 10 | **PiT** | Tencent 2025 · arxiv:2505.13219 | >99% token 交互 ≤6，ERF 小结论的第三组独立证据 |

### 🔵 高频参考（Phase 1 跑通后再读）

| # | 论文 | 信息 |
|---|------|------|
| 11 | **FREPix** | 频域异构 flow + pixel, FID 1.91 |
| 12 | **Graph Flow Matching** | AAAI 2026, 邻居+flow 用 GNN |
| 13 | **StraightFM** | 扩散先验拉直 flow 路径 |

---

## 📍 讨论节点

### 🔴 讨论 #1：读完 HDiT + ΔConvFusion 之后（2-3 天）
- HDiT 的消融方法论能不能直接搬到 flow？
- ΔConvFusion 的 ERF 测量在 flow 上复现难度多大？
- **目标**：确定实验矩阵，开始搭 flow DiT baseline

### 🟡 讨论 #2：读完 SiT + PiT 之后（再 2-3 天）
- 选哪个 interpolant？baseline 能跑通吗？
- PiT 的 distance 统计能否复现？
- **目标**：确认实验矩阵 + 开始跑 Phase 1

### 🟢 讨论 #3：Phase 1 第一批结果（1-2 周）
- 全 attn flow vs NA flow (k=3,5,7,11,15)，FID + GFLOPs 表
- Flow 下 ERF = ？Distance ≤ ？
- 采样步数 sensitivity
- **目标**：短文 or 完整论文？发现相同 or 不同？

---

## 📐 路线总览

```
现在 → HDiT + ΔConvFusion（学方法论）→ 讨论#1（实验设计）
     → SiT + PiT + 搭 baseline → 讨论#2（确认矩阵）
     → 跑 Phase 1（FID对比 + ERF测量 + Distance分布）→ 讨论#3（决策）
```

> **核心原则**：不是发明新架构，是**方法迁移 + 新场景验证 + 行为解释**。发现相同 → 短文；发现不同 → 完整论文。Phase 2 升级方向（Dilation/Hydra-NA/频域）等 Phase 1 结果出来再选。

---

## 📁 相关资源

- **旧阅读指南**：[像素空间生成-论文阅读指南.html](像素空间生成-论文阅读指南.html)
- **HDiT 架构拆解**：[HDiT-architecture-explained.html](HDiT-architecture-explained.html)
- **本地论文**：`pixel_try/papers/`
- **Zotero**：图像生成 合集

---

*Generated 2026-07-16 · 重写 2026-07-17（故事线 pivot：从架构创新 → Flow Matching 下 NA 行为验证）*
