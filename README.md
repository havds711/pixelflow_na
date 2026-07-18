# Diffusion Models 中 Self-Attention 局部性验证

> 2026-07-18 · 核心问题：Self-attention 在扩散模型中到底有多"全局"？ΔConvFusion 回答了 SDE 扩散，ODE 扩散（Flow Matching）还没人回答。

---

## 🔄 方向 Pivot（2026-07-18）

| ❌ 旧方向 — 方法开发 | ✅ 新方向 — 行为验证 |
|---|---|
| Fine-tune NA、t-adaptive kernel | **Post-hoc ERF 分析，零训练** |
| 核心卖点：「设计更好的 attention 机制」 | 核心问题：「FM 下 self-attention 也是局部的吗？」 |
| 需要大量 GPU 训练 | 加载 pretrained 模型 → 推理 → 算 ERF |

**背景**：ΔConvFusion (ICCV 2025) 问了一个通用问题——"Does self-attention in diffusion models primarily capture global dependencies, or does it behave more locally?" 他们用 post-hoc ERF 分析回答：**是局部的**。但他们分析的全是 SDE-based 扩散模型（SD1.5, SDXL, PixArt）。

**我们的问题**：这个结论搬到 ODE-based 扩散（Flow Matching — SiT, Flux, SD3）还成立吗？直线路径下模型去噪更简单，需要的 attention 上下文可能更少——也可能更多。**没有人在 FM 模型上做过同样的分析。**

Flow Matching 也是 diffusion model 的一种（SiT 论文标题就是 "Exploring Flow **and** Diffusion-based Generative Models"），我们不是比较两个不相干的框架，而是在同一个扩散模型大类下补上 ODE 分支的分析空白。

---

## 🎯 研究 Gap

ΔConvFusion 自己问的问题：
> "Does self-attention in diffusion models primarily capture global dependencies, or does it behave more locally?"

| 论文 | 分析过的扩散模型 | 没做的事 |
|------|:---:|------|
| **🎯 我们的工作** | ODE 扩散（FM: SiT, Flux） | — |
| ΔConvFusion (ICCV 2025) | SDE 扩散（DDPM: SD1.5, SDXL, PixArt） | 没测 FM 模型 |
| PiT (2025) | SDE 扩散（DDPM） | 没测 FM 模型 |
| HDiT (ICML 2024) | SDE 扩散（DDPM） | 没测 FM 模型 |

> **结论**：同一个问题（"diffusion 里 attention 是局部的吗？"），前人只回答了 SDE 分支。我们补上 ODE（Flow Matching）分支的答案。

---

## 📐 方法

### ΔConvFusion ERF 测量（Section 3.2）

```
1. 提取 Attention Map
   pretrained FM 模型 → 推理 → 抓每层 post-softmax attention weights

2. DFT → 高通 Butterworth 滤波
   分离 attention 的高频（结构信息）和低频（均匀偏置）分量

3. 计算 ASM（Attention Score Mass）
   ASM(K) = 高频 attention 在 K×K 局部窗口内的占比

4. 确定 ERF
   ERF = min{K | ASM(K) ≥ 80%}
```

### 为什么不需要训练

ΔConvFusion 的方法本身就是纯 post-hoc 的——加载已训好的模型，跑推理，抓 attention map，算 ERF。一步训练都没有。我们的工作就是把这个方法搬到他们没测过的模型上。

---

## 📋 实验矩阵

### 我们分析（ODE 扩散 / Flow Matching）

| 模型 | 框架 | 架构 | Tokens | 状态 |
|------|------|------|--------|------|
| SiT-XL/2 | Flow Matching | DiT | 256 (16×16) | ✅ 已有权重 |
| SiT-B/2 | Flow Matching | DiT | 256 | ✅ 可选 |
| Flux/SD3 | Flow Matching | MMDiT | 待定 | 🟡 后续 |

### ΔConvFusion 已分析（SDE 扩散 / DDPM）— 我们的对比基线

| 模型 | 框架 | 报告 ERF |
|------|------|----------|
| PixArt | DDPM (SDE) | < 15×15 |
| SD1.5 | DDPM (SDE) | < 20×20 |

---

## ⚠️ 风险评估

### 🔴 高风险

1. **ODE 和 SDE 扩散下 attention 局部性完全一样** — 那贡献就薄了。防御：(1) 第一手 FM 验证数据本身填补了分析空白；(2) 不做预设，任何差异都是完整论文种子。

2. **审稿人说纯验证不是 novel contribution** — 防御：ΔConvFusion 自己说 "Does self-attention in diffusion models capture global or local dependencies?" 用的是 diffusion models 这个词——暗示结论应该对全部扩散模型成立。但他们的实验只覆盖了 SDE 分支。我们补上 ODE 分支，要么确认这个通用结论，要么发现它不成立——两者都有发表价值。

### 🟡 中风险

3. **SiT-XL/2 token 数少（256），ERF 可能不够有区分度** — 防御：换 SiT 的更大分辨率变体或 Flux。

---

## 📋 论文标题思路

- **推荐**：*Is Self-Attention in Diffusion Models Really Local? A Post-Hoc Analysis of Flow Matching Models*
- **备选**：*Do Straight Flows Need Less Context? Revisiting Attention Locality in ODE-based Diffusion*

---

## 📁 相关资源

- **ΔConvFusion 论文**：Zotero `JGZ5NGCH`
- **SiT 论文**：Zotero `97FPMY55`
- **本地模型**：`SiT/checkpoints/SiT-XL-2-256.pt` (2.6GB)
- **VAE**：`vae/` (sd-vae-ft-mse)
- **测量数据**：`results/sit_measure/` (full attention 已有数据)

---

*Generated 2026-07-16 · Rewritten 2026-07-17 · Rewritten 2026-07-18（post-hoc ERF 分析）*
