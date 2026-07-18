# Flow Matching 下 Self-Attention 局部性验证

> 2026-07-18 · 核心问题：ΔConvFusion 证明了 DDPM 下 self-attention 是局部的，FM 下也是这样吗？

---

## 🔄 方向 Pivot（2026-07-18）

| ❌ 旧方向 — 方法开发 | ✅ 新方向 — 行为验证 |
|---|---|
| Fine-tune NA、t-adaptive kernel | **Post-hoc ERF 分析，零训练** |
| 核心卖点：「设计更好的 attention 机制」 | 核心问题：「FM 下 self-attention 也是局部的吗？」 |
| 需要大量 GPU 训练 | 加载 pretrained 模型 → 推理 → 算 ERF |

**关键发现**：ΔConvFusion (ICCV 2025) 的 ERF 分析方法是纯 post-hoc 的——他们只分析了 DDPM 模型（SD1.5, SDXL, PixArt），**没有任何人在 Flow Matching 模型上做过同样的分析。**

---

## 🎯 研究 Gap

| 论文 | 分析了 attention 局部性？ | 场景 | 没做的事 |
|------|:---:|------|------|
| **🎯 我们的工作** | ✅ | **Flow Matching** | — |
| ΔConvFusion (ICCV 2025) | ✅ ERF < 15 | DDPM | 没在 FM 下测 ERF |
| PiT (2025) | ✅ 99% 交互 ≤ 6 | DDPM | 没在 FM 下统计距离分布 |
| HDiT (ICML 2024) | ✅ NA 有效 | DDPM | 没在 FM 下验证 NA |

> **结论**：Self-attention 在 DDPM 下的局部性已被充分验证，但**没有任何人在 Flow Matching 下系统地重新验证这个结论。** 不是发明新东西，是填补验证空白。

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

ΔConvFusion 是对**已训练好模型**的 attention pattern 做后验分析。我们的问题——"FM 下 attention 也是局部的吗？"——用别人训好的 FM 模型直接测就能回答。

---

## 📋 实验矩阵

| 模型 | 训练框架 | 架构 | Tokens | 状态 |
|------|---------|------|--------|------|
| SiT-XL/2 | Flow Matching | DiT | 256 (16×16) | ✅ 已有权重 |
| SiT-B/2 | Flow Matching | DiT | 256 | ✅ 可选 |
| Flux/SD3 | Flow Matching | MMDiT | 待定 | 🟡 后续 |

### 对比基线（文献值）

| 模型 | 训练框架 | 报告 ERF |
|------|---------|----------|
| PixArt | DDPM | < 15×15 |
| SD1.5 | DDPM | < 20×20 |

---

## ⚠️ 风险评估

### 🔴 高风险

1. **FM 下 attention 行为和 DDPM 下完全一样** — 那贡献就薄了。防御：(1) 第一手 FM 验证数据本身有发表价值；(2) 不做预设，发现任何差异都是完整论文种子。

2. **审稿人说纯验证不是 novel contribution** — 防御：不只是验证，还要解释 ODE 直线路径 vs SDE 弯曲路径对 attention 的理论影响。

### 🟡 中风险

3. **SiT-XL/2 token 数少（256），ERF 可能不够有区分度** — 防御：换 SiT 的更大分辨率变体或 Flux。

---

## 📋 论文标题思路

- **推荐**：*Is Self-Attention Still Local under Flow Matching? A Post-Hoc Receptive Field Analysis*
- **备选**：*Do Straight Flows Need Less Context? Revisiting Attention Locality in Flow Matching Models*

---

## 📁 相关资源

- **ΔConvFusion 论文**：Zotero `JGZ5NGCH`
- **SiT 论文**：Zotero `97FPMY55`
- **本地模型**：`SiT/checkpoints/SiT-XL-2-256.pt` (2.6GB)
- **VAE**：`vae/` (sd-vae-ft-mse)
- **测量数据**：`results/sit_measure/` (full attention 已有数据)

---

*Generated 2026-07-16 · Rewritten 2026-07-17（storyline pivot: 架构创新 → NA 行为验证）· Rewritten 2026-07-18（pivot: 训练→post-hoc ERF 分析）*
