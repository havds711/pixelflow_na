# pixelflow_na — 研究路线图

> 2026-07-18 更新 · Pivot: pixel space 从零训 → latent space 基于 pretrained 验证

---

## 🎯 核心判断

**研究问题不需要 pixel space。** 「NA 在 FM 下的行为 + per-t ERF 动态变化」是 attention 层面的问题，跟 token 代表像素还是 latent feature 无关。在 latent space 用 SiT pretrained model 做，成本更低、结论更强、代码 80% 复用。

Pixel space 是 **Phase 2 扩展**，不是 Phase 1 前提。

---

## 📐 新路线总览

```
Phase 1: Latent Space 验证（1-2 周）
  SiT-XL/2 pretrained → 替换 attention → fine-tune → 测量
  ├─ full attention ERF/distance 测量（零训练成本，立刻出图）
  ├─ NA fine-tune（k=3,5,7,11,15）
  └─ 完整 FID + GFLOPs + per-t ERF + distance 数据

         ↓  决策点：per-t ERF 差异 ≥ 2×？

  YES ──→ Phase 1.5: t-adaptive kernel 方法
  NO  ──→ Phase 1.5: NA 在 FM 下的系统对比（短文/workshop）

         ↓

Phase 2: Pixel Space 扩展（Phase 1 有结论后）
  ├─ 用 Phase 1 的结论指导 pixel space 实验设计
  ├─ 复现/对比 HDiT 的 pixel-space 结论
  └─ 或：直接拿 pretrained pixel flow model（AsymFlow）做 post-hoc

         ↓

Phase 3: 长期
  ├─ 跨架构验证（SD/Flux/PixArt — 零训练 cost hook attention）
  ├─ 256×256+ 分辨率扩展
  ├─ 理论分析（flow ODE → receptive field 需求）
  └─ 动态架构 / 可解释性工具 / 视频生成
```

---

## Phase 1 详细：Latent Space SiT

### 为什么选 SiT

| 理由 | 说明 |
|------|------|
| **DiT backbone** | 跟你现有 `models/dit.py` 架构一致，attention 替换 zero-cost |
| **Flow Matching** | 线性 interpolant，跟你的 `flow_matching.py` 一致 |
| **Pretrained 权重** | SiT-XL/2, 675M, FID ~2.x — proven model |
| **Latent space** | 32×32×4 = 1024 tokens，跟你现在 token 数一样 |
| **开源** | 官方代码直接可跑，数据 pipeline 现成 |

### Step 1: Full attention 测量（零训练成本）

```bash
# 1. Clone SiT + 加载 pretrained weights
# 2. 把你的 measure.py 移植过去
# 3. 直接 run
python measure.py --ckpt sit_xl_2_pretrained --attn_type full
```

**立刻产出**：full attention SiT 的 per-t ERF 曲线 + distance distribution。这是你整篇文章最关键的一张图——per-t ERF 到底变不变？这张图出来之前，其他都不用做。

### Step 2: NA fine-tune

```
SiT-XL/2 pretrained (full attention)
    │
    ▼ 替换 attention 为 NA
SiT-NA (k=3,5,7,11,15)
    │
    ▼ fine-tune ~5000 steps（不是从零训 200 epochs）
SiT-NA fine-tuned
    │
    ▼ 测量
FID + GFLOPs + per-t ERF + distance 完整数据
```

### Step 3: 完整实验矩阵

| Variant | Attention | Kernel | 测量 |
|---------|-----------|--------|------|
| baseline | full | — | FID, GFLOPs, ERF(t), Dist(t) |
| na3 | NA | 3 | ↑ |
| na5 | NA | 5 | ↑ |
| na7 | NA | 7 | ↑ |
| na11 | NA | 11 | ↑ |
| na15 | NA | 15 | ↑ |

---

## 🔀 决策点

```
Phase 1 数据出来后：

per-t ERF 差异 ≥ 2×
  → Story 牢靠：「FM 下不同 t 需要不同 receptive field」
  → 全速推进 t-adaptive kernel 方法
  → 目标：完整论文

per-t ERF 差异中等（1.2-2×）
  → 存在但不惊艳
  → 补跨模型验证（SD/Flux hook attention, 零成本）
  → 判断是 FM 特有还是普遍现象

per-t ERF 差异很小（< 1.2×）
  → per-t 增量不够
  → 转向 NA 在 FM 下的系统对比（HDiT 的 FM 复现+扩展）
  → 目标：短文/workshop
```

---

## ⚠️ 当前 pixel space 实验的处理

**2026-07-18：停止所有 DiT-T pixel space 训练。**

| 实验 | 状态 | 处理 |
|------|------|------|
| full attention | Epoch 35/100 | ⬜ 停掉 |
| NA k=7 | Epoch 31/100 | ⬜ 停掉 |
| NA k=11 | Epoch 15/100 | ⬜ 停掉 |
| NA k=15 | Epoch 14/100 | ⬜ 停掉 |
| NA k=3 | 已挂 | — |

**保留价值**：`pixelflow_na` 代码库作为：
- 快速原型验证工具（DiT-T 跑得动 → 逻辑没问题）
- attention/measure 模块直接搬到 SiT
- Phase 2 pixel space 实验时的基础设施

---

## 📁 代码复用计划

| 模块 | pixelflow_na（现有） | → SiT |
|------|---------------------|-------|
| `models/attention.py` | FullAttention + NeighborAttention + make_attention() | ✅ 直接搬 |
| `measure.py` | ERF + distance 测量 | ✅ 直接搬，hook 逻辑通用 |
| `models/flow_matching.py` | FM loss + ODE 采样 | 参考，SiT 官方已有 |
| `models/dit.py` | DiT backbone | SiT 官方的 DiT 替换 attention |
| `sweep.py` | 实验矩阵自动化 | ✅ 参考逻辑 |
| `analyze.py` | 可视化 | ✅ 直接搬 |
| `data/dataset.py` | 数据加载 | SiT 官方已有（ImageNet latents） |

---

## 📖 阅读优先级调整

参见 `Flow-NA-论文阅读指南.html`（待更新）。

**新优先级**：
1. 🔴 **SiT** — 理解代码结构、数据格式、怎么 fine-tune
2. 🔴 **HDiT** — 消融方法论（Table 1 模板）不变
3. 🔴 **ΔConvFusion** — ERF 测量方法不变
4. 🟡 **On Inductive Biases of DiT** — 理论支撑不变
5. 🟢 **AsymFlow / MPDiT / FreqFlow** — Phase 1 跑通后再读

---

## ⏱️ 时间线

| 阶段 | 内容 | 预计 |
|------|------|------|
| Day 1-2 | Clone SiT + 跑通 inference + 移植 attention/measure | 1-2 天 |
| Day 3 | Full attention per-t ERF + distance 测量 → 第一张关键图 | 半天 |
| Day 3-7 | 实现 NA fine-tune pipeline + 跑 5 个 NA variants | 3-5 天 |
| Day 7-10 | 完整数据分析 + 画图 | 2-3 天 |
| Day 10+ | 决策：push t-adaptive or 写短文 | — |

---

*Updated 2026-07-18 — pivot 到 latent space 优先*
