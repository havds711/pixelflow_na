# Paper Storyline: "Everything We Knew About Attention Locality in Diffusion Was DDPM-Specific"

---

## 故事弧线

```
DDPM时代: "Attention是局部的!" ──→ FM时代: "不对, 是全局的" ──→ 为什么? ──→ 怎么办?
```

---

## Act 1 — The Established Truth (背景)

**前人共识: Diffusion models 的 self-attention 本质是局部的。**

| 论文 | 框架 | 核心发现 |
|---|---|---|
| ΔConvFusion (2024) | DiT-XL/2, DDPM | ERF < 15, 99% 交互距离 ≤ 6 tokens |
| PiT (2024) | DiT, DDPM | Attention 可以用卷积替代, 质量不降 |
| HDiT (2024) | DiT, DDPM | 高分辨率下 attention locality 更强 |

**这些论文的结论被广泛接受, 并催生了大量 efficient attention 工作:**
- Neighborhood Attention (NA) 替换全局 attention
- Sparse attention / dilated attention
- Attention → Conv 蒸馏
- 所有验证都在 **DDPM / SDE 框架**下完成

**社区默认假设: "Attention locality 是 diffusion models 的通用属性, 与 training framework 无关。"**

---

## Act 2 — The Gap (我们发现的问题)

**Flow Matching 已成为主流, 但没人重新验证过 attention locality。**

| 事件 | 时间 |
|---|---|
| DDPM (Ho et al.) | 2020 |
| DiT + DDPM | 2023 |
| ΔConvFusion / PiT / HDiT 发现 attention 局部 | 2024 |
| **Flow Matching 崛起** (SD3, Flux, SiT, etc.) | **2024** |
| **FM 下 attention locality → 无人验证** | ← **我们** |

DDPM 和 FM 有本质区别:
- DDPM: 预测噪声 ε, SDE 轨迹, 单步变化微小
- FM: 预测速度场 v=dx/dt, ODE 轨迹, 直连噪声和数据
- FM 的 ODE 路径更直接 → attention 可能需要更全局的信息?

---

## Act 3 — The Evidence (我们的数据)

**两个 FM 模型一致显示: attention 是全局的, 不是局部的。**

### 核心对比: DDPM 结论 vs FM 实测

| 前人结论 (DDPM) | 我们的测量 (FM) |
|---|---|
| ERF < 15, "attention is local" | SiT k=15 才到 93.7%, PixDiT k=15 才到 92.1% |
| 99% 交互 ≤ 6 tokens | SiT k=7 仅 13% 达标, 99% mass 仅 14% query 在 k≤15 达标 |
| "可用卷积替代" | NA k=7 只能保留 13-25% attention mass |
| Locality 无时间依赖 | PixDiT 早期→晚期 locality 翻倍 (15.8%→29.0%) |
| Locality 无内容依赖 | PixDiT 4 图 spread=5.3pp (SiT 仅 0.4pp) |

### Hard numbers for the paper

| Metric | SiT-XL/2 (FM) | PixDiT (FM) | ΔConvFusion on DiT (DDPM) |
|---|---|---|---|
| k=1 mass | 1.8% | 9.1% | ? (claim: "local") |
| k=7 >80% | 13.0% | 24.7% | ? (likely >>50% if "local") |
| k=15 >80% | 93.7% | 92.1% | ? |
| 推荐 k | **15** | **15** | **"≤7"** |

**关键数据缺口**: 我们需要在 DiT-XL/2 (DDPM) 上跑完全相同的流程, 以建立直接对比。如果 DiT (DDPM) 确实 k=7 就能 >80%, 故事就完整了。

### 额外发现

1. **Reverse Locality** (SiT): 浅层比深层更全局 — 与 CNN 完全相反
2. **架构差异**: PixDiT 比 SiT 局部 2× (dual-stream 设计)
3. **Head 多样性**: 同层 head 间 locality 差异可达 5× — 所有模型共有
4. **PixDiT 有 ODE 时间演化**: 去噪后期 attention 逐步聚焦局部
5. **PixDiT 有内容敏感性**: 不同图像类别 locality 差异显著

---

## Act 4 — Why Would This Happen? (分析)

**猜想 1: ODE vs SDE 路径差异**
- DDPM (SDE): 每步加噪声再降噪, 随机路径曲折 → attention 倾向于局部 (每步只需小修正)
- FM (ODE): 确定性直线路径, 噪声→图像一步到位 → attention 需要全局 (每步做大幅度调整)

**猜想 2: 预测目标差异**
- DDPM (ε-prediction): 预测噪声, 是局部操作 (噪声是 i.i.d. per pixel)
- FM (v-prediction): 预测速度场, 是全局操作 (需要知道整体结构才能给出正确方向)

**猜想 3: 训练信号差异**
- FM 的 loss 直接监督速度场, 信号更强 → 模型学到更全局的依赖
- DDPM 的 loss 间接监督, 模型倾向于保守的局部策略

**待验证**: 需要 DiT (DDPM) 实测数据来区分这些猜想。

---

## Act 5 — Implications (影响)

### 对 Efficient Attention 研究的影响

如果 FM attention 确实全局:
- **所有在 DDPM 上验证的 efficient attention 方法, 在 FM 上需要重新评估**
- NA / sparse attention / Conv 蒸馏 → 可能不适合 FM 模型
- 需要新的 efficient attention 策略 (per-head adaptive? hybrid global-local?)

### 对架构设计的影响

- PixDiT 的 dual-stream 可能是正确的方向 — 用额外的 pixel stream 分担局部细节, 释放 patch attention 的全局能力
- 不要试图在 FM 模型中强行缩小 attention window — 模型确实在用全局信息

### 对社区的影响

- FM 已成为主流 (SD3, Flux, Sora 等), 但 attention locality 分析仍基于 DDPM 时代结论
- 这是一个 **"纠正社区认知"** 的机会

---

## One-Sentence Pitch

> ΔConvFusion 等论文在 DDPM 上发现 diffusion attention "惊人地局部"，但我们首次在 Flow Matching 模型上做了同样的测量，发现 attention **惊人地全局** — 这意味着整个 efficient diffusion attention 研究方向可能需要重新校准。

---

## 论文结构建议

```
1. Introduction
   - DDPM时代: attention是局部的 (ΔConvFusion, PiT, HDiT)
   - FM成为主流, 没人重新验证
   - 我们测了, 推翻了

2. Background
   - DDPM vs Flow Matching
   - Attention locality measurement methodology

3. Method
   - Monkey-patch测量框架
   - k×k窗口 + min_k metric
   - 统一输出格式

4. Experiments
   4.1 SiT-XL/2: attention is global
   4.2 PixDiT: more local but still global
   4.3 [待完成] DiT-XL/2 (DDPM): confirming DDPM is local
   4.4 Cross-model comparison
   4.5 Head diversity, depth pattern, time evolution, content sensitivity

5. Analysis: Why FM ≠ DDPM
   - ODE vs SDE
   - v-prediction vs ε-prediction

6. Implications
   - Efficient attention in FM era
   - Architecture design for FM

7. Limitations & Future Work
```

---

## 需要的补充实验

| 优先级 | 实验 | 目的 |
|---|---|---|
| 🔴 P0 | DiT-XL/2 (DDPM) 测量 | 建立 DDPM 基准, 完成对比链条 |
| 🟡 P1 | SiT-B/2, SiT-L/2 | 规模效应 |
| 🟡 P1 | DiT trained with FM loss | 区分架构 vs 训练方式 |
| 🟢 P2 | MDTv2 | 不同 attention variant |
| 🟢 P2 | More seeds / more images | 统计显著性 |
