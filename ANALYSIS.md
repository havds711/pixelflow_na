# Attention Locality in Diffusion Transformers — 完整分析报告

> 数据: `outputs/` 下 6 个模型的 attention locality 测量
> 实验脚本: `measure/measure_locality_*.py`
> 分析脚本: `analysis/analyze_all.py` (统一分析), `analysis/analyze_sit.py`, `analysis/analyze_pixeldit.py`
> 日期: 2026-07-19 (更新)

---

## 1. 模型总览

| 模型 | 架构 | 框架 | 采样 | 层数 | 头数 | 参数量 | 状态 |
|---|---|---|---|---|---|---|---|
| **SiT-XL/2** | DiT + FM (Linear) | FM | ODE Euler 20步, t=1→0 | 28 | 16 | 675M | ✅ |
| **SiT-B/2** | DiT + FM (Linear) | FM | ODE Euler 20步, t=1→0 | 12 | 12 | 130M | ✅ |
| **SiT-L/2** | DiT + FM (Linear) | FM | ODE Euler 20步, t=1→0 | 24 | 16 | 450M | ✅ |
| **PixDiT** | Dual-stream DiT | FM | Euler 25步, t=0→1 | 30 (26p+4px) | 16 | ~675M | ✅ |
| **DiT-XL/2** | DiT + DDPM | DDPM | DDPM 250步, t=999→0 | 28 | 16 | 675M | ✅ |
| **MDTv2-XL/2** | Masked DiT | DDPM | DDPM 250步, t=999→0 | 29 (28b+1s) | 16 | ~675M | ✅ |

---

## 2. 全局对比：k×k Attention Locality

### 2.1 核心指标：k=7 >80% 达标率（越高越局部）

| 排名 | 模型 | k=7 >80% | 可视化 |
|---|---|---|---|
| #1 | **PixDiT** | **24.69%** | ████████████████████████ |
| #2 | **MDTv2-XL/2** | **19.70%** | ███████████████████ |
| #3 | **SiT-B/2** | **19.10%** | ███████████████████ |
| #4 | **DiT-XL/2** | 13.33% | █████████████ |
| #5 | **SiT-L/2** | 13.23% | █████████████ |
| #6 | **SiT-XL/2** | 13.01% | █████████████ |

**关键发现: PixDiT 的 dual-stream 设计使其 attention 比最全局的 SiT-XL/2 局部近 2 倍。**

### 2.2 完整 k-达标率表

| k | SiT-XL/2 | SiT-B/2 | SiT-L/2 | PixDiT | DiT-XL/2 | MDTv2-XL/2 |
|---|---|---|---|---|---|---|
| k=1 | 0.2% | 0.2% | 0.2% | **3.2%** | 0.4% | 0.9% |
| k=3 | 8.7% | **14.9%** | 9.8% | 11.9% | 7.8% | 10.7% |
| k=5 | 10.7% | **17.0%** | 11.6% | 18.8% | 10.2% | 15.0% |
| k=7 | 13.0% | 19.1% | 13.2% | **24.7%** | 13.3% | 19.7% |
| k=9 | 16.0% | 22.6% | 15.3% | **31.9%** | 17.5% | 25.3% |
| k=11 | 20.6% | 28.0% | 19.0% | **41.5%** | 23.8% | 33.1% |
| k=13 | 34.5% | 43.8% | 31.3% | **57.4%** | 39.5% | 49.4% |
| k=15 | 93.7% | **96.6%** | 96.4% | 92.1% | 96.0% | 92.4% |

---

## 3. 深度模式：Reverse Locality

| 模型 | 浅层 >80%@k=7 | 深层 >80%@k=7 | 模式 |
|---|---|---|---|
| **SiT-XL/2** | 3.6% | 17.9% | ✅ Reverse (深>浅, 5.0×) |
| **SiT-B/2** | 14.8% | 19.6% | ✅ Reverse (深>浅, 1.3×) |
| **SiT-L/2** | 1.1% | 14.2% | ✅ Reverse (深>浅, 12.5×) |
| **PixDiT** | 20.1% | 17.8% | ❌ Normal (浅≈深) |
| **DiT-XL/2** | 14.6% | 13.2% | ❌ Normal (浅>深) |
| **MDTv2-XL/2** | 17.8% | 18.2% | ~ Neutral (浅≈深) |

**结论: Reverse Locality 是 SiT 系列（FM + Linear schedule）的特有属性，与框架强相关。DDPM 模型（DiT, MDTv2）没有此现象。**

---

## 4. 时间步依赖

| 模型 | 早期 >80%@k=7 | 晚期 >80%@k=7 | Δ | 模式 |
|---|---|---|---|---|
| SiT-XL/2 | 13.3% | 13.0% | -0.3% | ✓ 稳定 |
| SiT-B/2 | 19.9% | 18.0% | -1.9% | ✓ 稳定 |
| SiT-L/2 | 13.6% | 12.6% | -1.0% | ✓ 稳定 |
| **PixDiT** | 20.3% | **27.9%** | **+7.7%** | ⚠ 晚期更局部 |
| **DiT-XL/2** | 6.1% | **17.6%** | **+11.5%** | ⚠ 强烈趋势 |
| **MDTv2-XL/2** | 17.8% | 21.1% | +3.2% | ⚠ 轻微趋势 |

**结论: FM 模型（ODE 20步）时间步稳定；DDPM 模型（250步）attention 随去噪逐步聚焦局部。**

---

## 5. 图像内容依赖

| 模型 | 4图 spread | 模式 |
|---|---|---|
| SiT-XL/2 | 0.0038 | ✓ 基本独立 |
| SiT-B/2 | 0.0041 | ✓ 基本独立 |
| SiT-L/2 | 0.0026 | ✓ 基本独立 |
| **PixDiT** | **0.0526** | ⚠ 显著依赖 |
| DiT-XL/2 | 0.0103 | ✓ 基本独立 |
| MDTv2-XL/2 | 0.0041 | ✓ 基本独立 |

**结论: 只有 PixDiT 的 attention locality 对输入内容敏感，可能跟 dual-stream 设计有关。**

---

## 6. 规模效应（SiT 系列 B → L → XL）

| 模型 | 层数 | 头数 | k=7 >80% | Head mean k |
|---|---|---|---|---|
| **SiT-B/2** | 12 | 12 | **19.1%** | 12.03 |
| SiT-L/2 | 24 | 16 | 13.2% | 12.95 |
| SiT-XL/2 | 28 | 16 | 13.0% | 12.88 |

**反直觉发现: 最小的 SiT-B 反而最局部。不是"越大越局部"，而是"B 的 12 头设计让 attention 更聚焦"。L 和 XL 几乎一样。**

---

## 7. FM vs DDPM

| 框架 | 模型 | k=7 >80% |
|---|---|---|
| FM | SiT-XL, SiT-B, SiT-L, PixDiT | 平均 17.5% |
| DDPM | DiT-XL, MDTv2-XL | 平均 16.5% |

**结论: FM 和 DDPM 的整体 attention locality 差异很小（Δ=1%）。架构差异（PixDiT dual-stream）远大于训练框架差异。**

---

## 8. Per-Head NA 可行性

### 8.1 NA-Friendly Heads（mean min_k_80 ≤ 7）

| 模型 | NA-Friendly Heads | 占比 |
|---|---|---|
| **PixDiT** | 91/480 | **19.0%** |
| **SiT-B/2** | 27/144 | **18.8%** |
| **MDTv2-XL/2** | 77/464 | 16.6% |
| SiT-L/2 | 47/384 | 12.2% |
| DiT-XL/2 | 51/448 | 11.4% |
| SiT-XL/2 | 50/448 | 11.2% |

### 8.2 Per-Head NA 理论 FLOPs 节省（覆盖 50% query）

| 模型 | 加权平均 k | FLOPs 节省 |
|---|---|---|
| **SiT-B/2** | 12.7 | **36.9%** |
| DiT-XL/2 | 13.2 | 32.1% |
| SiT-L/2 | 13.3 | 31.0% |
| MDTv2-XL/2 | 14.3 | 20.1% |
| SiT-XL/2 | 14.5 | 18.0% |
| PixDiT | 14.7 | 16.1% |

**结论: SiT-B 由于头数少（12头）+ 相对局部，per-head NA 理论收益最高。但整体来看，pretrained 模型上直接做 per-head NA 收益有限（16-37%）。**

---

## 9. 文件结构

```
pixelflow_na/
├── README.md
├── ANALYSIS.md                         # 本文档 — 完整分析报告
├── ATTENTION_LOCALITY_GUIDE.md         # 接入新模型操作手册
├── SETUP.md                            # 环境搭建说明
├── STORY.md                            # 论文故事线
│
├── checkpoints/                        # 模型权重
│   ├── DiT-XL-2-256x256.pt            (2.7 GB)
│   ├── mdt_xl2_v1_ckpt.pt             (2.8 GB)
│   ├── SiT-B-2-256.pt                 (522 MB)
│   ├── SiT-L-2-256.pt                 (1.8 GB)
│   └── SiT/checkpoints/SiT-XL-2-256.pt
│
├── measure/                            # 测量脚本
│   ├── measure_locality_sit.py         # SiT B/L/XL 共用
│   ├── measure_locality_pixeldit.py    # PixDiT
│   ├── measure_locality_dit.py         # DiT-XL
│   └── measure_locality_mdt.py         # MDTv2-XL
│
├── analysis/                           # 分析脚本 & 报告
│   ├── analyze_all.py                  # 统一分析脚本 (新)
│   ├── analyze_sit.py                  # SiT-XL 单独分析
│   └── analyze_pixeldit.py             # PixDiT 单独分析
│
├── outputs/                            # 测量数据 + 分析报告
│   ├── attention_locality_sit.pt              # SiT-XL (193 MB)
│   ├── attention_locality_sit_b/              # SiT-B  (62 MB)
│   ├── attention_locality_sit_l/              # SiT-L  (165 MB)
│   ├── attention_locality_pixeldit.pt         # PixDiT  (258 MB)
│   ├── attention_locality_dit_xl/             # DiT-XL  (2.4 GB)
│   ├── attention_locality_mdtv2_xl/           # MDTv2  (2.5 GB)
│   └── analysis/                              # 分析报告
│       ├── report_sit_xl.txt
│       ├── report_sit_b.txt
│       ├── report_sit_l.txt
│       ├── report_pixeldit.txt
│       ├── report_dit_xl.txt
│       ├── report_mdt_xl.txt
│       └── report_cross_model.txt             # 跨模型对比
│
├── models/                             # DiT + FM 基础模块
├── SiT/                                # SiT 官方代码
├── pixeldit/                           # PixDiT 实现
├── dit_repo/                           # Meta DiT 官方代码
├── mdt_repo/                           # MDT 官方代码
├── utils/  data/  vae/
└── experiments/                        # 辅助实验脚本
```

---

## 10. 关键结论汇总

| # | 假设/问题 | 结论 | 证据 |
|---|---|---|---|
| 1 | FM 模型 attention 偏向全局？ | **部分成立** — SiT 系列全局，但 PixDiT (也是FM) 显著更局部 | PixDiT k=7: 24.7% vs SiT-XL: 13.0% |
| 2 | Reverse Locality 跨模型一致？ | **不成立** — 仅 SiT 系列有此现象 | DiT/MDT/PixDiT 无 reverse |
| 3 | Head 多样性 5× 普遍存在？ | **确认** — 所有模型都有显著的 head 间差异 | 层内 head std: 2.4-5.3 |
| 4 | 规模效应 (更大→更局部?) | **反直觉** — SiT-B (最小) 最局部 | B(19.1%) > L(13.2%) ≈ XL(13.0%) |
| 5 | ODE 轨迹上 locality 稳定？ | **FM 稳定，DDPM 不稳定** | FM std<0.01, DDPM 有明显晚期聚焦趋势 |
| 6 | 内容独立性？ | **仅 PixDiT 不独立** | PixDiT spread=0.053, 其他 <0.01 |
| 7 | FM vs DDPM 本质区别？ | **差异小** (Δ=1%) — 架构 > 训练框架 | PixDiT (FM) > MDT (DDPM) > SiT (FM) ≈ DiT (DDPM) |
| 8 | Per-Head NA 可行性？ | **pretrained 模型直接替换收益有限** (16-37%) | 需要从零训练可学习 NA |

---

## 11. 开放问题

- [ ] PixDiT 的 image-dependence — 不同内容导致 5pp 差异的机制？
- [ ] SiT-B 为什么比 SiT-L/XL 更局部？是头数差异 (12 vs 16) 还是深度差异？
- [ ] 文本条件 (MMDiT/SD3) 的 attention locality 会因 prompt 变化吗？
- [ ] 更大的 token 数 (如 32×32=1024) 下 locality 怎么变化？
- [ ] Fine-tuned model（如经过 NA fine-tune）的 locality 会变化吗？
- [ ] 从零训练 Per-Head Adaptive NA 后 attention 会如何重新分布？
- [ ] MDTv2 的 side block (L28) attention 为什么和其他 block 行为不同？
