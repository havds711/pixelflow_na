# pixelflow_na — Attention Locality 测量框架

不改推理逻辑，抓取每层每个 head 的 post-softmax attention weights，计算 k×k 窗口 attention mass。

## 核心问题

> Self-attention 在扩散模型中到底有多「全局」？

前人（ΔConvFusion ICCV'25, HDiT ICML'24）在 DDPM 上发现 attention 本质是局部的。**这些结论在 Flow Matching 下还成立吗？**

→ **不成立。** FM 模型的 attention 是全局的，高度依赖架构设计。详见 [`ANALYSIS.md`](ANALYSIS.md)。

## 文档索引

| 文档 | 内容 |
|---|---|
| [`ANALYSIS.md`](ANALYSIS.md) | 完整分析：假设验证矩阵、SiT vs PixDiT 对比、后续计划 |
| [`STORY.md`](STORY.md) | 论文故事线、pitch、大纲 |
| [`ATTENTION_LOCALITY_GUIDE.md`](ATTENTION_LOCALITY_GUIDE.md) | 接入新模型操作手册（5步） |
| [`SETUP.md`](SETUP.md) | 服务器环境搭建 + 新模型参数速查 |

## 快速结果

| 指标 | SiT-XL/2 (FM) | PixDiT (FM) |
|---|---|---|
| k=1 mean mass | 1.8% | 9.1% |
| k=7 >80% | 13.0% | 24.7% |
| k=15 >80% | 93.7% | 92.1% |
| 推荐 k | **15** | **15** |

## 环境

按模型系列隔离：

| 模型系列 | env | 状态 |
|---|---|---|
| SiT (B/L/XL) | `natten` | ✅ 已有 |
| PixDiT | `pixel` | ✅ 已有 |
| DiT (DDPM) | `dit` | 🔴 待建 |
| MDTv2 | `mdt` | 🟡 待建 |

## 文件结构

```
pixelflow_na/
├── README.md
├── ANALYSIS.md                    # 完整分析报告
├── STORY.md                       # 论文故事线
├── ATTENTION_LOCALITY_GUIDE.md    # 接入新模型操作手册
│
├── measure/                       # 测量脚本
│   ├── measure_locality_sit.py
│   └── measure_locality_pixeldit.py
├── analysis/                      # 分析脚本
│   ├── analyze_sit.py
│   └── analyze_pixeldit.py
├── experiments/                   # 辅助实验
│   ├── measure_sit.py             # 快速 ERF (随机 latent)
│   ├── finetune_sit_na.py
│   └── precompute_latents.py
│
├── models/                        # DiT + FM 基础模块
├── SiT/                           # SiT 官方代码 + environment.yml
├── pixeldit/                      # PixDiT 自包含实现
├── utils/  data/
└── outputs/                       # .pt 数据文件
```

## 接入新模型

参见 [`ATTENTION_LOCALITY_GUIDE.md`](ATTENTION_LOCALITY_GUIDE.md)，标准 5 步，输出统一格式到 `outputs/attention_locality_{模型名}.pt`。
