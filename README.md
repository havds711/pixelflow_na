# pixelflow_na — Diffusion Models Attention Locality 测量框架

在 pretrained diffusion models 的推理过程中，不改推理逻辑，抓取每层每个 head 的 post-softmax attention weights，计算不同 k×k 窗口内的 attention 质量累积。

## 核心问题

> Self-attention 在扩散模型中到底有多「全局」？

前人（ΔConvFusion, PiT, HDiT）在 DDPM / SDE 扩散上发现 attention 本质是局部的（ERF < 15, 99% 交互 ≤ 6）。**这些结论在 Flow Matching / ODE 扩散下还成立吗？**

## 架构

```
                        ┌──────────────────────────┐
                        │    统一的测量接口            │
                        │  monkey-patch attention    │
                        │  k×k 窗口 mass 计算         │
                        │  统一输出格式 (.pt)          │
                        └──────┬───────────────────┘
                               │
          ┌────────────────────┼────────────────────────┐
          │                    │                         │
     ┌────▼─────┐      ┌──────▼──────┐          ┌──────▼──────┐
     │  SiT 系列  │      │  PixDiT     │          │  DiT / MDT  │
     │  B/L/XL   │      │  dual-stream│          │  SD3 / MAR  │
     │  conda:sit│      │  conda:pixel│          │  各自 env    │
     └──────────┘      └─────────────┘          └─────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  测量流程                                                        │
│                                                                  │
│  x_T (noise) ──→ Euler ODE ──→ x_0 (image)                      │
│                     │                                             │
│                model(x,t,y)                                      │
│                     │                                             │
│              attention: QK^T/√d → softmax → ×V                    │
│                                    │                              │
│                              抓取 weights                         │
│                                    │                              │
│                   ┌────────────────┼──────────────────┐          │
│                   │                │                   │          │
│              k=1 窗口         k=7 窗口            k=15 窗口        │
│              mass=?           mass=?              mass=?          │
│                   │                │                   │          │
│                   └────────────────┼──────────────────┘          │
│                                    │                              │
│                          min_k: 最小k使mass>80%                    │
│                                    │                              │
│                         保存 → attention_locality_*.pt            │
└─────────────────────────────────────────────────────────────────┘
```

### 环境隔离

不同模型的代码源和依赖完全不同，按源分环境：

| 模型系列 | conda env | 说明 |
|---|---|---|
| SiT (B/L/XL) | `sit` | 已有 `SiT/environment.yml`, PyTorch≥1.13 |
| PixDiT | `pixel` | 独立实现, pixeldit/ 目录自包含 |
| DiT (B/L/XL) | `dit` | Meta 原版, 依赖类似但独立 |
| MDTv2 | `mdt` | 独立 repo |
| MMDiT / SD3 | `mmdit` | diffusers≥0.31, transformers新版 |
| FlowAR / MAR | `flowar` | 完全不同范式 |

隔离原因：diffusers 版本差异最大（SD3≥0.31 vs SiT 旧版），transformer/torch/CUDA 版本各模型敏感。

```bash
# SiT (已有)
conda env create -f SiT/environment.yml

# 其他模板
conda create -n dit python=3.10 -y && conda activate dit
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
git clone <repo> && cd <repo> && pip install -e .
```

## 方法

标准推理不改逻辑。只在 softmax 之后抓 attention weights，计算每个 query 的 k×k 窗口内累积 mass。对每个 query token，取 k×k 窗口（边界平移，确保每个 query 总是 k² 个 key），累加 attention mass。k = 1,3,5,7,9,11,13,15。

## 已测模型

| 模型 | 框架 | 推理 | Grid | Layers | 数据 |
|---|---|---|---|---|---|
| SiT-XL/2 | Flow Matching | ODE 20步 | 16×16 (256) | 28 | `outputs/attention_locality/attention_locality.pt` |
| PixDiT-XL c2i | Flow Matching | Euler 25步 | 16×16 (256) | 30 | `outputs/pixeldit_attention_locality/attention_locality_pixeldit.pt` |

### 关键结果

| 指标 | SiT-XL/2 | PixDiT c2i |
|---|---|---|
| k=1 mean mass | 1.8% | 9.1% |
| k=7 >80% 达标率 | 13.0% | 24.7% |
| k=15 >80% 达标率 | 93.7% | 92.1% |
| 深度模式 | Reverse (浅全局→深局部) | 中层最局部 |
| ODE时间依赖 | 几乎无 | 晚期更局部 |
| 图像依赖 | 几乎无 | 有显著差异 |
| NA-friendly heads | 11% | 19% |
| 推荐 k | **15** | **15** |

**结论: Attention Locality 不是 FM 通用属性, 高度依赖架构设计。详情见 `ANALYSIS.md`。**

## 文件结构

```
pixelflow_na/
├── README.md
├── ANALYSIS.md                            # 完整分析报告 (SiT + PixDiT + 后续计划)
├── ATTENTION_LOCALITY_GUIDE.md            # 接入新模型的步骤指南
├── ATTENTION_LOCALITY_README.md           # 测量结果快速参考
│
├── measure_locality_sit.py                # SiT-XL/2 测量
├── measure_locality_pixeldit.py           # PixDiT c2i 测量
├── analyze_data.py                        # SiT 数据分析脚本
├── analyze_pixeldit.py                    # PixDiT 数据分析脚本
│
├── measure_sit.py                         # SiT 快速 ERF 测量 (随机 latent)
├── finetune_sit_na.py                     # SiT NA fine-tune
├── precompute_latents.py                  # VAE 预提取 latents
│
├── models/                                # DiT + Flow Matching 基础模块
│   ├── attention.py
│   ├── dit.py
│   └── flow_matching.py
│
├── SiT/                                   # SiT 官方代码
│   ├── models.py
│   ├── train.py
│   ├── download.py
│   └── environment.yml
│
├── pixeldit/                              # PixDiT 自包含实现 + 测量
│   ├── measure_attention_locality.py
│   └── pixdit_core/
│       ├── modules.py                     # RotaryAttention, RoPE, adaLN
│       └── pixeldit_c2i.py                # PixDiT 完整模型 (patch+ pixel dual-stream)
│
├── utils/
├── data/
└── outputs/
    ├── attention_locality/
    │   └── attention_locality.pt          # SiT 184 MB
    └── pixeldit_attention_locality/
        └── attention_locality_pixeldit.pt # PixDiT 247 MB
```

## 数据格式

每个 `.pt` 文件统一结构：

```python
{
    'masses':    [steps, batch, layers, heads, 256, 8],  # float16
    'min_k_80':  [steps, batch, layers, heads, 256],      # uint8 (最小 k 使 mass>0.8)
    'min_k_50':  ...,
    'min_k_90':  ...,
    'min_k_95':  ...,
    'min_k_99':  ...,
    't_schedule': [steps],                                 # float32
}
```

k_values = [1, 3, 5, 7, 9, 11, 13, 15]

## 接入新模型

参见 `ATTENTION_LOCALITY_GUIDE.md`，标准 5 步：

1. 确认模型基本信息（采样方式、t 范围、grid、层数、heads）
2. 找到 attention 代码，判断是手动实现还是融合算子，对应抓取/replace
3. 从原仓库复制采样逻辑，**不自己发明**
4. 接入统一的 k×k 窗口计算
5. 输出统一格式到 `outputs/attention_locality_{模型名}.pt`

## 相关资源

- ΔConvFusion: Zotero `JGZ5NGCH`
- SiT: Zotero `97FPMY55`
- "Can We Achieve Efficient Diffusion without Self-Attention?" — 蒸馏 attention 到卷积
