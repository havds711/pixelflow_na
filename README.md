# pixelflow_na — Diffusion Models Attention Locality 测量框架

在 pretrained diffusion models 的推理过程中，不改推理逻辑，抓取每层每个 head 的 post-softmax attention weights，计算不同 k×k 窗口内的 attention 质量累积。

## 核心问题

> Self-attention 在扩散模型中到底有多「全局」？

前人（ΔConvFusion, PiT, HDiT）在 DDPM / SDE 扩散上发现 attention 本质是局部的（ERF < 15, 99% 交互 ≤ 6）。**这些结论在 Flow Matching / ODE 扩散下还成立吗？**

## 方法

```
标准推理:  x_t → model(x, t, y) → 下一帧
              └── attention: QK^T/√d → softmax → ×V
                                                └── 这里多抓一份 weights

不改推理逻辑。只在 softmax 之后抓 attention weights，计算每个 query 的 k×k 窗口内累积质量。
```

对每个 query token，取 k×k 窗口（边界平移，确保每个 query 总是 k² 个 key），累加 attention mass。k = 1,3,5,7,9,11,13,15。

## 已测模型

| 模型 | 框架 | 推理 | Grid | Layers | 数据 |
|------|------|------|------|--------|------|
| SiT-XL/2 | Flow Matching | ODE 20步 | 16×16 (256) | 28 | `outputs/attention_locality_sit.pt` |
| PixDiT-XL c2i | Flow Matching | Euler 25步 | 16×16 (256) | 30 | `outputs/attention_locality_pixeldit.pt` |

### 关键结果

| 指标 | SiT-XL/2 | PixDiT c2i |
|------|:---:|:---:|
| k=15 >80% 达标率 | 93.7% | 92.1% |
| k=7 >80% 达标率 | 13.0% | 24.7% |
| 推荐 k | **15** | **15** |

**结论：两个 FM 模型 attention 本质全局。k=15 才能覆盖 >90%。NA 小 kernel 无法直接替换。**

## 文件结构

```
pixelflow_na/
├── README.md                             # 总览
├── ATTENTION_LOCALITY_GUIDE.md           # 接入新模型的步骤指南
├── measure_locality_sit.py               # SiT-XL/2 测量脚本 (conda natten)
├── measure_locality_pixeldit.py          # PixDiT c2i 测量脚本 (conda pixel)
├── measure_sit.py                        # SiT 快速 ERF 测量（随机 latent）
├── finetune_sit_na.py                    # SiT NA fine-tune
├── precompute_latents.py                 # VAE 预提取 latents
├── models/                               # DiT + Flow Matching 基础模块
├── SiT/                                  # SiT 官方代码
├── vae/                                  # VAE 权重
└── outputs/
    ├── attention_locality_sit.pt         # 184 MB
    └── attention_locality_pixeldit.pt    # 247 MB
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
