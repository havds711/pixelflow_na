# pixelflow_na — 构建文档

> 2026-07-18 · Post-hoc ERF 分析，零训练

---

## 项目目标

验证 **Flow Matching 下 self-attention 的局部性**：
- 复现 ΔConvFusion (ICCV 2025) 的 ERF 测量方法
- 在 FM 模型（SiT 等）上 post-hoc 测量
- 核心问题：FM 下 self-attention 和 DDPM 下一样局部吗？

---

## 技术选型

| 选择 | 理由 |
|------|------|
| **Post-hoc 分析** | ΔConvFusion 方法本身就是纯后验的，不需要训练 |
| **SiT-XL/2 pretrained** | DiT backbone + Flow Matching + 开源权重 + 675M |
| **Latent space (32×32)** | SiT 原生 latent，patch=2 → 256 tokens → attention 提取极快 |
| **DFT + Butterworth 高通** | ΔConvFusion §3.2 原版方法，scikit-image 有 NumPy 参考 |
| **NATTEN + PyTorch fallback** | 仅测量时需要（将来如果做 NA 实验），目前用不到 |
| **随机 latent 输入** | ERF 是 attention 模式的属性，不依赖真实图像内容 |

---

## 代码结构

```
pixelflow_na/
├── README.md                    # 研究提案 + 论文阅读指南
├── PLAN.md                      # 技术实现计划
├── ROADMAP.md                   # 研究路线图（活跃）
├── BUILD.md                     # 本文件
├── requirements.txt
│
├── erf_deltaconv.py             # 🆕 ΔConvFusion ERF 方法实现
│                                #   DFT + Butterworth 高通 + ASM + 80% 阈值
│
├── measure_sit.py               # SiT ERF/distance 测量（简化版，备用）
├── finetune_sit_na.py           # ⚠️ 已废弃 — NA fine-tune（保留供参考）
├── precompute_latents.py        # ⚠️ 已废弃 — VAE 预编码（保留供参考）
│
├── models/                      # 核心模型（Phase 2 pixel space 基础设施）
│   ├── attention.py             # FullAttention + NeighborAttention + make_attention()
│   ├── dit.py                   # DiT backbone
│   └── flow_matching.py         # FlowMatchingTrainer + sample_ode()
│
├── SiT/                         # SiT 模型 + 工具
│   ├── models.py                # SiT 定义 + get_attention_weights()
│   ├── attention.py             # FullAttention + NeighborAttention
│   ├── measure.py               # SiT ERF 测量（备用）
│   ├── train.py                 # 原始 SiT 训练脚本
│   ├── download.py              # 自动下载 pretrained weights
│   └── checkpoints/
│       └── SiT-XL-2-256.pt      # Pretrained SiT-XL/2 (~2.6 GB)
│
├── vae/                         # SD VAE (kl-f8, ft-MSE)
│
├── data/
│   └── dataset.py               # 数据加载（备用）
│
├── utils/
│   └── fid.py                   # FID 计算（备用）
│
├── analyze.py                   # 结果可视化（可复用）
├── sweep.py                     # ⚠️ 已废弃 — kernel sweep
├── train.py                     # ⚠️ 已废弃 — DiT 训练
│
└── results/                     # 实验结果
    ├── sit_measure/             # Full attention baseline（简化版方法）
    └── sit_measure_na*/         # NA variants（⚠️ 旧数据，来自废弃路线）
```

---

## ΔConvFusion ERF 测量方法

### 步骤

```
1. 提取 Attention Map
   pretrained SiT → 随机 latent 推理 → get_attention_weights()
   → 每层产出 [B, heads, N, N] post-softmax attention

2. 高频分离
   for each query position (x_i, y_j):
       A_2d = attn[q].reshape(H, W)           # 2D attention map
       F = fft2(A_2d)                          # DFT
       F_high = F * butterworth_highpass(H,W)  # 高通滤波
       Lambda = ifft2(F_high).real             # 高频 attention map

3. ASM 计算
   for K in 1..max_k:
       ASM(K) = Σ Lambda in K×K window around query
   归一化: ASM(K) / ASM(max_k)

4. ERF 确定
   erf = min{K | ASM(K) >= 0.8}
```

### 关键参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| Butterworth cutoff | 0.1~0.2 (归一化频率) | 高通截止频率 |
| Butterworth order | 2 | 滤波器阶数 |
| max_k | grid_size (16) | 最大窗口 |
| ASM 阈值 | 0.8 (80%) | ERF 定义 |
| n_samples | 16-32 | 测量样本数 |

---

## 快速开始

```bash
# 安装依赖（如果还没有）
pip install -r requirements.txt
pip install timm diffusers

# Step 1: 验证环境
python -c "from SiT.models import SiT_XL_2; print('SiT OK')"

# Step 2: ΔConvFusion ERF 测量
python erf_deltaconv.py --device cuda:0 --n_samples 16

# Step 3: 可视化
python analyze.py --results results/sit_erf_deltaconv/sit_erf_full.json
```

---

## 已知限制

1. **token 数偏少**：SiT-XL/2 latent=32×32, patch=2 → 16×16 grid = 256 tokens。ERF 可能不够有区分度（vs PixArt 的更大分辨率）
2. **单一模型**：目前只有 SiT，后续需 Flux/SD3 做跨模型验证
3. **Butterworth 参数需调优**：cutoff 的选择影响 ERF 绝对值，需要与文献方法对齐
4. **随机 latent vs 真实 latent**：ERF 是 attention 结构属性，不应依赖输入分布，但需验证

---

*Updated 2026-07-18 — pivot 到 post-hoc ERF 分析，废弃训练路线*
