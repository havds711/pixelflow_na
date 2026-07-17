# pixelflow_na — 构建文档

> 2026-07-17 · 代码框架构建记录

---

## 项目目标

验证 **Neighborhood Attention (NA) 在 Flow Matching 下的行为**：
- 复现 HDiT / ΔConvFusion / PiT 在 DDPM 上的测量方法
- 在 Flow Matching（线性 interpolant）场景下重新测量
- 核心问题：DDPM 上 NA 结论（ERF < 15, 99% token 交互 ≤ 6）在 Flow Matching 下还成立吗？

---

## 技术选型

| 选择 | 理由 |
|------|------|
| **64×64 分辨率** | 32×32 grid（1024 tokens），单卡 3090 可训练 |
| **DiT-T backbone** | dim=192, depth=6, heads=3 (~4.5M)，快速实验验证 |
| **DiT-S backbone** | dim=384, depth=12, heads=6 (~33.5M)，正式实验 |
| **Patch size 2** | 得到 32×32 token grid，每个 token 覆盖 2×2 像素 |
| **ImageNet-64 (parquet)** | 本地 `/PixelDiT-vae/c2i/imagenet_parquet/`，160K 张，124 类，256→64 降采样 |
| **CIFAR-10** | 保留用于冒烟测试，32→64 resize（不推荐用于正式实验） |
| **线性 interpolant** | Flow Matching 最简单路径：x_t = (1-t)x_0 + t·x_1，v = x_1 - x_0 |
| **Euler ODE 采样** | 从 t=1 积分到 t=0，可调步数 |
| **NATTEN CUDA + PyTorch fallback** | 训练用 NATTEN（O(N·k²)），测量用 mask（需完整 attention matrix） |
| **Per-t 分析** | ERF 和 distance 按 t 分组输出，支持 t-adaptive kernel 假设验证 |

---

## 代码结构

```
pixelflow_na/
├── README.md                    # 研究提案 + 论文阅读指南
├── PLAN.md                      # 技术实现计划
├── BUILD.md                     # 本文件
├── requirements.txt
│
├── models/                      # 核心模型
│   ├── attention.py             # FullAttention + NeighborAttention + make_attention()
│   ├── dit.py                   # DiT backbone (DiTConfig, DiTBlock, PatchEmbed, AdaLNZero)
│   └── flow_matching.py         # FlowMatchingTrainer + sample_ode() + FID stats
│
├── data/
│   └── dataset.py               # CIFAR-10 / ImageNet-64 / imagefolder 加载
│
├── utils/
│   └── fid.py                   # InceptionV3 FID 计算
│
├── train.py                     # 训练入口
├── measure.py                   # ERF + Distance 分布测量入口
├── sweep.py                     # Kernel size 扫描（全自动：训练→测量→FID）
├── analyze.py                   # 结果可视化
│
├── checkpoints/                 # 模型检查点
└── outputs/                     # 测量结果 + 图表
```

---

## 模型架构细节

### DiT-S (@64×64, patch=2)

| 参数 | 值 |
|------|-----|
| Image size | 64×64×3 |
| Patch size | 2×2 → 32×32 grid = 1024 tokens |
| Hidden dim | 384 |
| Depth | 12 layers |
| Heads | 6 (dim_per_head = 64) |
| MLP ratio | 4.0 |
| Params | ~127M |
| Token rate | 1024 tokens / forward |

### AdaLN-Zero 调制

每层 DiTBlock 通过 AdaLN 接收时间条件信号：
```
c = t_emb (+ label_emb for CFG)
→ AdaLN(c) → (shift, scale, gate) × 2 (attn + MLP)
→ x = x + gate * attn(scale * LN(x) + shift)
→ x = x + gate * mlp(scale * LN(x) + shift)
```
初始化为零（最后一层 projection 权重/bias = 0），模型从恒等映射开始。

### Attention 变体

#### Full Attention
- 标准 multi-head self-attention
- QK^T: [B, heads, N, N]，N=1024 → 1M entries/head
- 每层 ~2M FLOPs attention 部分

#### Neighborhood Attention (NA)
- 每个 query 只关注 k×k spatial window 内的 keys
- Chebyshev 距离（L∞）：max(|Δi|, |Δj|) ≤ k//2
- Mask 实现：先算 QK^T（O(N²)），再加 spatial mask（-inf 到窗口外）
- 对 1024 tokens 足够，真正 O(N·k²) 需要 CUDA kernel
- k=7: 窗口面积 49 → 理论上 1024×49 ≈ 50K entries/head (vs 1M for full)

### Flow Matching 细节

- **训练**：t ~ U(0,1)，x_t = (1-t)x_0 + t·x_1，target v = x_1 - x_0
- **采样**：从 x_1 ~ N(0,I)，Euler 步进 x ← x + v(x,t)·dt，t 从 1→0
- **CFG**：v = v_uncond + cfg_scale × (v_cond - v_uncond)，训练时 10% dropout

---

## 实验矩阵

| 实验 | Attention | Kernel Size | 测量 |
|------|-----------|-------------|------|
| full | Full | — | FID, GFLOPs, ERF, Dist |
| na3 | NA | 3×3 | ↑ |
| na5 | NA | 5×5 | ↑ |
| na7 | NA | 7×7 | ↑ |
| na11 | NA | 11×11 | ↑ |
| na15 | NA | 15×15 | ↑ |

每个变体：
- 100 epochs CIFAR-10 (resize 64×64)，bs=64，lr=1e-4
- 20 ODE 采样步数
- 不同采样步数 (5/10/20/50) 的 sensitivity（Phase 2）

---

## 关键测量方法

### ERF (Effective Receptive Field) — ΔConvFusion 方法

1. 对多张输入、多个时间步 t，提取每层 post-softmax attention weights
2. 对每个 query token q，计算 attention-weighted mean squared 2D distance：
   - ERF(q) = sqrt(Σ_k α_{q→k} · ||pos_q - pos_k||²)
3. 每层 ERF = 所有 query 的 ERF(q) 的平均
4. 比较 full vs NA 下 ERF 随层的变化

### Distance Distribution — PiT 方法

1. 提取 attention weights
2. 对每个 token pair (i,j)，weight by attention score
3. 按 Euclidean 距离分桶，得 P(distance = d)
4. 计算累计分布 P(distance ≤ k)
5. 报告 P50、P95、P99

### GFLOPs 估算

```
Full: 2 × heads × N² × head_dim per layer
NA:   2 × heads × N × k² × head_dim per layer
+ QKV proj + output proj + MLP
```

---

## 已知限制 & 改进方向

1. **NA 效率**：当前 mask-based 实现仍是 O(N²) 显存。对 1024 token 没问题，>4096 需 NAT CUDA kernel
2. **CIFAR-10 分辨率**：32→64 resize 可能不够真实。Phase 2 建议换 ImageNet-64 或 FFHQ-64
3. **FID 精度**：当前用 torchvision InceptionV3，5K 样本。正式结果建议用 clean-fid (10K+)
4. **训练时长**：DiT-T (~21M) 在 CIFAR-10 上 ~0.5h/100 epochs；DiT-S (~127M) ~2h/100 epochs（单卡 3090）
5. **64×64 像素空间 vs 潜在空间**：像素空间模型可能需要更多训练步数才能生成高质量图像。可以考虑先用 DiT-T 快速验证 pipeline

---

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 快速测试：Tiny 模型, CIFAR-10, 5 epochs, 1000 samples only
python train.py --model DiT_T --dataset cifar10 --epochs 5 \
                --max_samples 1000 --batch_size 32 --device cuda

# 完整训练：Small 模型, 全 attention baseline
python train.py --model DiT_S --dataset cifar10 --epochs 200 \
                --batch_size 64 --attn_type full --device cuda

# NA 训练（kernel=7）
python train.py --model DiT_S --dataset cifar10 --epochs 200 \
                --attn_type na --na_kernel_size 7 --device cuda

# 测量 ERF + Distance
python measure.py --ckpt checkpoints/dit_DiT_S_full_k7_cifar10.pt \
                  --attn_type full --n_samples 64

# 完整 sweep（所有 kernel sizes）
python sweep.py --dataset cifar10 --model DiT_S --epochs 200

# 可视化
python analyze.py --results outputs/sweep_results.json
```

---

## 数据路径

- **CIFAR-10**: `/home/Wugj/PixelDiT/data/cifar-10-batches-py/`
- ImageNet-64: 需要下载 → `data/imagenet64/`
- 自动搜索路径: `~/PixelDiT/data/` → `~/data/` → `./data/`
