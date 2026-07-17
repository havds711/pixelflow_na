# pixelflow_na — 实现计划

> 2026-07-17 · 目标：拿预训练 flow model，迁移到 pixel-space，验证 NA 行为

---

## 一、基座模型选择

| 候选 | 参数 | 优势 | 劣势 |
|------|------|------|------|
| **SiT-XL/2** | 675M | DiT 架构 = 你的代码高度兼容；开源权重；flow matching | latent space，需迁移到 pixel |
| FLUX.2 klein | 9B | SOTA 质量 | 太大，4×3090 可能跑不动 fine-tune |
| **AsymFLUX.2 klein** | 9B | 自带 pixel-space fine-tune 代码 | 同上，且依赖 FLUX 生态 |

**推荐：SiT-XL/2**。理由：
- DiT backbone = 跟你 `pixel_dit.py` 的 `PixelSpaceDiT` 架构一致，attention 替换零成本
- 675M 参数，服务器单卡/多卡都能跑
- 官方代码直接可跑，Flow Matching 训练/采样都已实现
- latent→pixel 迁移：参考 AsymFlow 的 Procrustes 方法，改动量可控

---

## 二、整体路线

```
SiT-XL/2 预训练权重（latent space, flow matching）
        │
        ▼
  ┌─ Step 1: latent→pixel 迁移 ─────────────────┐
  │  Procrustes 对齐 + pixel head 替换 + fine-tune │
  │  产出：PixelSiT（全 attention 的 pixel baseline）│
  └──────────────────────────────────────────────┘
        │
        ▼
  ┌─ Step 2: NA 替换 + 行为测量 ────────────────┐
  │  替换 attention → ERF 测量 → distance 分布    │
  │  产出：实验数据（FID / GFLOPs / ERF / dist）  │
  └──────────────────────────────────────────────┘
```

---

## 三、代码框架

```
pixelflow_na/
├── README.md                    # 论文阅读指南（现有）
├── PLAN.md                      # 本文件
├── requirements.txt
│
├── baseline/                    # Step 1: latent→pixel 迁移
│   ├── sit_model.py             # 从 SiT 官方 fork，加 pixel head
│   ├── procrustes_align.py      # Procrustes 对齐（参考 AsymFlow）
│   └── train_pixel_finetune.py  # pixel-space fine-tune 脚本
│
├── na_experiments/              # Step 2: NA 替换 + 测量
│   ├── attention_swap.py        # 全 attn → NA 的替换逻辑
│   ├── erf_measure.py           # ΔConvFusion 的 ERF 测量方法
│   ├── distance_measure.py      # PiT 的 attention distance 统计
│   └── sweep_kernel.py          # kernel size 扫描实验
│
└── analysis/                    # 结果分析
    └── plot_results.py          # FID/ERF/distance 对比图
```

---

## 四、Step 1 详细：latent→pixel 迁移

### 4.1 核心思路（来自 AsymFlow §3.2）

SiT 在 latent space 训好了，要把它的输出从 latent 映射到 pixel：

```
原始 SiT:  noise → [SiT DiT blocks] → latent prediction → VAE decoder → image
我们的:    noise → [SiT DiT blocks] → Procrustes proj → pixel prediction → 直接出图
```

具体步骤：
1. 加载 SiT 预训练权重，冻结大部分 DiT blocks
2. 去掉最后的 latent output head，换成 pixel output head（`Linear(d_model, 3×patch²)`）
3. **Procrustes 对齐**：在 latent 空间和 pixel 空间之间学一个正交投影矩阵 P
   - 用少量真实图像（如 ImageNet 1K 张），通过 SiT 的 VAE encoder 得到 latent
   - 解 Procrustes 问题：min ||X_pixel - P·X_latent||²，s.t. P^T P = I
   - P 就是 latent→pixel 的最佳正交近似
4. Fine-tune：解冻全部层，用 flow matching loss 在 pixel space 训练几百步

### 4.2 关键改动

```python
# 原始 SiT 输出
class SiT(nn.Module):
    def forward(self, x, t, y):
        # x: [B, C_latent, H_latent, W_latent]  # 如 [B, 4, 32, 32]
        x = patchify(x)  # → [B, N, d_model]
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x)  # → [B, N, C_latent * patch²]
        x = unpatchify(x)        # → [B, C_latent, H_latent, W_latent]
        return x

# 改成 pixel 输出
class PixelSiT(nn.Module):
    def __init__(self, sit_pretrained, img_size=256):
        # 复用 SiT 的 DiT blocks
        self.blocks = sit_pretrained.blocks  # 冻结 or fine-tune
        self.procrustes = ProcrustesProjection(latent_dim=4, pixel_dim=3)
        # 新 pixel head：输出放回 pixel space
        self.pixel_head = nn.Linear(d_model, 3)  # 3 = RGB，1 token = 1 pixel

    def forward(self, x, t, y):
        # x: [B, 3, H, W]  直接 pixel 输入
        x = pixel_embed(x)  # Linear(3, d_model)，每像素一个 token
        for block in self.blocks:
            x = block(x, c)
        x = self.pixel_head(x)  # → [B, N, 3]
        x = reshape_to_image(x) # → [B, 3, H, W]
        return x
```

### 4.3 资源预估

- SiT-XL/2 权重下载：~2.7 GB
- 加载预训练模型：单卡 24GB 显存够用
- Procrustes 对齐：CPU 就能跑（矩阵 SVD）
- Fine-tune 几百步：4×3090 或服务器单卡，几小时

---

## 五、Step 2 详细：NA 替换 + 行为测量

### 5.1 替换逻辑

基于你现有的 `pixel_dit.py` block 实现（`NeighborAdaDiTBlock` 等），只需要改 attention 调用：

```python
# attention_swap.py
def swap_attention(model, attn_type, kernel_size=7):
    """
    把 PixelSiT 的每个 DiT block 的 self-attention 替换为指定类型。
    attn_type: "full" | "na" | "hydra" | "dcna"
    """
    for i, block in enumerate(model.blocks):
        block.attn = make_attention(
            attn_type, block.dim, block.num_heads, kernel_size
        )
```

### 5.2 ERF 测量（复现 ΔConvFusion 方法）

```python
# erf_measure.py
def measure_erf(model, sample_input):
    """
    1. 前向传播，勾住每层 attention 的 attention weights
    2. 对每个 query token，计算它跟所有 key token 的 attention 权重
    3. 以 query 位置为中心，统计 attention 随距离的衰减
    4. 拟合高斯，半径 = ERF
    """
    attentions = {}  # layer_idx -> attention_weights
    # hook 注册到每个 block
    # ...
    # 计算 ERF
    for layer_idx, attn in attentions.items():
        erf = fit_gaussian_erf(attn)  # 返回如 11.3
        print(f"Layer {layer_idx}: ERF = {erf:.1f}")
```

### 5.3 Distance 分布（复现 PiT 方法）

```python
# distance_measure.py
def measure_distance_distribution(model, sample_input):
    """
    对每个 token pair (i,j)，计算 Euclidean distance，
    加权 attention score，统计距离分布。
    输出：cumulative P(distance ≤ k)
    """
```

### 5.4 实验矩阵

| 实验 | Attention | kernel_size | 采样步数 | 测量指标 |
|------|-----------|-------------|----------|----------|
| baseline | full | — | 5/10/20/50 | FID, GFLOPs, ERF, dist |
| na3 | NA | 3 | 5/10/20/50 | 同上 |
| na5 | NA | 5 | 5/10/20/50 | 同上 |
| na7 | NA | 7 | 5/10/20/50 | 同上 |
| na11 | NA | 11 | 5/10/20/50 | 同上 |
| na15 | NA | 15 | 5/10/20/50 | 同上 |

---

## 六、服务器运行计划

### 环境
- 推荐：单卡 A100 或 4×3090
- SiT-XL 推理：~10GB VRAM
- Fine-tune (batch_size=8, 256×256)：~20GB VRAM

### 时间线

| 阶段 | 内容 | 预计时间 |
|------|------|----------|
| 1 | Clone SiT + 加载预训练权重 + 验证推理 | 半天 |
| 2 | 实现 Procrustes 对齐 + pixel head | 1天 |
| 3 | Fine-tune PixelSiT baseline | 几小时-1天 |
| 4 | 实现 attention swap（已有 block 代码直接用） | 半天 |
| 5 | 实现 ERF + distance 测量 | 1天 |
| 6 | 跑完整实验矩阵 | 1-2天 |
| 7 | 分析 + 画图 | 半天 |

---

## 七、风险与降级方案

| 风险 | 降级方案 |
|------|----------|
| SiT 预训练权重的 VAE 跟我 pixel head 不兼容 | 不从 latent 迁移，直接在 pixel space 从头训一个小 DiT-S（1-2天） |
| Procrustes 对齐质量差（pixel 输出模糊） | 加 perceptual loss（LPIPS），参考 FREPix 的做法 |
| Fine-tune 需要很多步才能收敛 | 先只测 ERF + distance（不依赖 FID），这些用预训练+少量 fine-tune 就能测 |
| 服务器没 GPU | 64×64 分辨率 + DiT-S + 4×3090 本地也能跑 Phase 1 核心实验 |
