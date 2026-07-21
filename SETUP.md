# 服务器环境搭建 & 新模型接入

> 阅读顺序：先看 `ATTENTION_LOCALITY_GUIDE.md`（5步通用流程），再看本文档（具体模型参数和命令）。

---

## 1. 拉代码 & 放权重

```bash
cd /path/to/pixelflow_na
git pull

# 创建 checkpoints 目录，scp 权重进去
mkdir -p checkpoints
# 本地 scp:
# scp DiT-XL-2-256x256.pt SiT-B-2-256.pt SiT-L-2-256.pt mdt_xl2_v1_ckpt.pt server:/path/to/pixelflow_na/checkpoints/
```

---

## 2. 环境搭建

### 2.1 DiT (DDPM) — env: `dit`

```bash
conda create -n dit python=3.10 -y
conda activate dit
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Clone Meta 官方 DiT
cd /path/to/pixelflow_na
git clone https://github.com/facebookresearch/DiT.git dit_repo
cd dit_repo && pip install -e .
```

### 2.2 MDTv2 — env: `mdt`

```bash
conda create -n mdt python=3.10 -y
conda activate mdt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Clone MDT
cd /path/to/pixelflow_na
git clone https://github.com/sail-sg/MDT.git mdt_repo
cd mdt_repo && pip install -e .
```

### 2.3 SiT B/L — env: `natten`（已有，复用）

SiT-B/2 和 SiT-L/2 跟 SiT-XL/2 代码完全一样，只换 checkpoint。直接用现有 `natten` 环境跑。

---

## 3. 四个新模型的测量要点

### 模型参数速查表

| 参数 | DiT-XL/2 | SiT-B/2 | SiT-L/2 | MDTv2-XL/2 |
|---|---|---|---|---|
| 框架 | DDPM | FM (Linear) | FM (Linear) | FM + masked |
| 预测目标 | **ε (noise)** | **v (velocity)** | **v (velocity)** | **v (velocity)** |
| 采样 | DDPM 250步 | ODE 20步 | ODE 20步 | DPM-Solver? |
| t 范围 | [0, 999] (long) | [1→0] (float) | [1→0] (float) | 待确认 |
| 图像尺寸 | 256×256 | 256×256 | 256×256 | 256×256 |
| patch | 2 | 2 | 2 | 2 |
| Grid | 16×16=256 | 16×16=256 | 16×16=256 | 16×16=256 |
| 层数 | 28 | 12 | 24 | 待确认 |
| Heads | 16 | 12 | 16 | 待确认 |
| CFG | class label | class label (要关) | class label (要关) | 无 CFG |
| 权重 | `checkpoints/DiT-XL-2-256x256.pt` | `checkpoints/SiT-B-2-256.pt` | `checkpoints/SiT-L-2-256.pt` | `checkpoints/mdt_xl2_v1_ckpt.pt` |
| env | `dit` | `natten` | `natten` | `mdt` |
| 模板脚本 | `measure/measure_locality_sit.py` | `measure/measure_locality_sit.py` | `measure/measure_locality_sit.py` | 新建 |

### 3.1 SiT-B/2 & SiT-L/2 — 最简单

代码零改动。直接跑 `measure/measure_locality_sit.py`，改 `--ckpt` 指向 B 或 L 的权重即可：

```bash
conda activate natten
cd /path/to/pixelflow_na

# SiT-B/2
python measure/measure_locality_sit.py \
  --ckpt checkpoints/SiT-B-2-256.pt \
  --num-steps 20 \
  --output outputs/attention_locality_sit_b

# SiT-L/2
python measure/measure_locality_sit.py \
  --ckpt checkpoints/SiT-L-2-256.pt \
  --num-steps 20 \
  --output outputs/attention_locality_sit_l
```

### 3.2 DiT-XL/2 (DDPM) — 关键新模型

**需要改的地方**（基于 `measure/measure_locality_sit.py`）：

1. **采样逻辑不同**：DDPM 用 `x_{t-1} = 1/√α_t · (x_t - (1-α_t)/√(1-ᾱ_t) · ε_θ) + σ_t·z`，不是 ODE
2. **t 是 long (0-999)**，不是 float [0,1]
3. **模型预测 ε (noise)**，不是 velocity
4. **attention 结构相同**：都是 DiT block，手动实现 `softmax(QK^T/√d)`，情况 A，可直接抓

**做法**：
- 从 `dit_repo/sample.py` 复制 DDPM 采样循环
- 用 `measure/measure_locality_sit.py` 的 attention hook 逻辑（相同架构，直接复用）
- 接入统一的 k×k 窗口计算
- 输出到 `outputs/attention_locality_dit_xl/`

### 3.3 MDTv2-XL/2 — 需确认 attention 类型

**需要先确认**：
1. 打开 `mdt_repo` 找 attention 实现 → 判断是手动 softmax（情况 A）还是 `scaled_dot_product_attention`（情况 B）
2. 确认采样逻辑（DDPM? ODE? 几步？t 怎么给？）
3. 确认层数和 heads 数
4. MDT mask 在推理时是否应用？（大概率不应用 → attention 是标准的）

**做法**：
- 从 `mdt_repo` 复制采样逻辑
- 如果是情况 A → 参考 `measure/measure_locality_sit.py` 的 hook
- 如果是情况 B → 参考 `measure/measure_locality_pixeldit.py` 的 monkey-patch
- 输出到 `outputs/attention_locality_mdtv2_xl/`

---

## 4. 执行顺序

```
1. SiT-B + SiT-L  ← 零改动，先跑，建立 FM 规模曲线
2. DiT-XL/2       ← 核心对比，需要改采样但 attention 复用
3. MDTv2-XL/2     ← 需要先读代码确认，最后做
```

每个模型跑完确认 `.pt` 文件正确生成后再做下一个。

---

## 5. 验证检查清单

- [ ] 模型 load 成功，无报错
- [ ] t 值范围正确（SiT=float[1→0], DiT=long[0→999]）
- [ ] CFG 已关闭（guidance=1 或去掉 cond/uncond 分支）
- [ ] attention weights 维度正确：[B, heads, 256, 256]
- [ ] masses 在 [0,1] 范围内，每行和为 1.0
- [ ] 输出文件格式符合 GUIDE Step 5
- [ ] 文件命名：`outputs/attention_locality_{模型名}/attention_locality_{模型名}.pt`

---

## 6. 四个新模型 (PixArt-α, PixArt-Σ, SD 1.5, SD XL)

> **状态**: 环境就绪，脚本就绪，等待 checkpoint 上传后运行。
>
> 这四个模型引入了两个关键新维度：
> - **Cross-attention** — 所有现有 6 个模型都是纯 self-attention DiT；PixArt 和 SD 都有 cross-attention
> - **UNet 架构** — SD 使用 UNet backbone（多分辨率 self-attention），与 DiT 结构形成对比

### 6.1 模型参数速查表

| 参数 | PixArt-α XL/2 | PixArt-Σ XL/2 | SD 1.5 | SD XL |
|---|---|---|---|---|
| 架构 | DiT + cross-attn | DiT + cross-attn | UNet + cross-attn | UNet + cross-attn |
| 框架 | DDPM (ε-pred) | DDPM (v-pred?) | DDPM (ε-pred) | DDPM (ε-pred) |
| 采样 | DDPM 250步 | DDPM 250步 | DDIM 50步 | DDIM 50步 |
| t 范围 | [0, 999] long | [0, 999] long | [0, 999] long | [0, 999] long |
| 图像尺寸 | 256×256 | 256×256 | 512×512 | 512×512 |
| Latent | 32×32 (VAE) | 32×32 (VAE) | 64×64 (VAE) | 64×64 (VAE) |
| patch | 2 | 2 | N/A (UNet) | N/A (UNet) |
| Self-attn Grid | **16×16=256** | **16×16=256** | **32×32 / 16×16 / 8×8** | **32×32 / 16×16 / 8×8** |
| 层数 | 28 | 28 | ~11 self-attn 层 | ~13 self-attn 层 |
| Heads | 16 | 16 | 8 (varies) | 10 (varies) |
| 条件 | class label (embed) | class label (embed) | text prompt | text prompt |
| Cross-attn | class embed (1 token) | class embed (1 token) | T5 text (77 tokens) | dual T5 text |
| 权重 | `checkpoints/PixArt-XL-2-256x256.pth` | `checkpoints/PixArt-Sigma-XL-2-256x256.pth` | `checkpoints/sd/v1-5-pruned-emaonly.safetensors` | `checkpoints/sd/sd_xl_base_1.0.safetensors` |
| env | `pixart` | `pixart` | `sd15` | `sd15` |
| 测量脚本 | `measure/measure_locality_pixart.py` | `measure/measure_locality_pixart.py` | `measure/measure_locality_sd.py` | `measure/measure_locality_sd.py` |

### 6.2 环境搭建

```bash
# 一键创建 pixart 和 sd15 两个环境
cd /path/to/pixelflow_na
bash setup_4models_envs.sh

# 或者手动:
# --- pixart ---
conda create -n pixart python=3.10 -y
conda activate pixart
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate timm einops numpy tqdm

# --- sd15 ---
conda create -n sd15 python=3.10 -y
conda activate sd15
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate numpy tqdm
```

### 6.3 权重放置

```bash
mkdir -p checkpoints/sd

# 本地 scp:
# scp PixArt-Sigma-XL-2-256x256.pth server:/path/to/pixelflow_na/checkpoints/
# scp PixArt-XL-2-256x256.pth       server:/path/to/pixelflow_na/checkpoints/
# scp v1-5-pruned-emaonly.safetensors server:/path/to/pixelflow_na/checkpoints/sd/
# scp sd_xl_base_1.0.safetensors       server:/path/to/pixelflow_na/checkpoints/sd/
```

### 6.4 运行测量

```bash
# ====== PixArt-α (DDPM 250步, 30-90分钟) ======
conda activate pixart
python measure/measure_locality_pixart.py \
  --ckpt checkpoints/PixArt-XL-2-256x256.pth \
  --model-name pixart_alpha --num-steps 250

# ====== PixArt-Σ (DDPM 250步, 30-90分钟) ======
conda activate pixart
python measure/measure_locality_pixart.py \
  --ckpt checkpoints/PixArt-Sigma-XL-2-256x256.pth \
  --model-name pixart_sigma --num-steps 250

# ====== SD 1.5 (DDIM 50步, ~15-30分钟) ======
conda activate sd15
python measure/measure_locality_sd.py \
  --model sd15 --ckpt checkpoints/sd/v1-5-pruned-emaonly.safetensors \
  --num-steps 50 --resolution 512

# ====== SD XL (DDIM 50步, ~20-40分钟) ======
conda activate sd15
python measure/measure_locality_sd.py \
  --model sdxl --ckpt checkpoints/sd/sd_xl_base_1.0.safetensors \
  --num-steps 50 --resolution 512
```

### 6.5 PixArt 脚本技术说明

**模型定义（自包含）**:
- `measure/measure_locality_pixart.py` 内嵌了完整的 PixArt 模型定义（`PixArt`, `PixArtBlock`, `WindowAttention`, `MultiHeadCrossAttention` 等），无需克隆 PixArt-alpha repo
- 架构: 28 层 DiT block，每层 = self-attn (WindowAttention) + cross-attn + MLP, adaLN-single
- 原始 PixArt 使用 xformers `memory_efficient_attention` → 脚本替换为手动 `softmax(QK^T/√d)` 以暴露 attention weights
- Cross-attention 也替换为手动实现（移除 xformers 依赖），保持跨环境兼容

**条件处理**:
- PixArt-α/Σ 的 ImageNet 训练使用了 class label → `LabelEmbedder` → 作为 cross-attention 的 condition（1 个 token）
- 脚本将 class label embedding 作为 1-token 的 cross-attention condition，模拟训练时的行为
- 不使用 T5 text encoder（纯 class-conditional 模式，与现有 DiT/SiT 模型一致）

**采样**:
- 两者都用 DDPM 250 步（与 DiT-XL/2 一致，方便对比）
- PixArt 预测 ε + Σ（pred_sigma=True），取前半部分作为 ε
- DDPM schedule: linear β, 1000 diffusion steps, 降采样到 250

**Checkpoint 加载**:
- 自动处理 `state_dict`/`model`/`ema` 等常见 wrapper key
- 自动去除 `module.`/`denoiser.`/`ema_denoiser.` 等前缀
- 使用 `strict=False` 加载，打印缺失/多余 keys 供检查

**注意**:
- 如果 checkpoint 中的 key 名与脚本定义的 key 名不匹配，`strict=False` 会让模型加载但不完整
- 检查输出的 "Missing keys" 数量：少量（如 `y_embedder.embedding_table.weight` 初始化值）可忽略
- 如果大量 keys 缺失，说明 checkpoint 使用了不同的模型定义（如 `PixArtMS` T2I 版本），需要调整

### 6.6 SD 脚本技术说明

**架构探测**:
- 脚本自动遍历 UNet 找到所有 `attn1` 模块（self-attention）
- Dry run 第一步确认每层的 grid size 和 heads 数
- 按 grid size 分组存储数据（不同分辨率用不同 k-values）

**Attention 抓取**:
- 替换 diffusers 默认的 `AttnProcessor2_0`（使用 `F.scaled_dot_product_attention`）为自定义 `CaptureAttnProcessor`
- 自定义 processor 使用等价的手动 softmax，在 softmax 之后保存 attention weights
- 只抓取 self-attention（`attn1`），不影响 cross-attention（`attn2`）

**条件处理**:
- SD 使用 text prompt（不是 class label），通过 pipeline 编码为 T5 text embeddings
- 使用 4 个固定 prompt（对应 ImageNet 的 4 个类别名），与 class-conditional 模型保持一致的图像多样性
- 关闭 CFG（classifier-free guidance），避免 cond/uncond 混合影响 attention 分析

**多分辨率存储**:
- SD UNet 在多个分辨率上有 self-attention：
  - 32×32 (1024 tokens): 5 层, k = [1,3,...,31]
  - 16×16 (256 tokens): 5 层, k = [1,3,...,15]
  - 8×8 (64 tokens): 1 层, k = [1,3,5,7]
- 数据按 grid_size 分组存储为 `data['by_grid'][grid_size]`，每组是标准的 `[S,B,L,H,N,K]` 格式
- 同时保存 `layer_grid_sizes` 和 `layer_names` 用于映射回 UNet 结构

**SD 1.5 vs SD XL**:
- SD 1.5: UNet 860M 参数，8 attention heads, head_dim=40 (varies)
- SD XL: UNet 2.6B 参数，更多 transformer blocks，dual text encoder
- 两者都在 512×512 分辨率下测量（SD XL 原生 1024×1024，用 512 降低显存）

### 6.7 验证检查清单

- [ ] 模型 load 成功，无大量 missing keys（<10 个 ok）
- [ ] PixArt: t 值范围正确（DDPM long [0, 999]）
- [ ] SD: scheduler timesteps 正确设置（DDIM, 50步）
- [ ] CFG 已关闭（guidance=1 或无 cond/uncond）
- [ ] PixArt attention weights 维度：[B, 16, 256, 256]
- [ ] SD: 每个 grid_size 的 attention weights 维度正确
- [ ] masses 在 [0,1] 范围内，每行和为 1.0
- [ ] PixArt 输出: `outputs/attention_locality_{pixart_alpha,pixart_sigma}/attention_locality_{pixart_alpha,pixart_sigma}.pt`
- [ ] SD 输出: `outputs/attention_locality_{sd15,sdxl}/attention_locality_{sd15,sdxl}.pt`

### 6.8 科学价值

这四个模型完成后，数据集将覆盖 **10 个模型 × 3 个维度**：

| 维度 | 现有模型 (6) | 新增模型 (4) |
|---|---|---|
| **框架** | DDPM (DiT, MDT) + FM (SiT, PixDiT) | DDPM (PixArt-α, SD 1.5, SD XL) + DDPM/v-pred (PixArt-Σ) |
| **架构** | DiT 全系列 | DiT+cross-attn (PixArt) + UNet (SD) |
| **注意力类型** | 纯 self-attention | self-attn + cross-attn |

可回答的新问题：
- Cross-attention 是否影响 self-attention 的 locality 模式？
- UNet 多分辨率 attention 在不同尺度上的 locality 是否一致？
- Text prompt 条件 vs class label 条件下 attention locality 有何差异？
- PixArt-α (DDPM) vs PixArt-Σ (v-pred) 的 locality 差异 = 训练配方的影响？
- SD 1.5 vs SD XL 的 locality 随模型规模如何变化？
