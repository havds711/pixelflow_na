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
