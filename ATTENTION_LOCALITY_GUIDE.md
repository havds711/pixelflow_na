# Attention Locality 测量 — 通用接入指南

给任意一个 pretrained diffusion model 接入 attention locality 测量的标准步骤。

## 总体原则

**不改推理逻辑**。只在 softmax 之后多抓一份 attention weights，其他原封不动。

## Step 1: 确认模型基本信息

向用户确认以下参数：

| 参数 | 问题 | 示例 (SiT) | 示例 (PixDiT) |
|------|------|-----------|--------------|
| 采样方式 | DDIM / ODE / Euler? 几步？ | ODE 20步 | Euler 25步 |
| t 范围 | t ∈ [0,1] 还是 [0,999]? 噪声端是 t=0 还是 t=1? | t=1→0 (噪声→干净) | t=0→1 (噪声→干净) |
| t dtype | float 还是 long? | float32 | float32 |
| 模型预测 | epsilon / velocity / x0? | velocity | velocity |
| 图像尺寸 | H×W, patch_size? | 256×256, patch=2 | 256×256, patch=16 |
| Grid | 多少 token? | 16×16=256 | 16×16=256 |
| 层数 | 几个 attention 层？ | 28 | 30 |
| Heads | 每层几个 head? | 16 | 16 |
| 条件 | CFG? class label? text? | class label (1000类) | class label (1000类) |
| 权重路径 | checkpoint 在哪？ | `SiT/checkpoints/SiT-XL-2-256.pt` | `PixelDiT/c2i/imagenet256_pixeldit_xl_epoch320.ckpt` |
| conda 环境 | 哪个环境能跑？ | `natten` | `pixel` |

## Step 2: 找到 attention 计算位置

打开模型 forward 代码，找到 attention 的实现。分两种情况：

### 情况 A：手动实现的 attention（可直接拿 weights）

```python
# 如果你看到这样的代码：
attn = (q @ k.transpose(-2, -1)) * scale
attn = F.softmax(attn, dim=-1)
x = attn @ v
# → 直接在 softmax 之后抓 attn 即可
```

做法：写一个新的 forward 方法（或用 hook），在 softmax 之后把 `attn` 存下来。SiT 就是这么做的。

### 情况 B：用了融合算子（需要替换）

```python
# 如果你看到：
x = F.scaled_dot_product_attention(q, k, v)
# 或者
x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
# → 融合算子不暴露中间 weights，需要 monkey-patch
```

做法：把 `scaled_dot_product_attention(q, k, v)` 替换为等价的：
```python
attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
if mask is not None:
    attn_weights = attn_weights + mask
attn_weights = F.softmax(attn_weights, dim=-1)
_COLLECTED_ATTN[id(self)] = attn_weights.detach()   # ← 只多了这一行
x = torch.matmul(attn_weights, v)
```

**关键：原模型的 RoPE、QK norm、mask、dropout 等逻辑全部保留，只替换 SDPA 调用。**

## Step 3: 确认采样循环

从原仓库的 eval/sample 脚本中复制采样逻辑，**不要自己发明**。

通常采样代码包含：
```python
x = noise  # 初始状态
for step in range(num_steps):
    t = compute_current_t(step)
    pred = model(x, t, y)    # ← 在这里 forward 时自动抓 attention
    x = update_x(x, pred, t) # ← 原版采样更新
```

注意：
- t 值必须落在训练范围内（如果训练时 t~U(0,1)，推理时 t 也必须在 [0,1]）
- 如果原版有 CFG，去掉它（或者用 guidance=1），避免 cond/uncond 混合影响分析
- batch 大小取 4 即可，class label 取固定几个

## Step 4: 接入 k×k 窗口计算

所有模型共用同一套 k×k 分析方法：

```python
k_values = [1, 3, 5, 7, 9, 11, 13, 15]  # 如果 grid=16

# 预计算每个 (query, k) 的窗口 key index
def precompute_window_indices(grid_size, k_values):
    # 每个 query 的每个 k，返回 k² 个 key 的 flattened index
    # 边界策略：窗口平移，始终完整 k×k

# 运行时向量化累加
for k in k_values:
    idx = gather_idx[k].expand(B, heads, -1, -1)
    mass = attn.gather(dim=-1, index=idx).sum(dim=-1)  # [B, heads, N]
```

## Step 5: 统一的输出格式

所有模型输出相同结构的 `.pt` 文件：

```python
{
    'masses':    torch.zeros(num_steps, B, n_layers, n_heads, N, 8),  # float16
    'min_k_80':  torch.zeros(num_steps, B, n_layers, n_heads, N),     # uint8
    'min_k_50':  ...
    'min_k_90':  ...
    'min_k_95':  ...
    'min_k_99':  ...
    't_schedule': torch.zeros(num_steps),  # float32
}
```

文件命名：`outputs/attention_locality_{模型名}.pt`

## 接入检查清单

- [ ] 确认模型基本信息（Step 1 全部参数）
- [ ] 找到 attention 代码，判断是情况 A 还是 B（Step 2）
- [ ] 复制原版采样逻辑，不改推理（Step 3）
- [ ] 接入 k×k 窗口计算（Step 4）
- [ ] 输出统一格式（Step 5）
- [ ] 用正确环境跑通，检查 t 值范围
- [ ] 结果文件存到 `outputs/attention_locality_{模型名}.pt`
- [ ] 脚本放到 `pixelflow_na/measure_locality_{模型名}.py`

## 常见坑

1. **t 值范围搞错**：训练用 [0,1] 但你传了 [0,999] → attention 是垃圾
2. **t 方向搞反**：噪声在 t=1 但你把噪声当 t=0 → 模型看到反的进度
3. **CFG 没去掉**：cond 和 uncond 各做一次 forward，两次的 attention 混在一起
4. **用了错误的 conda 环境**：模型跑不起来
5. **batch 太大**：attention weights 很占显存，batch=4 足够
