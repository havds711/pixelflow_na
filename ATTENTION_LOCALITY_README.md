# Attention Locality 测量 — 代码改动说明

对 pretrained Flow Matching 模型，在**不改推理逻辑**的前提下抓取 post-softmax attention weights，计算每个 query 在不同 k×k 窗口内捕获的 attention mass。

## 核心思路

```
标准推理:  x_t → model(x, t, y) → 下一帧 x_{t+dt}
                                └── 正常的 attention 计算

我们的改动: 在 attention 算到 softmax 之后、matmul 之前，
           多抓一份 attention weights，其他什么都不变。
```

## 各模型改动方式

### SiT-XL/2

文件: `measure_locality_sit.py`

`FullAttention` 本身就是手动实现（`attn = softmax(QK^T/√d)` → `attn @ V`），attn 天然可拿到。给模型绑定 `_forward_and_get_attention` 方法，一次 forward 同时返回 velocity + 28 层 attention weights。

### PixDiT c2i

文件: `measure_locality_pixeldit.py`

`RotaryAttention` 使用 `scaled_dot_product_attention(q, k, v)` 融合算子。monkey-patch 替换为等价的 `softmax(QK^T/√d) @ V`，中间多一行 `_COLLECTED_ATTN[id(self)] = attn_weights.detach()`。**其他逻辑一行不改**（RoPE、QK norm、proj 完全相同）。采样逻辑与 `eval_attention_window.py` 一致。

## 文件结构

```
pixelflow_na/
├── measure_locality_sit.py         # SiT-XL/2 测量脚本（conda natten）
├── measure_locality_pixeldit.py    # PixDiT c2i 测量脚本（conda pixel）
├── ATTENTION_LOCALITY_README.md    # 本文档
└── outputs/
    ├── attention_locality_sit.pt       # SiT 数据 (184 MB)
    └── attention_locality_pixeldit.pt  # PixDiT 数据 (247 MB)
```

## 数据格式

每个 `.pt` 文件：
```
masses:    [steps, batch, layers, heads, 256queries, 8kvalues]  float16
min_k_80:  [steps, batch, layers, heads, 256queries]             uint8
min_k_50/90/95/99: 同上
t_schedule: [steps]  float32
```

k_values = [1, 3, 5, 7, 9, 11, 13, 15]

## 关键发现

1. 两个 FM 模型 attention **本质全局**：k=15 才能覆盖 ~93% 的 >80% 达标率
2. PixDiT c2i 比 SiT-XL/2 **更局部化**（k=7: 24.7% vs 13.0%）
3. 存在 **head 级别双峰分布**：同一层内有些 head k=3 就够了，有些必须 k=15
4. t 对 attention 局部性影响很小
5. 空间位置（corner/edge/interior）几乎无影响
