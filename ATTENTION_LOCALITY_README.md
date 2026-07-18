# Attention Locality 测量 — 代码改动说明

## 做了什么

对 SiT-XL/2 和 PixDiT c2i 两个 pretrained Flow Matching 模型，在**不改推理逻辑**的前提下，抓取每次 forward 时的 post-softmax attention weights，计算每个 query 在不同 k×k 窗口内捕获了多少 attention mass。

## 核心思路

```
标准推理:  x_t → model(x, t, y) → 下一帧 x_{t+dt}
                                └── 正常的 attention 计算

我们的改动: 在 attention 计算到 softmax 之后、matmul 之前，
           多抓一份 attention weights，其他什么都不变。
```

### SiT-XL/2 的改动

文件: `pixelflow_na/measure_attention_locality.py`

`FullAttention` 本身就是手动实现的（`attn = softmax(QK^T/√d)` → `attn @ V`），attn 天然可拿到。做法：给模型绑定一个 `_forward_and_get_attention` 方法，一次 forward 同时返回 velocity + 28 层 attention weights。

### PixDiT c2i 的改动

文件: `PixelDiT/c2i/measure_attention_locality.py`

`RotaryAttention` 使用 `scaled_dot_product_attention(q, k, v)` 融合算子，不暴露中间 attention weights。做法：

- monkey-patch `RotaryAttention.forward()`，把 `scaled_dot_product_attention` 替换为等价的 `softmax(QK^T/√d) @ V`，中间多一行 `_COLLECTED_ATTN[id(self)] = attn_weights.detach()`
- **其他逻辑一行不改**：RoPE、QK norm、proj、dropout 完全相同
- 采样逻辑和 `eval_attention_window.py` 完全一致：t=0→1 Euler 积分，无 CFG

## k×k 窗口累加方法

每个 query 位置 (i,j)，对每个奇数 k，取以该位置为中心的 k×k 窗口（边界处平移使窗口完整留在图内，每个 query 总是得到 k² 个 key），累加对这些 key 的 attention weights。

预计算每个 (query, k) 组合的 key index → 运行时用 `attn.gather(dim=-1, index)` 向量化求和。

## 数据文件结构

每个 `.pt` 文件包含：
```
masses:    [steps, batch, layers, heads, 256queries, 8kvalues]  float16
min_k_80:  [steps, batch, layers, heads, 256queries]             uint8
min_k_50/90/95/99: 同上
t_schedule: [steps]  float32  (每一步的归一化 t 值)
```

k_values = [1, 3, 5, 7, 9, 11, 13, 15]

SiT: 20 steps × 4 images × 28 layers × 16 heads × 256 queries
PixDiT: 25 steps × 4 images × 30 layers × 16 heads × 256 queries

## 关键发现

1. 两个 FM 模型 attention 都是**本质全局的**：k=15 才能覆盖 ~93% 的 >80% 达标率
2. PixDiT c2i 比 SiT-XL/2 **更局部化**一些（k=7: 24.7% vs 13.0%）
3. 存在明显的 **head 级别双峰分布**：同一层内有些 head 非常局部（k=3 即可），有些非常全局（必须 k=15）
4. t 对 attention 局部性影响很小
5. 空间位置（corner/edge/interior）几乎无影响

## 文件位置汇总

| 内容 | 路径 |
|------|------|
| SiT 测量脚本 | `pixelflow_na/measure_attention_locality.py` |
| PixDiT 测量脚本 | `PixelDiT/c2i/measure_attention_locality.py` |
| SiT 数据 | `pixelflow_na/outputs/attention_locality/attention_locality.pt` |
| PixDiT 数据 | `PixelDiT/c2i/attention_locality_output/attention_locality_pixeldit.pt` |
| 本文档 | `pixelflow_na/ATTENTION_LOCALITY_README.md` |
