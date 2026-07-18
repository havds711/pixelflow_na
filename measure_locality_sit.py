#!/usr/bin/env python3
"""
SiT-XL/2 真实 ODE 推理 — Attention Locality 全量测量

对 ODE 推理的每一步，提取每层每个 head 的 post-softmax attention，
计算每个 query 在不同 k×k 窗口内的累积 attention mass。
窗口在边界处平移（不截断），每个 query 始终得到完整的 k² 个 key。

全量保存原始数据到 .pt 文件，同时打印汇总表格。

Usage:
  python measure_attention_locality.py --device cuda:0
"""

import argparse, os, sys, json, math
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SiT'))
sys.path.insert(0, os.path.dirname(__file__))
from SiT.models import SiT_XL_2
from SiT.download import find_model
from SiT.models import modulate


# ═════════════════════════════════════════════════════════════════════════════
# 1. Monkey-patch: 一次 forward 同时输出 velocity 和所有层 attention weights
# ═════════════════════════════════════════════════════════════════════════════

def _forward_and_get_attention(self, x, t, y):
    """返回 (velocity_pred, attention_list) — 一次前向完成两件事"""
    tokens = self.x_embedder(x) + self.pos_embed
    t_emb = self.t_embedder(t)
    y_emb = self.y_embedder(y, train=False)
    c = t_emb + y_emb

    grid_size = int(tokens.shape[1] ** 0.5)
    all_attn = []

    for block in self.blocks:
        shift_msa, scale_msa, gate_msa, _, _, _ = block.adaLN_modulation(c).chunk(6, dim=1)
        x_norm = modulate(block.norm1(tokens), shift_msa, scale_msa)

        if hasattr(block.attn, 'extract_attention_weights'):
            attn_weights = block.attn.extract_attention_weights(
                x_norm, grid_h=grid_size, grid_w=grid_size
            )
            all_attn.append(attn_weights)
        else:
            all_attn.append(None)

        tokens = block(tokens, c)

    x = self.final_layer(tokens, c)
    x = self.unpatchify(x)
    if self.learn_sigma:
        x, _ = x.chunk(2, dim=1)
    return x, all_attn


# ═════════════════════════════════════════════════════════════════════════════
# 2. 预计算窗口索引（边界平移，完整 k×k）
# ═════════════════════════════════════════════════════════════════════════════

def precompute_window_indices(grid_size: int, k_values: list):
    """
    对每个 (query_pos, k)，返回窗口内 key 的 flattened index (shape [k²])。
    边界策略：窗口平移使其完全在图像内。每个 query 总是得到 k² 个 key。
    """
    H = W = grid_size
    # indices[k][qi][qj] = tensor [k²]
    indices = {k: torch.zeros(H, W, k * k, dtype=torch.long) for k in k_values}

    for k in k_values:
        half = k // 2
        for qi in range(H):
            for qj in range(W):
                ci = max(half, min(qi, H - 1 - half))
                cj = max(half, min(qj, W - 1 - half))
                r0, r1 = ci - half, ci + half + 1
                c0, c1 = cj - half, cj + half + 1
                idx_list = []
                for r in range(r0, r1):
                    for c in range(c0, c1):
                        idx_list.append(r * W + c)
                indices[k][qi, qj] = torch.tensor(idx_list, dtype=torch.long)

    # 合并为 [N, k²] 以便 gather，N = H*W
    gather_indices = {}
    for k in k_values:
        gather_indices[k] = indices[k].reshape(H * W, k * k)  # [N, k²]
    return gather_indices


# ═════════════════════════════════════════════════════════════════════════════
# 3. ODE 推理 + 全量 attention 采集
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_ode_and_collect(
    model,
    class_labels: list,
    num_steps: int = 20,
    k_values: list = None,
    device: str = "cuda",
    seed: int = 42,
):
    """
    ODE 推理（Euler, t=1→0），每步对每层每个 head 计算 k×k 窗口内 attention mass。

    Returns:
      masses:    [num_steps, B, n_layers, n_heads, N, n_k]  float16
      min_k:     [num_steps, B, n_layers, n_heads, N]        uint8  (min k for >80%)
      min_k_50:  同上
      min_k_90:  同上
      min_k_95:  同上
      min_k_99:  同上
      t_schedule: [num_steps]  float32
    """
    if k_values is None:
        k_values = [1, 3, 5, 7, 9, 11, 13, 15]

    torch.manual_seed(seed)
    model.eval()

    B = len(class_labels)
    labels = torch.tensor(class_labels, device=device)
    grid_size = model.grid_size  # 16
    N = grid_size * grid_size    # 256
    n_layers = 28
    n_heads = 16
    n_k = len(k_values)
    dt = 1.0 / num_steps

    # 预计算窗口索引
    print("Precomputing window indices...")
    gather_idx = precompute_window_indices(grid_size, k_values)
    # 搬到 GPU 方便 gather
    gather_idx_gpu = {k: v.to(device) for k, v in gather_idx.items()}

    # 存储
    masses = torch.zeros(num_steps, B, n_layers, n_heads, N, n_k, dtype=torch.float16)
    min_k_80 = torch.zeros(num_steps, B, n_layers, n_heads, N, dtype=torch.uint8)
    min_k_50 = torch.zeros(num_steps, B, n_layers, n_heads, N, dtype=torch.uint8)
    min_k_90 = torch.zeros(num_steps, B, n_layers, n_heads, N, dtype=torch.uint8)
    min_k_95 = torch.zeros(num_steps, B, n_layers, n_heads, N, dtype=torch.uint8)
    min_k_99 = torch.zeros(num_steps, B, n_layers, n_heads, N, dtype=torch.uint8)
    t_schedule = torch.zeros(num_steps, dtype=torch.float32)

    # 初始噪声
    x = torch.randn(B, model.in_channels, grid_size * 2, grid_size * 2, device=device)

    for step in tqdm(range(num_steps), desc="ODE inference + attention collection"):
        t_current = 1.0 - step * dt
        t_schedule[step] = t_current
        t_tensor = torch.full((B,), t_current, device=device)

        # 一次 forward → velocity + 所有 attention
        v, attn_list = model._forward_and_get_attention(x, t_tensor, labels)

        for layer_idx, attn in enumerate(attn_list):
            if attn is None:
                continue
            # attn: [B, n_heads, N, N]  post-softmax, on GPU
            B2, H2, _, _ = attn.shape

            for ki, k in enumerate(k_values):
                # gather_idx_gpu[k]: [N, k²]
                # 扩展为 [1, 1, N, k²] 用于 gather
                idx = gather_idx_gpu[k].unsqueeze(0).unsqueeze(0)  # [1, 1, N, k²]
                idx = idx.expand(B2, H2, -1, -1)                   # [B, heads, N, k²]

                # attn: [B, heads, N, N]
                # 对每个 query (dim 2)，收集窗口内的 key (dim 3) → [B, heads, N, k²]
                gathered = attn.gather(dim=-1, index=idx)
                mass = gathered.sum(dim=-1)  # [B, heads, N]

                masses[step, :, layer_idx, :, :, ki] = mass.half().cpu()

        # 向量化计算 min_k：对每种阈值，找第一个 mass > threshold 的 k
        # masses_step: [B, L, H, N, K] on CPU
        m_step = masses[step]  # [B, L, H, N, K] float16 → float32 for comparison
        k_tensor = torch.tensor(k_values, dtype=torch.float32)  # [K]

        for threshold, storage in [(0.50, min_k_50), (0.80, min_k_80),
                                    (0.90, min_k_90), (0.95, min_k_95),
                                    (0.99, min_k_99)]:
            # 对每个 (b,l,h,q)，找第一个满足 mass >= threshold 的 ki
            above = m_step.float() >= threshold  # [B, L, H, N, K] bool
            # argmax 找到第一个 True（如果没有 True，argmax 返回 0，我们后面用 mask 处理）
            first_ki = above.float().argmax(dim=-1)  # [B, L, H, N]
            # 对应的 k 值
            min_k_val = k_tensor[first_ki]  # [B, L, H, N]
            # 没有任何 k 达标的位置保持 0
            any_above = above.any(dim=-1)  # [B, L, H, N]
            storage[step] = (min_k_val * any_above).byte()

        # Euler 步进: dx/dt = v, dt < 0 → x ← x - v·dt = x + v·(-dt)
        x = x - v * dt

    return {
        'masses': masses,           # [20, B, 28, 16, 256, 8] float16
        'min_k_80': min_k_80,       # [20, B, 28, 16, 256] uint8
        'min_k_50': min_k_50,
        'min_k_90': min_k_90,
        'min_k_95': min_k_95,
        'min_k_99': min_k_99,
        't_schedule': t_schedule,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 4. 汇总分析 + 打印表格
# ═════════════════════════════════════════════════════════════════════════════

def analyze_and_print(data: dict, k_values: list):
    masses = data['masses']          # [S, B, L, H, N, K]
    min_80 = data['min_k_80']        # [S, B, L, H, N]
    min_50 = data['min_k_50']
    min_90 = data['min_k_90']
    min_95 = data['min_k_95']
    min_99 = data['min_k_99']
    t_sched = data['t_schedule']

    S, B, L, H, N, K = masses.shape
    total_pairs = S * B * H * N  # 每层的 (step, img, head, query) 对总数
    k_arr = np.array(k_values)

    # 空间区域划分
    gs = int(N ** 0.5)
    region = np.zeros((gs, gs), dtype=int)
    for i in range(gs):
        for j in range(gs):
            d = min(i, j, gs - 1 - i, gs - 1 - j)
            if d <= 1:
                region[i, j] = 0  # corner
            elif d <= 3:
                region[i, j] = 1  # edge
            else:
                region[i, j] = 2  # interior
    region_flat = region.reshape(-1)
    region_counts = np.bincount(region_flat, minlength=3)  # [n_corner, n_edge, n_interior]
    region_names = ['corner', 'edge  ', 'interior']

    # ── 表1: 全局 ──
    print("\n" + "=" * 110)
    print("表1 — 全局 Summary：k×k 窗口捕获 attention mass 的比例")
    print(f"     统计范围: {S} ODE steps × {B} images × {L} layers × {H} heads × {N} queries")
    print(f"     总 (head,query) 对: {total_pairs * L:,}")
    print("=" * 110)
    header = f"{'k':<8} {'>50%':>10} {'>80%':>10} {'>90%':>10} {'>95%':>10} {'>99%':>10}  {'推荐':>6}"
    print(header)
    print("-" * len(header))
    for ki, k in enumerate(k_values):
        r50 = (masses[:, :, :, :, :, ki] > 0.50).float().mean().item()
        r80 = (masses[:, :, :, :, :, ki] > 0.80).float().mean().item()
        r90 = (masses[:, :, :, :, :, ki] > 0.90).float().mean().item()
        r95 = (masses[:, :, :, :, :, ki] > 0.95).float().mean().item()
        r99 = (masses[:, :, :, :, :, ki] > 0.99).float().mean().item()
        flag = " ← 80%+" if r80 >= 0.8 else ""
        print(f"k={k:<5} {r50:>10.3f} {r80:>10.3f} {r90:>10.3f} {r95:>10.3f} {r99:>10.3f}  {flag}")

    # ── 表2: 每层 k 矩阵 ──
    print("\n" + "=" * 110)
    print("表2 — 每层 k×k 窗口 >80% 达标比例（avg over steps, images, heads, queries）")
    print("=" * 110)
    header2 = f"{'L':<4}" + "".join(f"{'k='+str(k):>10}" for k in k_values) + f"  {'推荐k':>6}  {'均值':>6}"
    print(header2)
    print("-" * len(header2))
    for l in range(L):
        row = f"{l:<4}"
        best_k = None
        ratios = []
        for ki, k in enumerate(k_values):
            r = (masses[:, :, l, :, :, ki] > 0.80).float().mean().item()
            ratios.append(r)
            row += f"{r:>10.3f}"
            if best_k is None and r >= 0.8:
                best_k = k
        best_str = str(best_k) if best_k else ">15"
        row += f"  {best_str:>6}  {np.mean(ratios):>6.3f}"
        print(row)

    # ── 表3: t 依赖（分早/中/晚期） ──
    print("\n" + "=" * 110)
    print("表3 — t 依赖：早/中/晚期 每层 k=7 >80% 达标比例")
    print("=" * 110)
    # early: step 0-5 (t≈1.0→0.75), mid: 6-13 (t≈0.7→0.35), late: 14-19 (t≈0.3→0.0)
    stages = {'early (t≈1.0-0.75)': slice(0, 6),
              'mid   (t≈0.70-0.35)': slice(6, 14),
              'late  (t≈0.30-0.00)': slice(14, 20)}
    ki7 = k_values.index(7) if 7 in k_values else 0
    header3 = f"{'L':<4}" + "".join(f"{name:>22}" for name in stages.keys())
    print(header3)
    print("-" * len(header3))
    for l in range(L):
        row = f"{l:<4}"
        for name, sl in stages.items():
            r = (masses[sl, :, l, :, :, ki7] > 0.80).float().mean().item()
            row += f"{r:>22.3f}"
        print(row)

    # ── 表4: Head 差异（全部 28 层） ──
    print("\n" + "=" * 110)
    print("表4 — Head 差异：全部 28 层，每层 16 head 的 min_k_80 统计")
    print("=" * 110)
    header4 = f"{'L':<4} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}  | 各 head 的 mean min_k"
    print(header4)
    print("-" * len(header4))
    for l in range(L):
        # min_80: [S, B, L, H, N] → 取第 l 层
        layer_data = min_80[:, :, l, :, :].float()  # [S, B, H, N]
        layer_data = layer_data.flatten()            # 所有此层的 query
        layer_data = layer_data[layer_data > 0]      # 排除未达标的 (值为0)

        if len(layer_data) == 0:
            print(f"{l:<4} {'N/A':>8}")
            continue

        # 每 head 统计
        head_means = []
        for h in range(H):
            h_data = min_80[:, :, l, h, :].float().flatten()
            h_data = h_data[h_data > 0]
            head_means.append(h_data.mean().item() if len(h_data) > 0 else 0)

        hdr = " ".join(f"{hm:>5.0f}" for hm in head_means)
        print(f"{l:<4} {layer_data.mean().item():>8.1f} {layer_data.std().item():>8.1f} "
              f"{layer_data.min().item():>8.0f} {layer_data.max().item():>8.0f}  | {hdr}")

    # ── 表5: 空间位置（全部 28 层） ──
    print("\n" + "=" * 110)
    print("表5 — 空间位置：corner / edge / interior 的 min_k_80 均值（avg over all else）")
    print(f"     各区域 query 数: corner={region_counts[0]}, edge={region_counts[1]}, interior={region_counts[2]}")
    print("=" * 110)
    header5 = f"{'L':<4}" + "".join(f"{name:>12}" for name in region_names) + "  interior-corner"
    print(header5)
    print("-" * len(header5))
    for l in range(L):
        row = f"{l:<4}"
        vals = []
        for ri in range(3):
            q_mask = (region_flat == ri)
            r_data = min_80[:, :, l, :, q_mask].float().flatten()
            r_data = r_data[r_data > 0]
            v = r_data.mean().item() if len(r_data) > 0 else 0
            vals.append(v)
            row += f"{v:>12.1f}"
        diff = vals[2] - vals[0]  # interior - corner
        row += f"  {diff:>+12.1f}"
        print(row)

    # ── 表6: 每层 min_k 分布汇总 ──
    print("\n" + "=" * 110)
    print("表6 — 每层 min_k_80 分布：各 k 值占比 (%)")
    print("=" * 110)
    header6 = f"{'L':<4}" + "".join(f"{'k='+str(k):>8}" for k in k_values) + f"  {'未达标':>8}"
    print(header6)
    print("-" * len(header6))
    for l in range(L):
        row = f"{l:<4}"
        layer_mk = min_80[:, :, l, :, :].flatten()  # uint8
        total = len(layer_mk)
        for k in k_values:
            pct = (layer_mk == k).sum().item() / total * 100
            row += f"{pct:>8.1f}"
        pct_fail = (layer_mk == 0).sum().item() / total * 100
        row += f"  {pct_fail:>8.1f}"
        print(row)


# ═════════════════════════════════════════════════════════════════════════════
# 5. Main
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str,
                   default=os.path.join(os.path.dirname(__file__), 'SiT/checkpoints/SiT-XL-2-256.pt'))
    p.add_argument('--num_steps', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', type=str, default='outputs/attention_locality_sit.pt',
                   help='Output .pt file path')
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    k_values = [1, 3, 5, 7, 9, 11, 13, 15]

    print("=" * 60)
    print("SiT-XL/2 ODE Inference — Attention Locality Measurement")
    print(f"ODE steps: {args.num_steps} | Batch: {args.batch_size} | Device: {device}")
    print(f"Grid: 16×16=256 tokens | Layers: 28 | Heads: 16")
    print(f"k values: {k_values}")
    print("=" * 60)

    # Load model
    print("\nLoading SiT-XL/2...")
    state_dict = find_model(args.ckpt)
    model = SiT_XL_2(input_size=32, num_classes=1000, learn_sigma=True).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    # 绑定自定义 forward
    model._forward_and_get_attention = _forward_and_get_attention.__get__(model, type(model))
    print(f"Loaded: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")

    # 固定 class labels
    class_labels = [207, 360, 387, 974]  # 4 个不同类别

    # Run
    print(f"\nRunning ODE inference ({args.num_steps} steps × {args.batch_size} images)...")
    data = run_ode_and_collect(
        model, class_labels,
        num_steps=args.num_steps,
        k_values=k_values,
        device=device,
        seed=args.seed,
    )

    # Analyze + print tables
    analyze_and_print(data, k_values)

    # Save
    save_path = args.output
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    torch.save(data, save_path)
    size_mb = os.path.getsize(save_path) / 1e6
    print(f"\n{'='*60}")
    print(f"Full data saved → {save_path} ({size_mb:.1f} MB)")
    print(f"Keys: masses [S,B,L,H,N,K], min_k_80/50/90/95/99 [S,B,L,H,N], t_schedule [S]")
    print(f"Done!")


if __name__ == '__main__':
    main()
