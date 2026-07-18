#!/usr/bin/env python3
"""
PixDiT c2i — Flow Matching 推理中 Attention 局部性测量

不改推理逻辑，只在每个 RotaryAttention forward 时抓取 post-softmax weights，
计算每个 query 在不同 k×k 窗口内的累积 attention mass。

用法:
  conda activate pixel
  python c2i/measure_attention_locality.py --num_steps 25 --device cuda:2
"""

import argparse, os, sys, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from pixdit_core.pixeldit_c2i import PixDiT
from pixdit_core.modules import apply_rotary_emb, RotaryAttention


# ═══════════════════════════════════════════════════════════════════════
# 1. Monkey-patch: 不改 forward 逻辑，只多抓 attention weights
# ═══════════════════════════════════════════════════════════════════════

_ORIG_FORWARDS = {}
_COLLECTED_ATTN = {}   # id(module) → attention_weight_tensor


def _patched_forward(self, x, pos, mask=None):
    """和原始 forward 完全一致，只多保存 attn_weights 到 _COLLECTED_ATTN"""
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
    qkv = qkv.permute(2, 0, 1, 3, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    q = self.q_norm(q)
    k = self.k_norm(k)
    q, k = apply_rotary_emb(q, k, freqs_cis=pos)
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()

    scale = self.head_dim ** -0.5
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
    if mask is not None:
        attn_weights = attn_weights + mask
    attn_weights = F.softmax(attn_weights, dim=-1)

    # 唯一改动：保存 attention weights
    _COLLECTED_ATTN[id(self)] = attn_weights.detach()

    x = torch.matmul(attn_weights, v)
    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def patch_attention(model):
    """替换所有 RotaryAttention 的 forward 为可抓取 weights 的版本"""
    _COLLECTED_ATTN.clear()
    for blk in list(model.patch_blocks) + list(model.pixel_blocks):
        attn = blk.attn
        if isinstance(attn, RotaryAttention):
            key = id(attn)
            if key not in _ORIG_FORWARDS:
                _ORIG_FORWARDS[key] = attn.forward
            attn.forward = _patched_forward.__get__(attn, type(attn))
    print(f"  Patched {len(set(id(b.attn) for b in list(model.patch_blocks)+list(model.pixel_blocks)))} RotaryAttention modules")


def collect_attn_weights(model):
    """取出本次 forward 收集到的所有 attention weights（按层序）"""
    result = []
    for blk in list(model.patch_blocks) + list(model.pixel_blocks):
        key = id(blk.attn)
        result.append(_COLLECTED_ATTN.get(key, None))
    return result


# ═══════════════════════════════════════════════════════════════════════
# 2. 预计算 k×k 窗口索引（边界平移，公平比对）
# ═══════════════════════════════════════════════════════════════════════

def precompute_window_indices(grid_size, k_values):
    H = W = grid_size
    indices = {}
    for k in k_values:
        half = k // 2
        idx_mat = torch.zeros(H * W, k * k, dtype=torch.long)
        for qi in range(H):
            for qj in range(W):
                ci = max(half, min(qi, H - 1 - half))
                cj = max(half, min(qj, W - 1 - half))
                r0, r1 = ci - half, ci + half + 1
                c0, c1 = cj - half, cj + half + 1
                pos = 0
                for r in range(r0, r1):
                    for c in range(c0, c1):
                        idx_mat[qi * W + qj, pos] = r * W + c
                        pos += 1
        indices[k] = idx_mat
    return indices


# ═══════════════════════════════════════════════════════════════════════
# 3. 模型加载
# ═══════════════════════════════════════════════════════════════════════

def build_model():
    model = PixDiT(
        in_channels=3, num_groups=16, hidden_size=1152,
        pixel_hidden_size=16, patch_depth=26, pixel_depth=4,
        patch_size=16, num_classes=1000, use_pixel_abs_pos=True,
    )
    model.learned_cond = nn.Parameter(torch.zeros(1, model.hidden_size))
    return model


def load_checkpoint(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    cleaned = {}
    for k, v in sd.items():
        new_k = k
        for prefix in ["ema_denoiser.", "denoiser."]:
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix):]
        cleaned[new_k] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)} (e.g. learned_cond)")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
    return model


# ═══════════════════════════════════════════════════════════════════════
# 4. 推理 + 采集（与 eval_attention_window 完全一致的采样逻辑）
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_sampling_and_collect(model, class_labels, num_steps=25, k_values=None,
                              device="cuda", seed=42):
    if k_values is None:
        k_values = [1, 3, 5, 7, 9, 11, 13, 15]

    torch.manual_seed(seed)
    model.eval()

    B = len(class_labels)
    labels = torch.tensor(class_labels, device=device)
    grid_size = 16
    N = grid_size * grid_size
    n_layers = len(model.patch_blocks) + len(model.pixel_blocks)  # 26 + 4 = 30
    n_heads = model.num_groups  # 16
    n_k = len(k_values)
    img_size = 256

    # 预计算窗口索引
    print("  Precomputing window indices...")
    gather_idx = precompute_window_indices(grid_size, k_values)
    gather_idx_gpu = {k: v.to(device) for k, v in gather_idx.items()}

    # 存储
    masses = torch.zeros(num_steps, B, n_layers, n_heads, N, n_k, dtype=torch.float16)
    min_k_80 = torch.zeros(num_steps, B, n_layers, n_heads, N, dtype=torch.uint8)
    min_k_50 = torch.zeros(num_steps, B, n_layers, n_heads, N, dtype=torch.uint8)
    min_k_90 = torch.zeros(num_steps, B, n_layers, n_heads, N, dtype=torch.uint8)
    min_k_95 = torch.zeros(num_steps, B, n_layers, n_heads, N, dtype=torch.uint8)
    min_k_99 = torch.zeros(num_steps, B, n_layers, n_heads, N, dtype=torch.uint8)
    t_schedule = torch.zeros(num_steps, dtype=torch.float32)

    # === 与 eval_attention_window.py 完全一致的采样 ===
    timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    x = torch.randn(B, 3, img_size, img_size, device=device)  # 初始噪声 at t=0

    for i in tqdm(range(num_steps), desc="Sampling + attention"):
        t_cur = timesteps[i]
        dt = timesteps[i + 1] - t_cur
        t_schedule[i] = t_cur.item()

        t_batch = torch.full((B,), t_cur, device=device, dtype=torch.float32)

        # 标准 forward（不改推理逻辑），我们的 patch 自动抓 attention
        v = model(x, t_batch, labels)

        # 收集 attention
        attn_list = collect_attn_weights(model)

        # 计算 k×k 窗口 mass
        for layer_idx, attn in enumerate(attn_list):
            if attn is None:
                continue
            # attn: [B, heads, N, N]
            for ki, k in enumerate(k_values):
                idx = gather_idx_gpu[k].unsqueeze(0).unsqueeze(0).expand(B, n_heads, -1, -1)
                gathered = attn.gather(dim=-1, index=idx)
                mass = gathered.sum(dim=-1)
                masses[i, :, layer_idx, :, :, ki] = mass.half().cpu()

        # 向量化 min_k
        m_step = masses[i]
        k_tensor = torch.tensor(k_values, dtype=torch.float32)
        for threshold, storage in [(0.50, min_k_50), (0.80, min_k_80),
                                    (0.90, min_k_90), (0.95, min_k_95),
                                    (0.99, min_k_99)]:
            above = m_step.float() >= threshold
            first_ki = above.float().argmax(dim=-1)
            any_above = above.any(dim=-1)
            storage[i] = (k_tensor[first_ki] * any_above).byte()

        # 标准 Euler step（与 eval 一致）：x ← x + v * dt
        x = x + v * dt

    return {
        'masses': masses,
        'min_k_80': min_k_80, 'min_k_50': min_k_50,
        'min_k_90': min_k_90, 'min_k_95': min_k_95, 'min_k_99': min_k_99,
        't_schedule': t_schedule,
    }


# ═══════════════════════════════════════════════════════════════════════
# 5. 表格输出（复用 pixelflow_na 的分析函数）
# ═══════════════════════════════════════════════════════════════════════

def print_tables(data, k_values):
    masses = data['masses']
    min_80 = data['min_k_80']
    t_sched = data['t_schedule']
    S, B, L, H, N, K = masses.shape
    total_pairs = S * B * H * N
    gs = int(N ** 0.5)
    n_k = len(k_values)

    # 空间区域
    region = np.zeros((gs, gs), dtype=int)
    for i in range(gs):
        for j in range(gs):
            d = min(i, j, gs-1-i, gs-1-j)
            if d <= 1: region[i,j] = 0
            elif d <= 3: region[i,j] = 1
            else: region[i,j] = 2
    region_flat = region.reshape(-1)
    region_counts = np.bincount(region_flat, minlength=3)
    region_names = ['corner', 'edge  ', 'interior']

    # 表1: 全局
    print("\n" + "=" * 100)
    print(f"表1 — 全局：k×k 窗口捕获 attention mass 的比例 ({L} layers × {H} heads × {N} queries)")
    print("=" * 100)
    hdr = f"{'k':<8}" + "".join(f"{p:>10}" for p in [">50%",">80%",">90%",">95%",">99%"])
    print(hdr)
    print("-" * len(hdr))
    for ki, k in enumerate(k_values):
        vals = [(masses[:,:,:,:,:,ki] > th).float().mean().item() for th in [0.5,0.8,0.9,0.95,0.99]]
        flag = " ← 80%+" if vals[1] >= 0.8 else ""
        print(f"k={k:<5} " + "".join(f"{v:>10.3f}" for v in vals) + flag)

    # 表2: 每层
    print(f"\n{'='*100}\n表2 — 每层 k×k >80% 达标比例\n{'='*100}")
    hdr2 = f"{'L':<4}" + "".join(f"{'k='+str(k):>10}" for k in k_values) + f"  {'推荐k':>6}"
    print(hdr2)
    print("-" * len(hdr2))
    for l in range(L):
        row = f"{l:<4}"
        best = None
        for ki, k in enumerate(k_values):
            r = (masses[:,:,l,:,:,ki] > 0.80).float().mean().item()
            row += f"{r:>10.3f}"
            if best is None and r >= 0.8:
                best = k
        row += f"  {str(best or '>15'):>6}"
        print(row)

    # 表3: Head 差异
    print(f"\n{'='*100}\n表3 — Head 差异：每层 16 head 的 min_k_80 统计\n{'='*100}")
    hdr3 = f"{'L':<4} {'mean':>8} {'std':>8} {'min':>6} {'max':>6}  | 各 head mean min_k"
    print(hdr3)
    print("-" * len(hdr3))
    for l in range(L):
        ld = min_80[:,:,l,:,:].float().flatten()
        ld = ld[ld > 0]
        if len(ld) == 0:
            print(f"{l:<4} N/A")
            continue
        hmeans = []
        for h in range(H):
            hd = min_80[:,:,l,h,:].float().flatten()
            hd = hd[hd > 0]
            hmeans.append(hd.mean().item() if len(hd) > 0 else 0)
        hm_str = " ".join(f"{hm:>5.0f}" for hm in hmeans)
        print(f"{l:<4} {ld.mean():>8.1f} {ld.std():>8.1f} {ld.min():>6.0f} {ld.max():>6.0f}  | {hm_str}")

    # 表4: min_k 分布
    print(f"\n{'='*100}\n表4 — 每层 min_k_80 分布 (%)\n{'='*100}")
    hdr4 = f"{'L':<4}" + "".join(f"{'k='+str(k):>8}" for k in k_values) + f"  {'未达标':>8}"
    print(hdr4)
    print("-" * len(hdr4))
    for l in range(L):
        row = f"{l:<4}"
        mk = min_80[:,:,l,:,:].flatten()
        total = len(mk)
        for k in k_values:
            row += f"{(mk==k).sum().item()/total*100:>8.1f}"
        row += f"  {(mk==0).sum().item()/total*100:>8.1f}"
        print(row)

    # 表5: 空间
    print(f"\n{'='*100}\n表5 — 空间位置 min_k_80 均值\n{'='*100}")
    hdr5 = f"{'L':<4}" + "".join(f"{n:>12}" for n in region_names) + "  int-corner"
    print(hdr5)
    print("-" * len(hdr5))
    for l in range(L):
        row = f"{l:<4}"
        vals = []
        for ri in range(3):
            qm = (region_flat == ri)
            rd = min_80[:,:,l,:,qm].float().flatten()
            rd = rd[rd > 0]
            v = rd.mean().item() if len(rd) > 0 else 0
            vals.append(v)
            row += f"{v:>12.1f}"
        row += f"  {vals[2]-vals[0]:>+12.1f}"
        print(row)


# ═══════════════════════════════════════════════════════════════════════
# 6. Main
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str,
                   default=str(Path(__file__).resolve().parent / 'imagenet256_pixeldit_xl_epoch320.ckpt'))
    p.add_argument('--num_steps', type=int, default=25)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', type=str,
                   default=str(Path(__file__).resolve().parent / 'attention_locality_output'))
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    k_values = [1, 3, 5, 7, 9, 11, 13, 15]

    print("=" * 60)
    print("PixDiT c2i — Attention Locality (Flow Matching sampling)")
    print(f"Steps: {args.num_steps} | Batch: {args.batch_size} | Device: {device}")
    print(f"Grid: 16×16 | Layers: 30 (26 patch + 4 pixel) | Heads: 16")
    print(f"k: {k_values}")
    print("=" * 60)

    print("\nLoading model...")
    model = build_model()
    load_checkpoint(model, args.ckpt)
    model = model.to(device)
    model.eval()
    print(f"  Params: {sum(p.numel() for p in model.parameters())/1e6:.0f}M")

    patch_attention(model)

    class_labels = [207, 360, 387, 974]

    print(f"\nRunning Flow Matching sampling ({args.num_steps} steps)...")
    t0 = time.time()
    data = run_sampling_and_collect(
        model, class_labels, num_steps=args.num_steps,
        k_values=k_values, device=device, seed=args.seed,
    )
    print(f"  Done in {time.time()-t0:.0f}s")

    print_tables(data, k_values)

    save_path = os.path.join(args.output, 'attention_locality_pixeldit.pt')
    torch.save(data, save_path)
    print(f"\nSaved → {save_path} ({os.path.getsize(save_path)/1e6:.0f} MB)")
    print("Done!")


if __name__ == '__main__':
    main()
