#!/usr/bin/env python3
"""
Measure ERF and Distance Distribution on pretrained SiT-XL/2 (latent space).

Zero training cost. Uses random latents as input — ERF and attention distance
are properties of the model's attention pattern, independent of data content.

Usage:
  python measure_sit.py --device cuda:0 --n_samples 32
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import json
from tqdm import tqdm
from typing import Optional
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SiT'))
sys.path.insert(0, os.path.dirname(__file__))

from SiT.models import SiT_XL_2, SiT_B_2, SiT_S_2
from SiT.download import find_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_coord_grid(grid_size: int, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    rows = torch.arange(grid_size)
    cols = torch.arange(grid_size)
    grid_i, grid_j = torch.meshgrid(rows, cols, indexing='ij')
    coords = torch.stack([grid_i.reshape(-1), grid_j.reshape(-1)], dim=-1).float()
    return coords.to(device)


def _find_percentile(cumulative: np.ndarray, p: float) -> float:
    for i in range(len(cumulative)):
        if cumulative[i] >= p:
            if i == 0:
                return float(i)
            frac = (p - cumulative[i - 1]) / (cumulative[i] - cumulative[i - 1] + 1e-8)
            return float(i - 1 + frac)
    return float(len(cumulative) - 1)


# ---------------------------------------------------------------------------
# ERF Measurement (ΔConvFusion method)
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_erf(
    model,
    n_samples: int = 32,
    grid_size: int = 16,
    latent_channels: int = 4,
    device: str = "cuda",
    t_values: list = None,
    seed: int = 42,
) -> dict:
    """
    Measure ERF on SiT model using random latents.

    Returns:
      - layer_erf: {layer_idx: {mean, std, median}}
      - per_t_erf: {t_val: mean_erf}  — 🔑 key data
      - mean_erf: float
    """
    if t_values is None:
        t_values = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]

    torch.manual_seed(seed)
    model.eval()

    # Accumulators: per-layer and per-t
    all_attn_maps = defaultdict(list)
    per_t_attn_maps = defaultdict(lambda: defaultdict(list))

    for t_val in tqdm(t_values, desc="Measuring ERF"):
        latent = torch.randn(n_samples, latent_channels, grid_size * 2, grid_size * 2,
                            device=device)  # 32x32 latent → grid_size patch → 16x16 tokens
        t_tensor = torch.full((n_samples,), t_val, device=device)
        y = torch.randint(0, 1000, (n_samples,), device=device)
        attn_list = model.get_attention_weights(latent, t_tensor, y)

        for layer_idx, attn in enumerate(attn_list):
            if attn is None:
                continue
            avg_attn = attn.mean(dim=(0, 1)).detach().cpu()  # [N, N]

            all_attn_maps[layer_idx].append(avg_attn)
            per_t_attn_maps[t_val][layer_idx].append(avg_attn)

    coords = _build_coord_grid(grid_size)

    def _compute_erf_for_maps(attn_maps_by_layer):
        layer_vals = []
        for layer_idx, attn_maps in sorted(attn_maps_by_layer.items()):
            if not attn_maps:
                continue
            avg_attn = torch.stack(attn_maps).mean(dim=0)
            erf_vals = []
            for q in range(avg_attn.shape[0]):
                weights = avg_attn[q]
                sq_dist = ((coords - coords[q]) ** 2).sum(dim=-1)
                msd = (weights * sq_dist).sum()
                if msd > 0:
                    erf_vals.append(torch.sqrt(msd).item())
            if erf_vals:
                layer_vals.append(np.mean(erf_vals))
        return np.mean(layer_vals) if layer_vals else 0.0

    # Per-layer ERF
    layer_erf = {}
    for layer_idx in sorted(all_attn_maps.keys()):
        attn_maps = all_attn_maps[layer_idx]
        avg_attn = torch.stack(attn_maps).mean(dim=0)
        erf_values = []
        for q in range(avg_attn.shape[0]):
            weights = avg_attn[q]
            sq_dist = ((coords - coords[q]) ** 2).sum(dim=-1)
            msd = (weights * sq_dist).sum()
            if msd > 0:
                erf_values.append(torch.sqrt(msd).item())
        if erf_values:
            layer_erf[layer_idx] = {
                'mean': float(np.mean(erf_values)),
                'std': float(np.std(erf_values)),
                'median': float(np.median(erf_values)),
            }

    # Per-t ERF
    per_t_erf = {}
    for t_val in t_values:
        per_t_erf[float(t_val)] = float(_compute_erf_for_maps(per_t_attn_maps[t_val]))

    all_means = [v['mean'] for v in layer_erf.values()]
    mean_erf = float(np.mean(all_means)) if all_means else 0.0

    return {
        'layer_erf': layer_erf,
        'mean_erf': mean_erf,
        'per_t_erf': per_t_erf,
        'grid_size': grid_size,
    }


# ---------------------------------------------------------------------------
# Distance Distribution Measurement (PiT method)
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_distance_distribution(
    model,
    n_samples: int = 32,
    grid_size: int = 16,
    latent_channels: int = 4,
    device: str = "cuda",
    t_values: list = None,
    seed: int = 42,
) -> dict:
    """
    Measure attention-weighted token interaction distance distribution.
    """
    if t_values is None:
        t_values = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]

    torch.manual_seed(seed)
    model.eval()
    N = grid_size * grid_size
    max_distance = int(np.sqrt(2) * (grid_size - 1)) + 1

    coords = _build_coord_grid(grid_size)
    diff = coords.unsqueeze(1) - coords.unsqueeze(0)
    distances = torch.sqrt((diff ** 2).sum(dim=-1))  # [N, N]

    dist_hist = torch.zeros(max_distance + 1, dtype=torch.float64)
    total_weight = 0.0
    per_layer_stats = {}
    per_t_stats = {}

    def _compute_dist_stats(attn, distances, max_dist):
        N = attn.shape[0]
        all_w = attn.reshape(-1)
        all_d = distances.reshape(-1)
        total_w = all_w.sum() + 1e-8
        mean_d = (all_w * all_d).sum().item() / total_w.item()

        order = all_d.argsort()
        sorted_w = all_w[order] / total_w
        sorted_d = all_d[order]
        cum = torch.cumsum(sorted_w, dim=0)
        p99_idx = (cum >= 0.99).nonzero(as_tuple=True)
        p99 = float(sorted_d[p99_idx[0][0]]) if len(p99_idx[0]) > 0 else float(max_dist)

        return mean_d, p99

    for t_val in tqdm(t_values, desc="Measuring Distance"):
        latent = torch.randn(n_samples, latent_channels, grid_size * 2, grid_size * 2, device=device)
        t_tensor = torch.full((n_samples,), t_val, device=device)
        y = torch.randint(0, 1000, (n_samples,), device=device)
        attn_list = model.get_attention_weights(latent, t_tensor, y)

        t_attn_maps = []
        for layer_idx, attn in enumerate(attn_list):
            if attn is None:
                continue
            avg_attn = attn.mean(dim=(0, 1)).detach().cpu()  # [N, N]
            t_attn_maps.append(avg_attn)

            for q in range(N):
                weights = avg_attn[q]
                for d_idx in range(max_distance + 1):
                    mask = (distances[q] >= d_idx) & (distances[q] < d_idx + 1)
                    dist_hist[d_idx] += weights[mask].sum().item()
                mask_far = distances[q] >= max_distance
                dist_hist[max_distance] += weights[mask_far].sum().item()
                total_weight += weights.sum().item()

            if layer_idx not in per_layer_stats:
                per_layer_stats[layer_idx] = {'mean_dist': [], 'p99': []}
            layer_dists = []
            for q in range(N):
                w = avg_attn[q]
                d = distances[q]
                md = (w * d).sum().item() / (w.sum().item() + 1e-8)
                layer_dists.append(md)
            per_layer_stats[layer_idx]['mean_dist'].append(float(np.mean(layer_dists)))

        if t_attn_maps:
            t_mean_attn = torch.stack(t_attn_maps).mean(dim=0)
            t_mean_d, t_p99 = _compute_dist_stats(t_mean_attn, distances, max_distance)
            per_t_stats[float(t_val)] = {'mean': float(t_mean_d), 'p99': float(t_p99)}

    if total_weight > 0:
        dist_hist = dist_hist / total_weight
    cumulative = torch.cumsum(dist_hist, dim=0)

    for k in per_layer_stats:
        per_layer_stats[k]['mean_dist'] = float(np.mean(per_layer_stats[k]['mean_dist']))
        per_layer_stats[k]['p99'] = float(np.mean(per_layer_stats[k].get('p99', [0])))

    return {
        'distance_hist': dist_hist.numpy().tolist(),
        'cumulative': cumulative.numpy().tolist(),
        'p99_distance': _find_percentile(cumulative.numpy(), 0.99),
        'p95_distance': _find_percentile(cumulative.numpy(), 0.95),
        'p50_distance': _find_percentile(cumulative.numpy(), 0.50),
        'mean_distance': float((dist_hist * torch.arange(max_distance + 1, dtype=torch.float64)).sum().item()),
        'max_distance': max_distance,
        'per_layer': per_layer_stats,
        'per_t_distance': per_t_stats,
    }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_sit_model(ckpt_path: str, device: str = "cuda"):
    """Load pretrained SiT-XL/2."""
    state_dict = find_model(ckpt_path)
    model = SiT_XL_2(input_size=32, num_classes=1000, learn_sigma=True).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded SiT-XL/2: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params")
    print(f"  Grid: {model.grid_size}x{model.grid_size} = {model.grid_size**2} tokens")
    return model


# ---------------------------------------------------------------------------
# Plotting (inline, no dependency on analyze.py)
# ---------------------------------------------------------------------------

def plot_erf_vs_t(per_t_erf: dict, save_path: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t_vals = sorted(float(k) for k in per_t_erf.keys())
    erf_vals = [per_t_erf[t] if t in per_t_erf else per_t_erf[str(t)] for t in t_vals]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t_vals, erf_vals, 'o-', color='#2563eb', markersize=8, linewidth=2)
    ax.fill_between(t_vals, [v * 0.9 for v in erf_vals], [v * 1.1 for v in erf_vals],
                     alpha=0.1, color='#2563eb')
    ax.set_xlabel('Timestep t (0=clean, 1=noise)', fontsize=13)
    ax.set_ylabel('ERF (grid cells)', fontsize=13)
    ax.set_title('Effective Receptive Field vs Timestep\nSiT-XL/2 (pretrained, full attention)',
                 fontsize=14)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)

    # Annotate the range
    erf_range = max(erf_vals) - min(erf_vals)
    ratio = max(erf_vals) / min(erf_vals) if min(erf_vals) > 0 else 0
    ax.text(0.98, 0.05, f'Range: {erf_range:.2f}\nMax/min: {ratio:.2f}x',
            transform=ax.transAxes, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=11)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {save_path}")


def plot_erf_per_layer(layer_erf: dict, save_path: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    layers = sorted(int(k) for k in layer_erf.keys())
    means = [layer_erf[l]['mean'] for l in layers]
    stds = [layer_erf[l]['std'] for l in layers]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(layers, means, 'o-', color='#2563eb', markersize=5, linewidth=1.5)
    ax.fill_between(layers,
                     [m - s for m, s in zip(means, stds)],
                     [m + s for m, s in zip(means, stds)],
                     alpha=0.15, color='#2563eb')
    ax.set_xlabel('Layer', fontsize=13)
    ax.set_ylabel('ERF (grid cells)', fontsize=13)
    ax.set_title('Effective Receptive Field per Layer\nSiT-XL/2 (pretrained, full attention)',
                 fontsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {save_path}")


def plot_distance_vs_t(per_t_distance: dict, save_path: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t_vals = sorted(float(k) for k in per_t_distance.keys())
    mean_vals = []
    p99_vals = []
    for t in t_vals:
        d = per_t_distance[t] if t in per_t_distance else per_t_distance[str(t)]
        mean_vals.append(d['mean'])
        p99_vals.append(d['p99'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(t_vals, mean_vals, 'o-', color='#2563eb', markersize=8, linewidth=2)
    ax1.set_xlabel('Timestep t', fontsize=13)
    ax1.set_ylabel('Mean Distance (grid cells)', fontsize=13)
    ax1.set_title('Mean Attention Distance vs t', fontsize=14)
    ax1.invert_xaxis()
    ax1.grid(True, alpha=0.3)

    ax2.plot(t_vals, p99_vals, 'o-', color='#dc2626', markersize=8, linewidth=2)
    ax2.set_xlabel('Timestep t', fontsize=13)
    ax2.set_ylabel('P99 Distance (grid cells)', fontsize=13)
    ax2.set_title('P99 Attention Distance vs t', fontsize=14)
    ax2.invert_xaxis()
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Token Interaction Distance vs Timestep\nSiT-XL/2 (pretrained, full attention)',
                 fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {save_path}")


def plot_distance_cumulative(cumulative: list, p99: float, p95: float, save_path: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cumulative, color='#2563eb', linewidth=2)
    ax.axhline(y=0.99, color='#dc2626', linestyle='--', alpha=0.5, label=f'99% (d={p99:.1f})')
    ax.axhline(y=0.95, color='#f59e0b', linestyle='--', alpha=0.5, label=f'95% (d={p95:.1f})')
    ax.set_xlabel('Distance (grid cells)', fontsize=13)
    ax.set_ylabel('Cumulative Probability', fontsize=13)
    ax.set_title('Token Interaction Distance Distribution\nSiT-XL/2 (pretrained, full attention)',
                 fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Measure ERF & Distance on pretrained SiT")
    parser.add_argument('--ckpt', type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                            'SiT/checkpoints/SiT-XL-2-256.pt'),
                        help='SiT checkpoint path')
    parser.add_argument('--n_samples', type=int, default=32, help='Number of random latents')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='outputs/sit_measure', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    print("=" * 60)
    print("SiT ERF & Distance Measurement")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Samples: {args.n_samples} | Device: {device}")
    print("=" * 60)

    # Load model
    model = load_sit_model(args.ckpt, device)
    grid_size = model.grid_size  # 16 for SiT-XL/2

    # --- ERF ---
    print("\n--- ERF Measurement (ΔConvFusion method) ---")
    erf_results = measure_erf(
        model, n_samples=args.n_samples, grid_size=grid_size,
        latent_channels=model.in_channels, device=device, seed=args.seed,
    )

    print(f"\n  Mean ERF: {erf_results['mean_erf']:.2f} grid cells")
    print(f"  Per-t ERF:")
    for t in sorted(erf_results['per_t_erf'].keys()):
        print(f"    t={t:.2f}: ERF = {erf_results['per_t_erf'][t]:.3f}")

    erf_range = max(erf_results['per_t_erf'].values()) - min(erf_results['per_t_erf'].values())
    erf_max = max(erf_results['per_t_erf'].values())
    erf_min = min(erf_results['per_t_erf'].values())
    erf_ratio = erf_max / erf_min if erf_min > 0 else 0
    print(f"\n  🔑 ERF range: {erf_range:.3f} | Max/min ratio: {erf_ratio:.2f}x")

    if erf_ratio >= 2.0:
        print(f"  ✅ DIFFERENCE ≥ 2× → Story is solid! Push t-adaptive kernel.")
    elif erf_ratio >= 1.3:
        print(f"  🟡 Moderate difference → worth pursuing, consider cross-model validation")
    else:
        print(f"  🔸 Small difference (<1.3x) → may pivot to system comparison angle")

    print(f"\n  Per-layer ERF (first 5 layers):")
    for layer_idx in sorted(erf_results['layer_erf'].keys())[:5]:
        d = erf_results['layer_erf'][layer_idx]
        print(f"    Layer {layer_idx}: ERF = {d['mean']:.2f} ± {d['std']:.2f}")

    # --- Distance ---
    print("\n--- Distance Distribution (PiT method) ---")
    dist_results = measure_distance_distribution(
        model, n_samples=args.n_samples, grid_size=grid_size,
        latent_channels=model.in_channels, device=device, seed=args.seed,
    )

    print(f"\n  Mean distance: {dist_results['mean_distance']:.2f}")
    print(f"  P50: {dist_results['p50_distance']:.2f} | P95: {dist_results['p95_distance']:.2f} | P99: {dist_results['p99_distance']:.2f}")

    print(f"\n  Per-t distance:")
    for t in sorted(dist_results['per_t_distance'].keys()):
        d = dist_results['per_t_distance'][t]
        print(f"    t={t:.2f}: mean={d['mean']:.2f}, P99={d['p99']:.2f}")

    # --- Plots ---
    print("\n--- Generating plots ---")
    plot_erf_vs_t(erf_results['per_t_erf'], os.path.join(args.output, 'erf_vs_t.png'))
    plot_erf_per_layer(erf_results['layer_erf'], os.path.join(args.output, 'erf_per_layer.png'))
    plot_distance_vs_t(dist_results['per_t_distance'], os.path.join(args.output, 'distance_vs_t.png'))
    plot_distance_cumulative(
        dist_results['cumulative'], dist_results['p99_distance'], dist_results['p95_distance'],
        os.path.join(args.output, 'distance_cumulative.png'),
    )

    # --- Save JSON ---
    results = {
        'config': {
            'model': 'SiT-XL/2',
            'attn_type': 'full',
            'grid_size': grid_size,
            'tokens': grid_size * grid_size,
            'latent_channels': model.in_channels,
            'n_samples': args.n_samples,
        },
        'erf': {
            'mean': erf_results['mean_erf'],
            'per_t': {str(k): v for k, v in erf_results['per_t_erf'].items()},
            'per_layer': {str(k): v for k, v in erf_results['layer_erf'].items()},
            'erf_range': erf_range,
            'erf_ratio': erf_ratio,
        },
        'distance': {
            'mean': dist_results['mean_distance'],
            'p50': dist_results['p50_distance'],
            'p95': dist_results['p95_distance'],
            'p99': dist_results['p99_distance'],
            'cumulative': dist_results['cumulative'],
            'per_t': {str(k): v for k, v in dist_results['per_t_distance'].items()},
            'per_layer': {str(k): {kk: vv for kk, vv in v.items()}
                         for k, v in dist_results['per_layer'].items()},
        },
    }

    json_path = os.path.join(args.output, 'sit_erf_full.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON saved → {json_path}")

    print("\n" + "=" * 60)
    print("Done! Check outputs/sit_measure/ for results.")
    print("=" * 60)


if __name__ == '__main__':
    main()
