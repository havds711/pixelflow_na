#!/usr/bin/env python3
"""
Measure Effective Receptive Field (ERF) and Attention Distance Distribution for SiT.

Methodology:
  - ERF: ΔConvFusion §3 — measure how attention decays with spatial distance,
         fit Gaussian, report radius (σ).
  - Distance: PiT §3 — weighted attention score by Euclidean distance between
              token pairs, report cumulative distribution P(distance ≤ k).

Usage:
  # Measure on pretrained SiT-XL/2 with full attention
  python measure.py --ckpt checkpoints/SiT-XL-2-256.pt \\
                    --model SiT-XL/2 --attn_type full \\
                    --n_samples 16

  # Measure on NA fine-tuned model
  python measure.py --ckpt checkpoints/sit_na_k7.pt \\
                    --model SiT-XL/2 --attn_type na --na_kernel_size 7 \\
                    --n_samples 16
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import os
import json
from tqdm import tqdm
from typing import Optional

from models import SiT, SiT_models


def parse_args():
    parser = argparse.ArgumentParser(description="Measure ERF and Distance Distribution for SiT")

    # Model
    parser.add_argument('--ckpt', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--model', type=str, default='SiT-XL/2',
                        choices=list(SiT_models.keys()),
                        help='SiT model variant')
    parser.add_argument('--attn_type', type=str, default='full',
                        choices=['full', 'na'], help='Attention type (must match checkpoint)')
    parser.add_argument('--na_kernel_size', type=int, default=7,
                        help='NA kernel size (must match checkpoint if attn_type=na)')
    parser.add_argument('--num_classes', type=int, default=1000, help='Number of classes')

    # Image/latent dims
    parser.add_argument('--image_size', type=int, default=256,
                        help='Original image resolution (latent = image_size // 8)')
    parser.add_argument('--latent_size', type=int, default=32,
                        help='Latent spatial size (default 32 for 256x256 with VAE f=8)')

    # Measurement
    parser.add_argument('--n_samples', type=int, default=16,
                        help='Number of samples to measure on')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Output
    parser.add_argument('--output', type=str, default='outputs', help='Output directory')

    # Hardware
    parser.add_argument('--device', type=str, default='cuda', help='Device')

    return parser.parse_args()


# ---------------------------------------------------------------------------
# ERF Measurement (ΔConvFusion method)
# ---------------------------------------------------------------------------

def measure_erf(
    model: SiT,
    n_samples: int = 16,
    latent_size: int = 32,
    device: str = "cuda",
    t_values: Optional[list] = None,
    seed: int = 42,
) -> dict:
    """
    Measure Effective Receptive Field per layer.

    Methodology (from ΔConvFusion §3):
      1. Forward random latents through the model at various timesteps t
      2. Extract attention weights from each layer
      3. For each query position, compute weighted average attention vs spatial distance
      4. ERF = sqrt(weighted mean squared distance)

    Returns:
        dict with keys:
          - 'layer_erf': dict of ERF values per layer
          - 'mean_erf': overall mean ERF
          - 'per_t_erf': dict mapping t -> mean ERF at that timestep
    """
    if t_values is None:
        t_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    torch.manual_seed(seed)
    model.eval()
    grid_size = model.grid_size

    # Accumulate attention maps: per-layer and per-t
    all_attn_maps = {}       # {layer_idx: [sample_attn_maps]}
    per_t_attn_maps = {}     # {t_val: {layer_idx: [sample_attn_maps]}}

    for t_val in t_values:
        # Latent space: use randn (VAE latents are roughly normal-distributed)
        x_0 = torch.randn(n_samples, 4, latent_size, latent_size, device=device)
        x_1 = torch.randn(n_samples, 4, latent_size, latent_size, device=device)
        x_t = (1 - t_val) * x_0 + t_val * x_1
        t_tensor = torch.full((n_samples,), t_val, device=device)

        attn_list = model.get_attention_weights(x_t, t_tensor, y=None)

        if t_val not in per_t_attn_maps:
            per_t_attn_maps[t_val] = {}

        for layer_idx, attn in enumerate(attn_list):
            if attn is None:
                continue
            avg_attn = attn.mean(dim=(0, 1))  # [N, N]

            # Per-layer (all t merged)
            if layer_idx not in all_attn_maps:
                all_attn_maps[layer_idx] = []
            all_attn_maps[layer_idx].append(avg_attn.detach().cpu())

            # Per-t per-layer
            if layer_idx not in per_t_attn_maps[t_val]:
                per_t_attn_maps[t_val][layer_idx] = []
            per_t_attn_maps[t_val][layer_idx].append(avg_attn.detach().cpu())

    coords = _build_coord_grid(grid_size)

    def _compute_erf_for_maps(attn_maps_dict):
        """Compute mean ERF across all layers for a given set of attention maps."""
        layer_vals = []
        for layer_idx, attn_maps in attn_maps_dict.items():
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

    # Per-layer ERF (all t merged)
    layer_erf = {}
    for layer_idx, attn_maps in all_attn_maps.items():
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
                'mean': np.mean(erf_values),
                'std': np.std(erf_values),
                'median': np.median(erf_values),
            }

    # Per-t ERF
    per_t_erf = {}
    for t_val in t_values:
        per_t_erf[float(t_val)] = float(_compute_erf_for_maps(per_t_attn_maps[t_val]))

    # Overall mean
    all_means = [v['mean'] for v in layer_erf.values()]
    mean_erf = np.mean(all_means) if all_means else 0.0

    return {
        'layer_erf': layer_erf,
        'mean_erf': mean_erf,
        'per_t_erf': per_t_erf,
        'grid_size': grid_size,
    }


# ---------------------------------------------------------------------------
# Distance Distribution Measurement (PiT method)
# ---------------------------------------------------------------------------

def measure_distance_distribution(
    model: SiT,
    n_samples: int = 16,
    latent_size: int = 32,
    device: str = "cuda",
    t_values: Optional[list] = None,
    max_distance: Optional[int] = None,
    seed: int = 42,
) -> dict:
    """
    Measure token interaction distance distribution.

    Methodology (from PiT §3):
      1. Extract attention weights from all layers
      2. For each token pair (i,j), compute Euclidean distance on the 2D grid
      3. Weight by attention score → get P(distance = d)
      4. Compute cumulative distribution P(distance ≤ k)

    Returns:
        dict with keys:
          - 'distance_hist': weighted histogram of distances
          - 'cumulative': cumulative distribution P(d ≤ k)
          - 'p99_distance', 'p95_distance', 'p50_distance'
          - 'mean_distance': mean attention-weighted distance
          - 'per_layer': per-layer distance stats
          - 'per_t_distance': per-t distance stats
    """
    if t_values is None:
        t_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    torch.manual_seed(seed)
    model.eval()
    grid_size = model.grid_size
    N = grid_size * grid_size

    if max_distance is None:
        # Max possible Euclidean distance on the grid (diagonal)
        max_distance = int(np.sqrt(2) * (grid_size - 1)) + 1

    coords = _build_coord_grid(grid_size)  # [N, 2]

    # Compute all pairwise distances once
    diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # [N, N, 2]
    distances = torch.sqrt((diff ** 2).sum(dim=-1))  # [N, N]

    # Accumulate
    dist_hist = torch.zeros(max_distance + 1, dtype=torch.float64)
    total_weight = 0.0
    per_layer_stats = {}
    per_t_stats = {}

    def _compute_dist_stats(attn, distances, max_dist):
        """Compute mean and p99 for one attention matrix."""
        N = attn.shape[0]
        dists_all = []
        weights_all = []
        for q in range(N):
            w = attn[q].detach().cpu()
            d = distances[q]
            dists_all.append(d)
            weights_all.append(w)
        dists_all = torch.stack(dists_all)
        weights_all = torch.stack(weights_all)

        # Mean
        total_w = weights_all.sum() + 1e-8
        mean_d = (weights_all * dists_all).sum().item() / total_w.item()

        # P99: sort by distance, compute cumulative weight
        flat_w = weights_all.flatten()
        flat_d = dists_all.flatten()
        order = flat_d.argsort()
        sorted_w = flat_w[order] / total_w
        sorted_d = flat_d[order]
        cum = torch.cumsum(sorted_w, dim=0)
        p99_idx = (cum >= 0.99).nonzero(as_tuple=True)
        p99 = float(sorted_d[p99_idx[0][0]]) if len(p99_idx[0]) > 0 else float(max_dist)

        return mean_d, p99

    for t_val in t_values:
        x_0 = torch.randn(n_samples, 4, latent_size, latent_size, device=device)
        x_1 = torch.randn(n_samples, 4, latent_size, latent_size, device=device)
        x_t = (1 - t_val) * x_0 + t_val * x_1
        t_tensor = torch.full((n_samples,), t_val, device=device)

        attn_list = model.get_attention_weights(x_t, t_tensor, y=None)

        # Per-t: average attention across all layers and heads
        t_attn_maps = []
        for layer_idx, attn in enumerate(attn_list):
            if attn is None:
                continue
            avg_attn = attn.mean(dim=(0, 1))  # [N, N]
            t_attn_maps.append(avg_attn)

            # Accumulate global histogram
            for q in range(N):
                weights = avg_attn[q].detach().cpu()
                ds = distances[q]
                for d_idx in range(max_distance + 1):
                    mask = (ds >= d_idx) & (ds < d_idx + 1)
                    dist_hist[d_idx] += weights[mask].sum().item()
                mask_far = ds >= max_distance
                dist_hist[max_distance] += weights[mask_far].sum().item()
                total_weight += weights.sum().item()

            # Per-layer stats
            if layer_idx not in per_layer_stats:
                per_layer_stats[layer_idx] = {'mean_dist': [], 'p99': []}
            layer_dists = []
            for q in range(N):
                weights = avg_attn[q].detach().cpu()
                ds = distances[q]
                mean_d = (weights * ds).sum().item() / (weights.sum().item() + 1e-8)
                layer_dists.append(mean_d)
            per_layer_stats[layer_idx]['mean_dist'].append(np.mean(layer_dists))

        # Per-t stats: average over all layers
        if t_attn_maps:
            t_mean_attn = torch.stack(t_attn_maps).mean(dim=0)
            t_mean_d, t_p99 = _compute_dist_stats(t_mean_attn, distances, max_distance)
            per_t_stats[float(t_val)] = {'mean': float(t_mean_d), 'p99': float(t_p99)}

    # Normalize histogram
    if total_weight > 0:
        dist_hist = dist_hist / total_weight

    # Cumulative distribution
    cumulative = torch.cumsum(dist_hist, dim=0)

    # Find key percentiles
    p99 = _find_percentile(cumulative, 0.99)
    p95 = _find_percentile(cumulative, 0.95)
    p50 = _find_percentile(cumulative, 0.50)

    # Mean distance
    bin_centers = torch.arange(max_distance + 1, dtype=torch.float64) + 0.5
    mean_dist = (dist_hist * bin_centers).sum().item()

    # Average per-layer stats
    for k in per_layer_stats:
        per_layer_stats[k]['mean_dist'] = np.mean(per_layer_stats[k]['mean_dist'])
        per_layer_stats[k]['p99'] = np.mean(per_layer_stats[k].get('p99', [0]))

    return {
        'distance_hist': dist_hist.numpy(),
        'cumulative': cumulative.numpy(),
        'p99_distance': p99,
        'p95_distance': p95,
        'p50_distance': p50,
        'mean_distance': mean_dist,
        'max_distance': max_distance,
        'per_layer': per_layer_stats,
        'per_t_distance': per_t_stats,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_coord_grid(grid_size: int, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """Build 2D coordinate grid [N, 2]."""
    rows = torch.arange(grid_size)
    cols = torch.arange(grid_size)
    grid_i, grid_j = torch.meshgrid(rows, cols, indexing='ij')
    coords = torch.stack([grid_i.reshape(-1), grid_j.reshape(-1)], dim=-1).float()
    return coords.to(device)


def _find_percentile(cumulative: torch.Tensor, p: float) -> float:
    """Find the distance value at the given percentile."""
    for i in range(len(cumulative)):
        if cumulative[i] >= p:
            # Linear interpolation
            if i == 0:
                return float(i)
            frac = (p - cumulative[i - 1].item()) / (cumulative[i].item() - cumulative[i - 1].item() + 1e-8)
            return float(i - 1 + frac)
    return float(len(cumulative) - 1)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    ckpt_path: str,
    model_type: str = 'SiT-XL/2',
    attn_type: str = 'full',
    na_kernel_size: int = 7,
    image_size: int = 256,
    num_classes: int = 1000,
    device: str = 'cuda',
) -> SiT:
    """
    Load a SiT model from checkpoint, optionally with custom attention.

    Args:
        ckpt_path: path to .pt checkpoint file
        model_type: key in SiT_models dict (e.g., 'SiT-XL/2')
        attn_type: 'full' or 'na'
        na_kernel_size: kernel size for NA (ignored if attn_type='full')
        image_size: original image resolution (latent = image_size // 8)
        num_classes: number of classes (1000 for ImageNet)
        device: torch device

    Returns:
        Loaded SiT model in eval mode
    """
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Determine checkpoint format
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        # Training checkpoint format: {"model": ..., "ema": ..., "opt": ..., "args": ...}
        state_dict = checkpoint['model']
        print("Loaded from training checkpoint (model weights)")
    elif isinstance(checkpoint, dict) and 'ema' in checkpoint:
        state_dict = checkpoint['ema']
        print("Loaded from training checkpoint (EMA weights)")
    else:
        # Raw state dict (from pretrained download or direct save)
        state_dict = checkpoint
        print("Loaded raw state dict")

    latent_size = image_size // 8

    model = SiT_models[model_type](
        input_size=latent_size,
        num_classes=num_classes,
        attn_type=attn_type,
        na_kernel_size=na_kernel_size,
    ).to(device)

    # FullAttention has identical weight structure to timm.Attention
    # (both use self.qkv and self.proj with same shapes)
    # So strict=True should work when attn_type='full'
    try:
        model.load_state_dict(state_dict, strict=True)
        print("Checkpoint loaded with strict=True (all keys matched)")
    except RuntimeError as e:
        print(f"Warning: strict loading failed, falling back to strict=False")
        print(f"  Missing/unexpected keys: {e}")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA not available, falling back to CPU")

    print("=" * 60)
    print(f"SiT — Measure ERF & Distance Distribution")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Model: {args.model} | Attention: {args.attn_type}")
    if args.attn_type == 'na':
        print(f"  NA kernel_size: {args.na_kernel_size}")
    print(f"Image size: {args.image_size} | Latent: {args.latent_size}x{args.latent_size}")
    print(f"Samples: {args.n_samples} | Device: {device}")
    print("=" * 60)

    # Load model
    model = load_model(
        args.ckpt, args.model, args.attn_type, args.na_kernel_size,
        args.image_size, args.num_classes, device,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded model: {n_params / 1e6:.1f}M params")
    print(f"Grid size: {model.grid_size}x{model.grid_size} = {model.grid_size ** 2} tokens")

    # --- ERF Measurement ---
    print("\n--- ERF Measurement (ΔConvFusion method) ---")
    erf_results = measure_erf(
        model, n_samples=args.n_samples, latent_size=args.latent_size,
        device=device, seed=args.seed,
    )

    print(f"\n  Mean ERF across all layers: {erf_results['mean_erf']:.2f} grid cells")
    print(f"  Per-layer ERF:")
    for layer_idx in sorted(erf_results['layer_erf'].keys()):
        d = erf_results['layer_erf'][layer_idx]
        print(f"    Layer {layer_idx:2d}: ERF = {d['mean']:.2f} ± {d['std']:.2f} (median {d['median']:.2f})")

    print(f"\n  Per-t ERF:")
    for t_val in sorted(erf_results['per_t_erf'].keys()):
        print(f"    t={t_val:.1f}: ERF = {erf_results['per_t_erf'][t_val]:.2f}")

    # --- Distance Distribution ---
    print("\n--- Distance Distribution (PiT method) ---")
    dist_results = measure_distance_distribution(
        model, n_samples=args.n_samples, latent_size=args.latent_size,
        device=device, seed=args.seed,
    )

    print(f"\n  Mean attention-weighted distance: {dist_results['mean_distance']:.2f}")
    print(f"  P50 distance: {dist_results['p50_distance']:.2f}")
    print(f"  P95 distance: {dist_results['p95_distance']:.2f}")
    print(f"  P99 distance: {dist_results['p99_distance']:.2f}")

    print(f"\n  Per-layer mean distance:")
    for layer_idx in sorted(dist_results['per_layer'].keys()):
        d = dist_results['per_layer'][layer_idx]
        print(f"    Layer {layer_idx:2d}: mean_dist = {d['mean_dist']:.2f}")

    # Save results
    result_name = f"measure_{args.model.replace('/', '-')}_{args.attn_type}"
    if args.attn_type == 'na':
        result_name += f"_k{args.na_kernel_size}"
    result_name += ".json"
    result_path = os.path.join(args.output, result_name)

    # Convert numpy arrays to lists for JSON
    results = {
        'erf': {str(k): v for k, v in erf_results['layer_erf'].items()},
        'erf_mean': erf_results['mean_erf'],
        'erf_per_t': erf_results['per_t_erf'],
        'distance': {
            'mean': dist_results['mean_distance'],
            'p50': dist_results['p50_distance'],
            'p95': dist_results['p95_distance'],
            'p99': dist_results['p99_distance'],
            'cumulative': dist_results['cumulative'].tolist(),
        },
        'config': {
            'attn_type': args.attn_type,
            'na_kernel_size': args.na_kernel_size,
            'model': args.model,
            'image_size': args.image_size,
            'latent_size': args.latent_size,
            'ckpt': args.ckpt,
        }
    }

    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {result_path}")


if __name__ == '__main__':
    main()
