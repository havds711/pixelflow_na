#!/usr/bin/env python3
"""
Analyze and plot sweep results.

Usage:
  python analyze.py --results outputs/sweep_results.json --output outputs/figures/
"""

import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot sweep results")
    parser.add_argument('--results', type=str, required=True, help='Sweep results JSON')
    parser.add_argument('--output', type=str, default='outputs/figures', help='Output directory')
    return parser.parse_args()


def plot_fid_vs_gflops(results: dict, save_path: str):
    """FID vs GFLOPs trade-off plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    names, fids, gflops, sizes = [], [], [], []
    for name, r in results.items():
        names.append(name)
        fids.append(r['fid'])
        gflops.append(r['gflops'])
        sizes.append(max(50, (r.get('kernel_size') or 32) * 10))

    scatter = ax.scatter(gflops, fids, c=range(len(names)), cmap='viridis',
                         s=sizes, alpha=0.8, edgecolors='white', linewidth=0.5)
    for i, name in enumerate(names):
        ax.annotate(name, (gflops[i], fids[i]), textcoords="offset points",
                    xytext=(5, 5), fontsize=9)
    ax.set_xlabel('GFLOPs', fontsize=12)
    ax.set_ylabel('FID ↓', fontsize=12)
    ax.set_title('FID vs GFLOPs Trade-off', fontsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {save_path}")


def plot_erf_per_layer(results: dict, save_path: str):
    """Per-layer ERF comparison."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, r in results.items():
        erf_dict = r.get('erf_per_layer', {})
        if not erf_dict:
            continue
        layers = sorted(int(k) for k in erf_dict.keys())
        values = [erf_dict[str(l)] for l in layers]
        ax.plot(layers, values, 'o-', label=name, markersize=4, alpha=0.8)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('ERF (grid cells)', fontsize=12)
    ax.set_title('Effective Receptive Field per Layer', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {save_path}")


def plot_erf_vs_t(results: dict, save_path: str):
    """
    🆕 Per-t ERF: how does ERF change with timestep?
    Key for t-adaptive kernel motivation.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, r in results.items():
        per_t = r.get('per_t_erf', {})
        if not per_t:
            continue
        # Sort by t value
        t_vals = sorted(float(k) for k in per_t.keys())
        erf_vals = [per_t[str(t)] if str(t) in per_t else per_t.get(t, 0) for t in t_vals]
        ax.plot(t_vals, erf_vals, 'o-', label=name, markersize=6, alpha=0.8)

    ax.set_xlabel('Timestep t (1=noise, 0=clean)', fontsize=12)
    ax.set_ylabel('ERF (grid cells)', fontsize=12)
    ax.set_title('ERF vs Timestep — Does model need bigger field at high t?', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # t=1 (noise) on left, t=0 (clean) on right
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {save_path}")


def plot_distance_vs_t(results: dict, save_path: str):
    """
    🆕 Per-t distance: how does attention distance distribution change with t?
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for name, r in results.items():
        per_t = r.get('per_t_distance', {})
        if not per_t:
            continue
        t_vals = sorted(float(k) for k in per_t.keys())
        mean_vals = []
        p99_vals = []
        for t in t_vals:
            d = per_t[str(t)] if str(t) in per_t else per_t.get(t, {})
            if isinstance(d, dict):
                mean_vals.append(d.get('mean', 0))
                p99_vals.append(d.get('p99', 0))
            else:
                mean_vals.append(0)
                p99_vals.append(0)

        ax1.plot(t_vals, mean_vals, 'o-', label=name, markersize=6, alpha=0.8)
        ax2.plot(t_vals, p99_vals, 'o-', label=name, markersize=6, alpha=0.8)

    ax1.set_xlabel('Timestep t', fontsize=12)
    ax1.set_ylabel('Mean Distance (grid cells)', fontsize=12)
    ax1.set_title('Mean Attention Distance vs t', fontsize=14)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()

    ax2.set_xlabel('Timestep t', fontsize=12)
    ax2.set_ylabel('P99 Distance (grid cells)', fontsize=12)
    ax2.set_title('P99 Attention Distance vs t', fontsize=14)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()

    fig.suptitle('Token Interaction Distance vs Timestep', fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {save_path}")


def plot_distance_cumulative(results: dict, save_path: str):
    """Cumulative distance distribution (PiT-style)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, r in results.items():
        cum = r.get('distance_cumulative', None)
        if cum is None:
            continue
        ax.plot(cum, label=name, alpha=0.8)
    ax.axhline(y=0.99, color='red', linestyle='--', alpha=0.5, label='99%')
    ax.axhline(y=0.95, color='orange', linestyle='--', alpha=0.5, label='95%')
    ax.set_xlabel('Distance (grid cells)', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title('Token Interaction Distance Distribution (PiT-style)', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {save_path}")


def plot_summary_table(results: dict, save_path: str):
    """Clean summary table."""
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('off')
    headers = ['Variant', 'FID ↓', 'GFLOPs', 'ERF', 'Dist Mean', 'Dist P99']
    rows = []
    for name in ['full', 'na3', 'na5', 'na7', 'na11', 'na15']:
        if name not in results:
            continue
        r = results[name]
        rows.append([
            name,
            f"{r['fid']:.2f}",
            f"{r['gflops']:.2f}",
            f"{r['erf_mean']:.2f}",
            f"{r['distance_mean']:.2f}",
            f"{r['distance_p99']:.2f}",
        ])

    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)
    ax.set_title('Experiment Summary', fontsize=14, pad=20)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {save_path}")


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    with open(args.results, 'r') as f:
        results = json.load(f)

    plot_fid_vs_gflops(results, os.path.join(args.output, 'fid_vs_gflops.png'))
    plot_erf_per_layer(results, os.path.join(args.output, 'erf_per_layer.png'))

    # 🆕 Per-t analysis (key for t-adaptive kernel motivation)
    plot_erf_vs_t(results, os.path.join(args.output, 'erf_vs_t.png'))
    plot_distance_vs_t(results, os.path.join(args.output, 'distance_vs_t.png'))

    if any(r.get('distance_cumulative') for r in results.values()):
        plot_distance_cumulative(results, os.path.join(args.output, 'distance_cumulative.png'))

    plot_summary_table(results, os.path.join(args.output, 'summary_table.png'))
    print(f"\nAll figures saved to {args.output}/")


if __name__ == '__main__':
    main()
