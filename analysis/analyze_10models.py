#!/usr/bin/env python3
"""
Comprehensive analysis of all 10 models — including 4 new models:
  PixArt-α, PixArt-Σ, SD 1.5, SD XL

Usage:
  python analysis/analyze_10models.py
"""

import numpy as np
import torch
import os
import sys

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')

# ─── Model data paths ───
MODELS = {
    'SiT-XL/2':    'attention_locality_sit_xl/attention_locality_sit_xl.pt',
    'SiT-B/2':     'attention_locality_sit_b/attention_locality_sit_b.pt',
    'SiT-L/2':     'attention_locality_sit_l/attention_locality_sit_l.pt',
    'PixDiT':      'attention_locality_pixeldit/attention_locality_pixeldit.pt',
    'DiT-XL/2':    'attention_locality_dit_xl/attention_locality_dit_xl.pt',
    'MDTv2-XL/2':  'attention_locality_mdtv2_xl/attention_locality_mdtv2_xl.pt',
    'PixArt-α':    'attention_locality_pixart_alpha/attention_locality_pixart_alpha.pt',
    'PixArt-Σ':    'attention_locality_pixart_sigma/attention_locality_pixart_sigma.pt',
    'SD 1.5':      'attention_locality_sd15/attention_locality_sd15.pt',
    'SD XL':       'attention_locality_sdxl/attention_locality_sdxl.pt',
}

# ─── Model metadata ───
META = {
    'SiT-XL/2':   {'family': 'SiT', 'framework': 'FM', 'arch': 'DiT', 'params': '675M', 'attn_type': 'self'},
    'SiT-B/2':    {'family': 'SiT', 'framework': 'FM', 'arch': 'DiT', 'params': '130M', 'attn_type': 'self'},
    'SiT-L/2':    {'family': 'SiT', 'framework': 'FM', 'arch': 'DiT', 'params': '450M', 'attn_type': 'self'},
    'PixDiT':     {'family': 'PixDiT', 'framework': 'FM', 'arch': 'Dual DiT', 'params': '675M', 'attn_type': 'self'},
    'DiT-XL/2':   {'family': 'DiT', 'framework': 'DDPM', 'arch': 'DiT', 'params': '675M', 'attn_type': 'self'},
    'MDTv2-XL/2': {'family': 'MDT', 'framework': 'DDPM', 'arch': 'Masked DiT', 'params': '675M', 'attn_type': 'self'},
    'PixArt-α':   {'family': 'PixArt', 'framework': 'DDPM', 'arch': 'DiT+cross', 'params': '600M', 'attn_type': 'self+cross'},
    'PixArt-Σ':   {'family': 'PixArt', 'framework': 'DDPM', 'arch': 'DiT+cross', 'params': '600M', 'attn_type': 'self+cross'},
    'SD 1.5':     {'family': 'SD', 'framework': 'DDPM', 'arch': 'UNet+cross', 'params': '860M', 'attn_type': 'self+cross'},
    'SD XL':      {'family': 'SD', 'framework': 'DDPM', 'arch': 'UNet+cross', 'params': '2.6B', 'attn_type': 'self+cross'},
}


def load_data(path):
    """Load .pt file."""
    return torch.load(path, map_location='cpu', weights_only=False)


def compute_k_pass_rate(min_k_80, k_values):
    """Compute fraction of (step, image, layer, head, query) where min_k_80 <= k."""
    # min_k_80: [S, B, L, H, N] uint8
    counts = {}
    for k in k_values:
        counts[k] = (min_k_80 <= k).float().mean().item()
    return counts


def compute_mean_min_k(min_k_80):
    """Average min_k_80 across all dims."""
    return min_k_80.float().mean().item()


def analyze_standard(data, key, k_values=None):
    """Analyze standard DiT-style data format."""
    masses = data['masses']       # [S, B, L, H, N, K] float16
    min_k_80 = data['min_k_80']   # [S, B, L, H, N] uint8

    if k_values is None:
        k_values = [1, 3, 5, 7, 9, 11, 13, 15]

    S, B, L, H, N, n_k_slots = masses.shape

    results = {}
    # Determine actual k values
    actual_kv = data.get('k_values', None)
    if actual_kv is not None:
        if isinstance(actual_kv, torch.Tensor):
            actual_kv = actual_kv.tolist()
    else:
        # Standard models: k = [1, 3, 5, 7, 9, 11, 13, 15]
        actual_kv = [1, 3, 5, 7, 9, 11, 13, 15]

    # Overall k-pass rates
    for ki, k in enumerate(actual_kv):
        if ki < n_k_slots:
            results[f'k={k}_pass'] = (min_k_80 <= k).float().mean().item()

    results['mean_min_k_80'] = min_k_80.float().mean().item()
    results['num_layers'] = L
    results['num_heads'] = H
    results['num_tokens'] = N
    results['num_steps'] = S

    # Per-layer breakdown
    layer_pass = {}
    for l in range(L):
        for ki, k in enumerate(actual_kv):
            if ki < n_k_slots:
                pass_rate = (min_k_80[:, :, l, :, :] <= k).float().mean().item()
                layer_pass.setdefault(k, []).append(pass_rate)

    # Depth analysis: early vs late layers
    third = L // 3
    early_layers = slice(0, third)
    late_layers = slice(L - third, L)

    results['early_vs_late'] = {}
    for k in [7, 15]:
        if k in actual_kv and actual_kv.index(k) < n_k_slots:
            early_pass = (min_k_80[:, :, early_layers, :, :] <= k).float().mean().item()
            late_pass = (min_k_80[:, :, late_layers, :, :] <= k).float().mean().item()
            results['early_vs_late'][f'k={k}'] = {'early': early_pass, 'late': late_pass}

    return results


def analyze_sd(data):
    """Analyze SD-style multi-resolution data format."""
    by_grid = data.get('by_grid', {})
    results = {'by_grid': {}}

    all_min_k_80 = []
    total_heads = 0
    total_layers = 0

    for gs in sorted(by_grid.keys()):
        st = by_grid[gs]
        masses = st['masses']         # [S, B, L, H, N, K]
        min_k_80 = st['min_k_80']     # [S, B, L, H, N]
        kv = st['k_values']

        S, B, L, H, N, K = masses.shape
        total_layers += L
        total_heads += H

        grid_results = {}
        for k in kv:
            grid_results[f'k={k}_pass'] = (min_k_80 <= k).float().mean().item()

        grid_results['mean_min_k_80'] = min_k_80.float().mean().item()
        grid_results['num_layers'] = L
        grid_results['num_heads'] = H
        grid_results['num_tokens'] = N

        results['by_grid'][gs] = grid_results

        # Collect for overall stats (weighted)
        all_min_k_80.append(min_k_80)

    # Overall (weighted by layers × heads × tokens)
    if all_min_k_80:
        all_flat = torch.cat([m.float().flatten() for m in all_min_k_80])
        results['overall_mean_min_k_80'] = all_flat.mean().item()

        # Weighted k-pass rates
        for k in [1, 3, 5, 7, 9, 11, 13, 15]:
            pass_rate = (all_flat <= k).float().mean().item()
            results[f'overall_k={k}_pass'] = pass_rate

    results['total_layers'] = total_layers
    results['grid_sizes'] = sorted(by_grid.keys())

    return results


def main():
    print("=" * 90)
    print("Attention Locality: 10-Model Comprehensive Analysis")
    print("(Includes 4 new models: PixArt-α, PixArt-Σ, SD 1.5, SD XL)")
    print("=" * 90)

    all_results = {}

    for model_name, rel_path in MODELS.items():
        full_path = os.path.join(OUTPUT_DIR, rel_path)
        if not os.path.exists(full_path):
            print(f"\n  ⚠ {model_name}: file not found at {full_path}")
            continue

        print(f"\n{'─' * 70}")
        print(f"  {model_name} ({META[model_name]['arch']}, {META[model_name]['framework']})")
        print(f"{'─' * 70}")

        try:
            data = load_data(full_path)
        except Exception as e:
            print(f"  Error loading: {e}")
            continue

        if 'by_grid' in data:
            # SD-style multi-resolution
            results = analyze_sd(data)
            all_results[model_name] = results

            print(f"  Grids: {results['grid_sizes']}")
            for gs, gr in results['by_grid'].items():
                print(f"  ── Grid {gs}×{gs} ({gr['num_tokens']} tokens), "
                      f"{gr['num_layers']} layers, {gr['num_heads']} heads ──")
                k_pass_7 = gr.get('k=7_pass', None)
                k_pass_15 = gr.get('k=15_pass', None)
                if k_pass_7 is not None:
                    print(f"      k=7 >80%: {k_pass_7*100:.1f}%")
                if k_pass_15 is not None:
                    print(f"      k=15 >80%: {k_pass_15*100:.1f}%")
                print(f"      mean min_k_80: {gr['mean_min_k_80']:.2f}")

            # Overall SD score
            print(f"  ── Overall ──")
            for k in [7, 15]:
                key = f'overall_k={k}_pass'
                if key in results:
                    print(f"      k={k} >80%: {results[key]*100:.1f}%")

        else:
            # Standard DiT-style
            results = analyze_standard(data, model_name)
            all_results[model_name] = results

            print(f"  Layers: {results['num_layers']}, Heads: {results['num_heads']}, "
                  f"Tokens: {results['num_tokens']}, Steps: {results['num_steps']}")
            print(f"  Core metrics:")
            for k in [1, 3, 5, 7, 9, 11, 13, 15]:
                key = f'k={k}_pass'
                if key in results:
                    print(f"      k={k:>2} >80%: {results[key]*100:5.1f}%")

            print(f"  mean min_k_80: {results['mean_min_k_80']:.2f}")

            # Depth analysis
            if 'early_vs_late' in results:
                evl = results['early_vs_late']
                for k, vals in evl.items():
                    print(f"  Depth ({k}): early={vals['early']*100:.1f}%, "
                          f"late={vals['late']*100:.1f}%, "
                          f"Δ={'late' if vals['late'] > vals['early'] else 'early'}"
                          f"{(max(vals['late'],vals['early'])/min(vals['late'],vals['early'])-1)*100:.0f}%")

    # ─── Cross-model comparison table ───
    print(f"\n\n{'=' * 90}")
    print("CROSS-MODEL COMPARISON")
    print(f"{'=' * 90}")

    # Standard DiT models (single grid) + SD models (composite score)
    print(f"\n{'Model':<16} {'Arch':<16} {'FW':<6} {'k=7 >80%':>10} {'k=15 >80%':>10} "
          f"{'Mean min_k':>10} {'Layers':>7} {'Notes':<20}")
    print("-" * 100)

    for model_name in MODELS:
        if model_name not in all_results:
            continue
        r = all_results[model_name]
        meta = META[model_name]

        if 'overall_k=7_pass' in r:
            k7 = r['overall_k=7_pass'] * 100
            k15 = r['overall_k=15_pass'] * 100
            mean_k = r['overall_mean_min_k_80']
            layers = r['total_layers']
        else:
            k7 = r.get('k=7_pass', 0) * 100
            k15 = r.get('k=15_pass', 0) * 100
            mean_k = r.get('mean_min_k_80', 0)
            layers = r.get('num_layers', 0)

        notes = ""
        if meta['attn_type'] == 'self+cross':
            notes = "has cross-attn"
        if 'SD' in model_name:
            notes += " multi-res"

        print(f"{model_name:<16} {meta['arch']:<16} {meta['framework']:<6} "
              f"{k7:>9.1f}% {k15:>9.1f}% {mean_k:>9.2f} {layers:>6}  {notes}")

    # ─── Key findings ───
    print(f"\n\n{'=' * 90}")
    print("KEY FINDINGS")
    print(f"{'=' * 90}")

    # 1. Cross-attention models vs self-attention only
    print("\n【Cross-Attention Impact】")
    self_only = {k: v for k, v in all_results.items() if META[k]['attn_type'] == 'self'}
    with_cross = {k: v for k, v in all_results.items() if META[k]['attn_type'] == 'self+cross'}

    if self_only and with_cross:
        self_k7 = np.mean([r.get('k=7_pass', r.get('overall_k=7_pass', 0))
                          for r in self_only.values()])
        cross_k7 = np.mean([r.get('k=7_pass', r.get('overall_k=7_pass', 0))
                           for r in with_cross.values()])
        print(f"  Self-attn only models avg k=7 >80%: {self_k7*100:.1f}%")
        print(f"  Models with cross-attn  avg k=7 >80%: {cross_k7*100:.1f}%")
        print(f"  → Cross-attention {'increases' if cross_k7 > self_k7 else 'decreases'} "
              f"self-attention locality (Δ={abs(cross_k7-self_k7)*100:.1f}pp)")

    # 2. UNet vs DiT
    print("\n【Architecture Comparison: DiT vs UNet】")
    dit_models = {k: v for k, v in all_results.items() if 'DiT' in META[k]['arch'] and 'UNet' not in META[k]['arch']}
    unet_models = {k: v for k, v in all_results.items() if 'UNet' in META[k]['arch']}

    if dit_models and unet_models:
        dit_k7 = np.mean([r.get('k=7_pass', r.get('overall_k=7_pass', 0))
                         for r in dit_models.values()])
        unet_k7 = np.mean([r.get('k=7_pass', r.get('overall_k=7_pass', 0))
                          for r in unet_models.values()])
        print(f"  DiT-based models avg k=7 >80%: {dit_k7*100:.1f}%")
        print(f"  UNet-based models  avg k=7 >80%: {unet_k7*100:.1f}%")
        print(f"  → UNet attention is {'more' if unet_k7 > dit_k7 else 'less'} local than DiT")

    # 3. FM vs DDPM
    print("\n【Training Framework: FM vs DDPM】")
    fm = {k: v for k, v in all_results.items() if META[k]['framework'] == 'FM'}
    ddpm = {k: v for k, v in all_results.items() if META[k]['framework'] == 'DDPM'}

    if fm and ddpm:
        fm_k7 = np.mean([r.get('k=7_pass', r.get('overall_k=7_pass', 0)) for r in fm.values()])
        ddpm_k7 = np.mean([r.get('k=7_pass', r.get('overall_k=7_pass', 0)) for r in ddpm.values()])
        print(f"  FM models   avg k=7 >80%: {fm_k7*100:.1f}%")
        print(f"  DDPM models avg k=7 >80%: {ddpm_k7*100:.1f}%")

    # 4. PixArt-α vs PixArt-Σ (training recipe)
    print("\n【PixArt: α vs Σ (training recipe)】")
    if 'PixArt-α' in all_results and 'PixArt-Σ' in all_results:
        a = all_results['PixArt-α']
        s = all_results['PixArt-Σ']
        a_k7 = a.get('k=7_pass', 0) * 100
        s_k7 = s.get('k=7_pass', 0) * 100
        print(f"  PixArt-α k=7 >80%: {a_k7:.1f}%")
        print(f"  PixArt-Σ k=7 >80%: {s_k7:.1f}%")
        print(f"  → Σ vs α: Δ={abs(s_k7-a_k7):.1f}pp")

    # 5. SD scale effect
    print("\n【SD Scale: 1.5 vs XL】")
    if 'SD 1.5' in all_results and 'SD XL' in all_results:
        sd15 = all_results['SD 1.5']
        sdxl = all_results['SD XL']
        sd15_k7 = sd15.get('overall_k=7_pass', 0) * 100
        sdxl_k7 = sdxl.get('overall_k=7_pass', 0) * 100
        print(f"  SD 1.5 k=7 >80%: {sd15_k7:.1f}%")
        print(f"  SD XL  k=7 >80%: {sdxl_k7:.1f}%")

    # 6. SD multi-resolution breakdown
    print("\n【SD Multi-Resolution Locality】")
    for model_name in ['SD 1.5', 'SD XL']:
        if model_name in all_results and 'by_grid' in all_results[model_name]:
            r = all_results[model_name]
            print(f"\n  {model_name}:")
            for gs in sorted(r['by_grid'].keys()):
                gr = r['by_grid'][gs]
                k_best = None
                for k in sorted([int(k.replace('k=', '').replace('_pass', ''))
                                for k in gr if k.startswith('k=')]):
                    if gr.get(f'k={k}_pass', 0) > 0.8:
                        k_best = k
                        break
                print(f"    {gs}×{gs}: {gr['num_layers']} layers, {gr['num_heads']} heads, "
                      f"k>80% = {k_best or '>' + str(max(gr.get('k_values', [])))}")

    print(f"\n{'=' * 90}")
    print("Analysis complete!")
    print(f"{'=' * 90}")


if __name__ == '__main__':
    main()
