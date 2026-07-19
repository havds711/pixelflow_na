#!/usr/bin/env python3
"""
统一分析脚本 — 对所有已测量模型的 Attention Locality 进行详细分析并保存报告。

用法:
  python analysis/analyze_all.py           # 分析所有模型
  python analysis/analyze_all.py --model sit_xl  # 只分析指定模型

模型配置:
  sit_xl   : SiT-XL/2,  FM, ODE 20步, 28层, 16头
  sit_b    : SiT-B/2,   FM, ODE 20步, 12层, 12头
  sit_l    : SiT-L/2,   FM, ODE 20步, 24层, 16头
  pixeldit : PixDiT,    FM, Euler 25步, 30层(26patch+4pixel), 16头
  dit_xl   : DiT-XL/2,  DDPM 250步, 28层, 16头
  mdt_xl   : MDTv2-XL/2,DDPM 250步, 29层(28blocks+1side), 16头
"""

import numpy as np
import torch
import os
import sys
import argparse
from collections import defaultdict

# ─── 模型配置 ───────────────────────────────────────────────────

MODEL_CONFIGS = {
    'sit_xl': {
        'name': 'SiT-XL/2',
        'family': 'SiT',
        'framework': 'Flow Matching (Linear)',
        'sampling': 'ODE Euler 20步, t=1→0',
        'data_path': 'outputs/attention_locality_sit_xl/attention_locality_sit_xl.pt',
        'layers': 28,
        'heads': 16,
        'num_steps': 20,
        'num_images': 4,
        'layer_names': None,  # no special names
        'depth_groups': [
            ('浅层 L0-L8  (9层)',  slice(0, 9)),
            ('中层 L9-L18 (10层)', slice(9, 19)),
            ('深层 L19-L27 (9层)', slice(19, 28)),
        ],
        'step_stages': [
            ('early  t=[1.00→0.75]', slice(0, 6)),
            ('mid    t=[0.70→0.35]', slice(6, 14)),
            ('late   t=[0.30→0.00]', slice(14, 20)),
        ],
    },
    'sit_b': {
        'name': 'SiT-B/2',
        'family': 'SiT',
        'framework': 'Flow Matching (Linear)',
        'sampling': 'ODE Euler 20步, t=1→0',
        'data_path': 'outputs/attention_locality_sit_b/attention_locality_sit_b.pt',
        'layers': 12,
        'heads': 12,
        'num_steps': 20,
        'num_images': 4,
        'layer_names': None,
        'depth_groups': [
            ('浅层 L0-L3  (4层)',  slice(0, 4)),
            ('中层 L4-L7  (4层)',  slice(4, 8)),
            ('深层 L8-L11 (4层)',  slice(8, 12)),
        ],
        'step_stages': [
            ('early  t=[1.00→0.75]', slice(0, 6)),
            ('mid    t=[0.70→0.35]', slice(6, 14)),
            ('late   t=[0.30→0.00]', slice(14, 20)),
        ],
    },
    'sit_l': {
        'name': 'SiT-L/2',
        'family': 'SiT',
        'framework': 'Flow Matching (Linear)',
        'sampling': 'ODE Euler 20步, t=1→0',
        'data_path': 'outputs/attention_locality_sit_l/attention_locality_sit_l.pt',
        'layers': 24,
        'heads': 16,
        'num_steps': 20,
        'num_images': 4,
        'layer_names': None,
        'depth_groups': [
            ('浅层 L0-L7  (8层)',   slice(0, 8)),
            ('中层 L8-L15 (8层)',   slice(8, 16)),
            ('深层 L16-L23 (8层)',  slice(16, 24)),
        ],
        'step_stages': [
            ('early  t=[1.00→0.75]', slice(0, 6)),
            ('mid    t=[0.70→0.35]', slice(6, 14)),
            ('late   t=[0.30→0.00]', slice(14, 20)),
        ],
    },
    'pixeldit': {
        'name': 'PixDiT',
        'family': 'PixDiT',
        'framework': 'Flow Matching',
        'sampling': 'Euler 25步, t=0→1',
        'data_path': 'outputs/attention_locality_pixeldit/attention_locality_pixeldit.pt',
        'layers': 30,
        'heads': 16,
        'num_steps': 25,
        'num_images': 4,
        'layer_names': {l: ('patch' if l < 26 else 'pixel') for l in range(30)},
        'depth_groups': [
            ('浅层 L0-L8   (9层 patch)',  slice(0, 9)),
            ('中层 L9-L17  (9层 patch)',  slice(9, 18)),
            ('深层 L18-L25 (8层 patch)',  slice(18, 26)),
            ('Pixel L26-L29 (4层)',       slice(26, 30)),
        ],
        'step_stages': [
            ('early  t=[0.00→0.32]', slice(0, 8)),
            ('mid    t=[0.36→0.64]', slice(8, 16)),
            ('late   t=[0.68→0.96]', slice(16, 25)),
        ],
    },
    'dit_xl': {
        'name': 'DiT-XL/2',
        'family': 'DiT',
        'framework': 'DDPM',
        'sampling': 'DDPM 250步, t=999→0 (long)',
        'data_path': 'outputs/attention_locality_dit_xl/attention_locality_dit_xl.pt',
        'layers': 28,
        'heads': 16,
        'num_steps': 250,
        'num_images': 4,
        'layer_names': None,
        'depth_groups': [
            ('浅层 L0-L8  (9层)',  slice(0, 9)),
            ('中层 L9-L18 (10层)', slice(9, 19)),
            ('深层 L19-L27 (9层)', slice(19, 28)),
        ],
        'step_stages': [
            ('early  t=[999→667]', slice(0, 83)),
            ('mid    t=[663→334]', slice(83, 166)),
            ('late   t=[330→0]',   slice(166, 250)),
        ],
    },
    'mdt_xl': {
        'name': 'MDTv2-XL/2',
        'family': 'MDTv2',
        'framework': 'DDPM + Masked Training',
        'sampling': 'DDPM 250步, t=999→0 (long)',
        'data_path': 'outputs/attention_locality_mdtv2_xl/attention_locality_mdtv2_xl.pt',
        'layers': 29,
        'heads': 16,
        'num_steps': 250,
        'num_images': 4,
        'layer_names': {l: ('block' if l < 28 else 'side') for l in range(29)},
        'depth_groups': [
            ('浅层 L0-L8   (9层 block)',  slice(0, 9)),
            ('中层 L9-L18  (10层 block)', slice(9, 19)),
            ('深层 L19-L27 (9层 block)',  slice(19, 28)),
            ('Side L28      (1层)',       slice(28, 29)),
        ],
        'step_stages': [
            ('early  t=[999→667]', slice(0, 83)),
            ('mid    t=[663→334]', slice(83, 166)),
            ('late   t=[330→0]',   slice(166, 250)),
        ],
    },
}

K_VALUES = [1, 3, 5, 7, 9, 11, 13, 15]
GS = 16  # grid size


def load_data(config):
    """Load .pt file with torch.load (handles both zip-based and standard formats)."""
    path = os.path.join(os.path.dirname(__file__), '..', config['data_path'])
    if not os.path.exists(path):
        print(f"  ERROR: File not found: {path}")
        return None

    try:
        d = torch.load(path, map_location='cpu', weights_only=True)
        masses = d['masses'].numpy().astype(float)
        min_k_80 = d['min_k_80'].numpy()
        min_k_50 = d['min_k_50'].numpy()
        min_k_90 = d['min_k_90'].numpy()
        min_k_95 = d['min_k_95'].numpy()
        min_k_99 = d['min_k_99'].numpy()
        t_sched = d['t_schedule'].numpy()
        return masses, min_k_80, min_k_50, min_k_90, min_k_95, min_k_99, t_sched
    except Exception as e:
        print(f"  ERROR loading {path}: {e}")
        return None


def spatial_regions():
    """Return flattened region labels for 16x16 grid: 0=corner, 1=edge, 2=interior."""
    region = np.zeros((GS, GS), dtype=int)
    for i in range(GS):
        for j in range(GS):
            d = min(i, j, GS - 1 - i, GS - 1 - j)
            if d <= 1:
                region[i, j] = 0
            elif d <= 3:
                region[i, j] = 1
            else:
                region[i, j] = 2
    return region.reshape(-1)


def analyze_model(model_key, config, masses, min_k_80, min_k_50, min_k_90, min_k_95, min_k_99, t_sched):
    """Generate detailed analysis report for a single model."""
    S, B, L, H, N, K = masses.shape
    region_flat = spatial_regions()
    ri_counts = np.bincount(region_flat, minlength=3)
    region_names = ['corner', 'edge', 'interior']
    total_pairs = S * B * H * N

    lines = []
    def p(s=''):
        lines.append(s)
        print(s)

    p('=' * 130)
    p(f'  {config["name"]} — {config["framework"]} Attention Locality 完整分析')
    p('=' * 130)
    p(f'  框架: {config["framework"]}  |  采样: {config["sampling"]}')
    p(f'  数据维度: [{S} steps × {B} images × {L} layers × {H} heads × {N} queries × {K} k-values]')
    p(f'  k values: {K_VALUES}')
    p(f'  t_schedule: [{t_sched[0]:.2f} → {t_sched[-1]:.2f}], {S} steps')
    p(f'  类别标签: img0=207(golden_retriever) img1=360(otter) img2=387(lesser_panda) img3=974(geyser)')
    p(f'  总测量点: {total_pairs * K:,}')
    p()

    # ═══ 表1: 全局 Summary ═══
    p('=' * 130)
    p('表1 — 全局 Summary: k×k 窗口内累积 attention mass 达标比例')
    p(f'      统计范围: {S} steps × {B} images × {L} layers × {H} heads × {N} queries')
    p('=' * 130)
    hdr = f'{"k":<8} {"累计mass均值":>14} {">50%":>10} {">80%":>10} {">90%":>10} {">95%":>10} {">99%":>10}  {"推荐":>4}'
    p(hdr)
    p('-' * len(hdr))
    for ki, k in enumerate(K_VALUES):
        d = masses[:, :, :, :, :, ki]
        avg_mass = d.mean()
        r50 = (d > 0.50).mean()
        r80 = (d > 0.80).mean()
        r90 = (d > 0.90).mean()
        r95 = (d > 0.95).mean()
        r99 = (d > 0.99).mean()
        flag = ' ★' if r80 >= 0.8 else ''
        p(f'k={k:<5} {avg_mass:>14.4f} {r50:>10.4f} {r80:>10.4f} {r90:>10.4f} {r95:>10.4f} {r99:>10.4f}  {flag}')

    # ═══ 表2: 每层 >80% ═══
    p()
    p('=' * 130)
    p('表2 — 每层 × 每个k: >80% attention mass 达标比例')
    p(f'      (avg over {S} steps × {B} imgs × {H} heads × {N} queries = {total_pairs // L:,} 点/层)')
    p('=' * 130)
    ltype_col = ' type' if config['layer_names'] else ''
    hdr2 = f'{"L":<4}{ltype_col:>6}' + ''.join(f'{"k="+str(k):>9}' for k in K_VALUES) + '  {"推荐k":>8}  {"层均值":>8}  {"分组":>8}'
    p(hdr2)
    p('-' * len(hdr2))

    # Determine depth group for each layer
    def get_depth_group(layer_idx, config):
        for name, sl in config['depth_groups']:
            start = sl.start or 0
            stop = sl.stop or config['layers']
            if start <= layer_idx < stop:
                # Extract short label
                return name.split(' ')[0] if ' ' in name else name[:4]
        return '?'

    for l in range(L):
        ltype_str = ''
        if config['layer_names']:
            lt = config['layer_names'].get(l, '?')
            ltype_str = f'{lt:>6}'
        row = f'{l:<4}{ltype_str}'
        best_k = None
        ratios = []
        for ki, k in enumerate(K_VALUES):
            r = (masses[:, :, l, :, :, ki] > 0.80).mean()
            ratios.append(r)
            row += f'{r:>9.4f}'
            if best_k is None and r >= 0.8:
                best_k = k
        best_str = str(best_k) if best_k else '>15'
        depth = get_depth_group(l, config)
        row += f'  {best_str:>8}  {np.mean(ratios):>8.4f}  {depth:>8}'
        p(row)

    # ═══ 表3: 深度趋势 ═══
    p()
    p('=' * 130)
    p('表3 — 深度趋势: 不同深度分组对比')
    p('=' * 130)
    for name, sl in config['depth_groups']:
        d = masses[:, :, sl, :, :, :]
        d_mk80 = min_k_80[:, :, sl, :, :].astype(float)
        d_nz = d_mk80.flatten()
        d_nz = d_nz[d_nz > 0]
        cov = len(d_nz) / d_mk80.size * 100 if d_mk80.size > 0 else 0
        p(f'\n{name}:  达标率={cov:.1f}%')
        if len(d_nz) > 0:
            p(f'  min_k_80: mean={d_nz.mean():.1f}, std={d_nz.std():.1f}, median={np.median(d_nz):.0f}')
        for ki, k in enumerate(K_VALUES):
            r50 = (d[:, :, :, :, :, ki] > 0.50).mean()
            r80 = (d[:, :, :, :, :, ki] > 0.80).mean()
            r95 = (d[:, :, :, :, :, ki] > 0.95).mean()
            r99 = (d[:, :, :, :, :, ki] > 0.99).mean()
            p(f'  k={k:<4}  >50%={r50:.4f}  >80%={r80:.4f}  >95%={r95:.4f}  >99%={r99:.4f}')

    # ═══ 表4: Head 差异 ═══
    p()
    p('=' * 130)
    p('表4 — Head 差异: 每层各 head 的 min_k_80 统计')
    p('=' * 130)
    hdr4 = f'{"L":<4}' + (f'{"type":<7}' if config['layer_names'] else '') + f'{"层均值":>8} {"std":>8} {"min":>6} {"max":>6}  | 各 head min_k_80 均值'
    p(hdr4)
    p('-' * 130)
    for l in range(L):
        ltype_str = ''
        if config['layer_names']:
            lt = config['layer_names'].get(l, '?')
            ltype_str = f'{lt:<7}'
        layer_data = min_k_80[:, :, l, :, :].astype(float).flatten()
        layer_nz = layer_data[layer_data > 0]
        if len(layer_nz) == 0:
            p(f'{l:<4} {ltype_str} N/A')
            continue
        head_means = []
        for h in range(H):
            hd = min_k_80[:, :, l, h, :].astype(float).flatten()
            hd = hd[hd > 0]
            head_means.append(hd.mean() if len(hd) > 0 else 0)
        hdr = ' '.join(f'{hm:>5.0f}' for hm in head_means)
        p(f'{l:<4} {ltype_str} {layer_nz.mean():>8.1f} {layer_nz.std():>8.1f} '
          f'{layer_nz.min():>6.0f} {layer_nz.max():>6.0f}  | {hdr}')

    # ═══ 表5: 空间位置 ═══
    p()
    p('=' * 130)
    p('表5 — 空间位置差异: corner / edge / interior min_k_80 均值')
    p(f'      query分布: corner(d≤1)={ri_counts[0]}, edge(d2-3)={ri_counts[1]}, interior(d≥4)={ri_counts[2]}')
    p('=' * 130)
    hdr5 = f'{"L":<4}' + (f'{"type":<7}' if config['layer_names'] else '') + ''.join(f'{n:>14}' for n in region_names) + '  {"int-corner Δ":>14}'
    p(hdr5)
    p('-' * len(hdr5))
    for l in range(L):
        ltype_str = ''
        if config['layer_names']:
            lt = config['layer_names'].get(l, '?')
            ltype_str = f'{lt:<7}'
        row = f'{l:<4} {ltype_str}'
        vals = []
        for ri in range(3):
            q_mask = (region_flat == ri)
            rd = min_k_80[:, :, l, :, q_mask].astype(float).flatten()
            rd = rd[rd > 0]
            v = rd.mean() if len(rd) > 0 else 0
            vals.append(v)
            row += f'{v:>14.1f}'
        delta = vals[2] - vals[0]
        row += f'  {delta:>+14.1f}'
        p(row)

    # ═══ 表6: min_k 分布 ═══
    p()
    p('=' * 130)
    p('表6 — min_k_80 分布直方图: 各 k 值占比 (%)')
    p('=' * 130)
    hdr6 = f'{"L":<4}' + (f'{"type":<7}' if config['layer_names'] else '') + ''.join(f'{"k="+str(k):>8}' for k in K_VALUES) + '  {"未达标":>8}'
    p(hdr6)
    p('-' * len(hdr6))
    for l in range(L):
        ltype_str = ''
        if config['layer_names']:
            lt = config['layer_names'].get(l, '?')
            ltype_str = f'{lt:<7}'
        row = f'{l:<4} {ltype_str}'
        layer_mk = min_k_80[:, :, l, :, :].flatten()
        total = len(layer_mk)
        for k in K_VALUES:
            pct = (layer_mk == k).sum() / total * 100
            row += f'{pct:>8.1f}'
        pct_fail = (layer_mk == 0).sum() / total * 100
        row += f'  {pct_fail:>8.1f}'
        p(row)

    # ═══ 表7: 五个阈值 ═══
    p()
    p('=' * 130)
    p('表7 — 五个 min_k 阈值统计对比 (50%/80%/90%/95%/99%)')
    p('=' * 130)
    hdr7 = f'{"阈值":<10} {"全局mean":>10} {"std":>8} {"median":>8} {"Q25":>6} {"Q75":>6} {"达标率":>8}'
    p(hdr7)
    p('-' * len(hdr7))
    for label, data in [('>50%', min_k_50), ('>80%', min_k_80), ('>90%', min_k_90),
                         ('>95%', min_k_95), ('>99%', min_k_99)]:
        d = data.flatten().astype(float)
        d_nz = d[d > 0]
        coverage = len(d_nz) / len(d) * 100
        p(f'{label:<10} {d_nz.mean():>10.1f} {d_nz.std():>8.1f} {np.median(d_nz):>8.0f} '
          f'{np.percentile(d_nz, 25):>6.0f} {np.percentile(d_nz, 75):>6.0f} {coverage:>7.1f}%')

    # ═══ 表8: 步间趋势 ═══
    p()
    p('=' * 130)
    p('表8 — 采样步间趋势: 早/中/晚期 >80%@k=7')
    p('=' * 130)
    ki7 = 3  # index for k=7
    hdr8 = f'{"L":<4}' + (f'{"type":<7}' if config['layer_names'] else '') + ''.join(f'{name:>34}' for name, _ in config['step_stages']) + '  {"early→late":>12}'
    p(hdr8)
    p('-' * len(hdr8))
    for l in range(L):
        ltype_str = ''
        if config['layer_names']:
            lt = config['layer_names'].get(l, '?')
            ltype_str = f'{lt:<7}'
        row = f'{l:<4} {ltype_str}'
        vals = []
        for name, sl in config['step_stages']:
            r = (masses[sl, :, l, :, :, ki7] > 0.80).mean()
            vals.append(r)
            row += f'{r:>34.4f}'
        delta = vals[-1] - vals[0]
        row += f'  {delta:>+12.4f}'
        p(row)
    # 全局
    p()
    for name, sl in config['step_stages']:
        r = (masses[sl, :, :, :, :, ki7] > 0.80).mean()
        p(f'  全局 {name}: {r:.4f}')

    # ═══ 表9: 图像间差异 ═══
    p()
    p('=' * 130)
    p('表9 — 图像间变异性: 4张不同类别图像 >80%@k=7')
    p('=' * 130)
    class_names = {0: '207-golden_retriever', 1: '360-otter', 2: '387-lesser_panda', 3: '974-geyser'}
    # Select representative layers to display
    step_size = max(1, L // 8)
    display_layers = list(range(0, L, step_size))[:8]
    if L - 1 not in display_layers:
        display_layers.append(L - 1)

    hdr9 = f'{"img":<6} {"类别":<22}' + ''.join(f'{"L="+str(l):>10}' for l in display_layers) + f'  {"avg{L}L":>10}'
    p(hdr9)
    p('-' * len(hdr9))
    img_ratios = []
    for b in range(B):
        vals = []
        for l in display_layers:
            r = (masses[:, b, l, :, :, ki7] > 0.80).mean()
            vals.append(r)
        r_all = (masses[:, b, :, :, :, ki7] > 0.80).mean()
        img_ratios.append(r_all)
        row = f'{b:<6} {class_names[b]:<22}'
        for v in vals:
            row += f'{v:>10.4f}'
        row += f'  {r_all:>10.4f}'
        p(row)
    p()
    for l in display_layers:
        vals = [(masses[:, b, l, :, :, ki7] > 0.80).mean() for b in range(B)]
        p(f'  L{l}: 4图 max-min spread = {max(vals)-min(vals):.4f}, std = {np.std(vals):.4f}')

    # ═══ 表10: Per-Head NA 可行性 ═══
    p()
    p('=' * 130)
    p('表10 — Per-Head NA 可行性: 每个 head 满足不同覆盖率的 min_k')
    p('=' * 130)
    # Compute per-head >80% rate: average over steps, images, queries
    r80_per_head = (masses > 0.80).mean(axis=(0, 1, 4))  # [L, H, K]

    for target_pct in [0.5, 0.8]:
        head_optimal_k = np.zeros((L, H), dtype=int)
        for l in range(L):
            for h in range(H):
                found = False
                for ki, k in enumerate(K_VALUES):
                    if r80_per_head[l, h, ki] >= target_pct:
                        head_optimal_k[l, h] = k
                        found = True
                        break
                if not found:
                    head_optimal_k[l, h] = 99
        total_heads = L * H
        p(f'\n  Target: {target_pct*100:.0f}% queries per head reach >80% mass:')
        w_sum = 0
        for k in K_VALUES:
            n = (head_optimal_k == k).sum()
            p(f'    k={k:<4}: {n:>3} heads ({n/total_heads*100:>5.1f}%)')
            w_sum += k * n
        n = (head_optimal_k == 99).sum()
        p(f'    k>15  : {n:>3} heads ({n/total_heads*100:>5.1f}%)')
        w_sum += 15 * n
        w_avg = w_sum / total_heads
        flops_saved = (1 - (w_avg ** 2) / (GS ** 2)) * 100
        p(f'    Weighted avg k: {w_avg:.1f}  →  理论 FLOPs 节省: {flops_saved:.1f}%')

    # ═══ 表11: NA-Friendly Heads 列表 ═══
    p()
    p('=' * 130)
    p('表11 — NA-Friendly Heads (mean min_k_80 ≤ 7)')
    p('=' * 130)
    na_friendly = []
    for l in range(L):
        for h in range(H):
            hd = min_k_80[:, :, l, h, :].astype(float).flatten()
            hd = hd[hd > 0]
            mean_k = hd.mean() if len(hd) > 0 else 99
            if mean_k <= 7:
                lt = config['layer_names'].get(l, '-') if config['layer_names'] else '-'
                na_friendly.append((l, lt, h, mean_k))
                p(f'  L{l} ({lt}) H{h}: mean_min_k={mean_k:.1f}')
    p(f'  Total: {len(na_friendly)} / {L*H} heads ({len(na_friendly)/(L*H)*100:.1f}%)')

    # ═══ 表12: 关键发现汇总 ═══
    p()
    p('=' * 130)
    p('表12 — 关键统计发现汇总')
    p('=' * 130)

    # Best k for 80%
    best_k = None
    for ki, k in enumerate(K_VALUES):
        if (masses[:, :, :, :, :, ki] > 0.80).mean() >= 0.8:
            best_k = k
            break

    p(f'\n  [Locality 强度]')
    for ki, k in enumerate(K_VALUES[:4]):  # k=1,3,5,7
        r = (masses[:, :, :, :, :, ki] > 0.80).mean()
        p(f'    k={k} >80% 达标率: {r:.4f} ({r*100:.1f}%)')
    if best_k:
        p(f'    全局推荐 k≥{best_k} 可达 80%+ attention mass')
    else:
        p(f'    所有 k≤15 均无法全局达到 80%+ (最强 k=15: {(masses[:,:,:,:,:,7]>0.80).mean()*100:.1f}%)')

    p(f'\n  [层级差异]')
    shallow_sl = config['depth_groups'][0][1]
    deep_sl = config['depth_groups'][-1][1]
    shallow = (masses[:, :, shallow_sl, :, :, ki7] > 0.80).mean()
    deep = (masses[:, :, deep_sl, :, :, ki7] > 0.80).mean()
    p(f'    浅层 >80%@k=7: {shallow:.4f}')
    p(f'    深层 >80%@k=7: {deep:.4f}')
    if shallow > 0:
        p(f'    浅/深比: {shallow/deep:.2f}x  {"(reverse locality)" if shallow < deep else "(normal locality)"}')
    else:
        p(f'    浅/深比: N/A (浅层几乎无达标)')

    p(f'\n  [时间步依赖]')
    early_sl = config['step_stages'][0][1]
    late_sl = config['step_stages'][-1][1]
    early = (masses[early_sl, :, :, :, :, ki7] > 0.80).mean()
    late = (masses[late_sl, :, :, :, :, ki7] > 0.80).mean()
    p(f'    早期 >80%@k=7: {early:.4f}')
    p(f'    晚期 >80%@k=7: {late:.4f}')
    p(f'    晚期/早期比: {late/early:.2f}x' if early > 0 else '    晚期/早期比: N/A')

    p(f'\n  [空间差异]')
    corner_nz = min_k_80[:, :, :, :, region_flat == 0].astype(float).flatten()
    edge_nz = min_k_80[:, :, :, :, region_flat == 1].astype(float).flatten()
    interior_nz = min_k_80[:, :, :, :, region_flat == 2].astype(float).flatten()
    for label, data in [('corner', corner_nz), ('edge', edge_nz), ('interior', interior_nz)]:
        d = data[data > 0]
        p(f'    {label} mean min_k_80: {d.mean():.1f}' if len(d) > 0 else f'    {label}: N/A')

    p(f'\n  [Head 多样性]')
    head_stds = []
    for l in range(L):
        hm = []
        for h in range(H):
            hd = min_k_80[:, :, l, h, :].astype(float).flatten()
            hd = hd[hd > 0]
            hm.append(hd.mean() if len(hd) > 0 else 0)
        head_stds.append(np.std(hm))
    p(f'    每层内 {H} head 的 mean min_k std: mean={np.mean(head_stds):.2f}, '
      f'min={np.min(head_stds):.2f}, max={np.max(head_stds):.2f}')

    p(f'\n  [采样步稳定性]')
    step_ratios = [(masses[s, :, :, :, :, ki7] > 0.80).mean() for s in range(S)]
    p(f'    {S}步 >80%@k=7 ratio: mean={np.mean(step_ratios):.4f}, std={np.std(step_ratios):.4f}, '
      f'min={np.min(step_ratios):.4f}, max={np.max(step_ratios):.4f}')

    p(f'\n  [图像间一致性]')
    img_ratios = [(masses[:, b, :, :, :, ki7] > 0.80).mean() for b in range(B)]
    p(f'    {B}图 >80%@k=7 ratio: mean={np.mean(img_ratios):.4f}, std={np.std(img_ratios):.4f}, '
      f'spread={max(img_ratios)-min(img_ratios):.4f}')

    p()
    p('=' * 130)
    p(f'  {config["name"]} 分析完成')
    p('=' * 130)

    return lines


def cross_model_comparison(all_results):
    """Generate cross-model comparison report."""
    p = print
    p()
    p('█' * 130)
    p('█  跨模型综合对比分析')
    p('█' * 130)

    ki7 = 3  # k=7
    K_IDX = {k: i for i, k in enumerate(K_VALUES)}

    # ─── 提取每个模型的关键指标 ───
    models_data = {}
    for model_key, (config, masses, min_k_80, min_k_50, min_k_90, min_k_95, min_k_99, t_sched) in all_results.items():
        S, B, L, H, N, K = masses.shape

        # Compute all key metrics
        metrics = {}
        for ki, k in enumerate(K_VALUES):
            metrics[f'k={k} >80%'] = (masses[:, :, :, :, :, ki] > 0.80).mean()
            metrics[f'k={k} >50%'] = (masses[:, :, :, :, :, ki] > 0.50).mean()
            metrics[f'k={k} avg_mass'] = masses[:, :, :, :, :, ki].mean()

        # Depth
        shallow_sl = config['depth_groups'][0][1]
        deep_sl = config['depth_groups'][-1][1]
        metrics['shallow >80%@k=7'] = (masses[:, :, shallow_sl, :, :, ki7] > 0.80).mean()
        metrics['deep >80%@k=7'] = (masses[:, :, deep_sl, :, :, ki7] > 0.80).mean()
        metrics['reverse_locality'] = metrics['deep >80%@k=7'] > metrics['shallow >80%@k=7']

        # Time dependence
        early_sl = config['step_stages'][0][1]
        late_sl = config['step_stages'][-1][1]
        metrics['early >80%@k=7'] = (masses[early_sl, :, :, :, :, ki7] > 0.80).mean()
        metrics['late >80%@k=7'] = (masses[late_sl, :, :, :, :, ki7] > 0.80).mean()

        # Head diversity
        head_means_all = []
        for l in range(L):
            for h in range(H):
                hd = min_k_80[:, :, l, h, :].astype(float).flatten()
                hd = hd[hd > 0]
                head_means_all.append(hd.mean() if len(hd) > 0 else 0)
        metrics['head_mean_k_global'] = np.mean([x for x in head_means_all if x > 0])
        metrics['na_friendly_heads_pct'] = sum(1 for x in head_means_all if 0 < x <= 7) / (L * H) * 100

        # Image variability
        img_ratios = [(masses[:, b, :, :, :, ki7] > 0.80).mean() for b in range(B)]
        metrics['img_spread'] = max(img_ratios) - min(img_ratios)
        metrics['img_std'] = np.std(img_ratios)

        # Step stability
        step_ratios = [(masses[s, :, :, :, :, ki7] > 0.80).mean() for s in range(S)]
        metrics['step_std'] = np.std(step_ratios)

        # Spatial
        region_flat = spatial_regions()
        corner_nz = min_k_80[:, :, :, :, region_flat == 0].astype(float).flatten()
        interior_nz = min_k_80[:, :, :, :, region_flat == 2].astype(float).flatten()
        corner_nz = corner_nz[corner_nz > 0]
        interior_nz = interior_nz[interior_nz > 0]
        metrics['corner_mean_k'] = corner_nz.mean() if len(corner_nz) > 0 else 0
        metrics['interior_mean_k'] = interior_nz.mean() if len(interior_nz) > 0 else 0

        # Per-head NA FLOPs saving
        r80_per_head = (masses > 0.80).mean(axis=(0, 1, 4))
        head_optimal_k = np.zeros((L, H), dtype=int)
        for l in range(L):
            for h in range(H):
                found = False
                for ki, k in enumerate(K_VALUES):
                    if r80_per_head[l, h, ki] >= 0.5:
                        head_optimal_k[l, h] = k
                        found = True
                        break
                if not found:
                    head_optimal_k[l, h] = 99
        avg_k_50pct = sum(head_optimal_k.flatten()) / (L * H)
        metrics['na_avg_k_50pct'] = min(avg_k_50pct, 15)
        metrics['na_flops_saved_50pct'] = (1 - (metrics['na_avg_k_50pct'] ** 2) / (GS ** 2)) * 100

        models_data[model_key] = metrics

    # ─── 表A: 核心指标对比 ───
    p()
    p('=' * 150)
    p('表A — 核心 Attention Locality 指标对比')
    p('=' * 150)

    compare_keys = [
        ('k=1 >80%', 'k=1 >80%'),
        ('k=3 >80%', 'k=3 >80%'),
        ('k=5 >80%', 'k=5 >80%'),
        ('k=7 >80%', 'k=7 >80%'),
        ('k=9 >80%', 'k=9 >80%'),
        ('k=11 >80%', 'k=11 >80%'),
        ('k=13 >80%', 'k=13 >80%'),
        ('k=15 >80%', 'k=15 >80%'),
        ('shallow >80%@k=7', '浅层>80%@k=7'),
        ('deep >80%@k=7', '深层>80%@k=7'),
        ('early >80%@k=7', '早期>80%@k=7'),
        ('late >80%@k=7', '晚期>80%@k=7'),
        ('head_mean_k_global', 'Head mean k'),
        ('na_friendly_heads_pct', 'NA-friendly heads%'),
        ('img_spread', '图像间spread'),
        ('step_std', '步间std'),
        ('corner_mean_k', 'Corner mean k'),
        ('interior_mean_k', 'Interior mean k'),
        ('na_avg_k_50pct', 'NA avg k(50%)'),
        ('na_flops_saved_50pct', 'NA FLOPs saved%'),
    ]

    model_keys = list(all_results.keys())
    hdr = f'{"指标":<28}' + ''.join(f'{config["name"]:>18}' for _, (config, *_) in all_results.items())
    p(hdr)
    p('-' * len(hdr))

    for key, display_name in compare_keys:
        row = f'{display_name:<28}'
        for mk in model_keys:
            v = models_data[mk].get(key, 0)
            row += f'{v:>18.4f}' if isinstance(v, float) else f'{str(v):>18}'
        p(row)

    # ─── 表B: 模型属性矩阵 ───
    p()
    p('=' * 150)
    p('表B — 模型属性 & 架构差异矩阵')
    p('=' * 150)
    attr_hdr = f'{"属性":<28}' + ''.join(f'{config["name"]:>18}' for _, (config, *_) in all_results.items())
    p(attr_hdr)
    p('-' * len(attr_hdr))
    attrs = [
        ('框架', 'framework'),
        ('采样方式', 'sampling'),
        ('层数', 'layers'),
        ('头数', 'heads'),
        ('步数', 'num_steps'),
    ]
    for label, key in attrs:
        row = f'{label:<28}'
        for mk in model_keys:
            config = all_results[mk][0]
            row += f'{str(config[key]):>18}'
        p(row)

    # ─── 表C: k=7 >80% 排名 ───
    p()
    p('=' * 150)
    p('表C — k=7 >80% 达标率排名 (越高越局部)')
    p('=' * 150)
    ranked = sorted(model_keys, key=lambda mk: models_data[mk]['k=7 >80%'], reverse=True)
    for rank, mk in enumerate(ranked, 1):
        config = all_results[mk][0]
        v = models_data[mk]['k=7 >80%']
        bar = '█' * int(v * 100)
        p(f'  #{rank} {config["name"]:<16} {v:.4f}  {bar}')

    # ─── 表D: 关键发现定性总结 ───
    p()
    p('=' * 150)
    p('表D — 定性发现汇总')
    p('=' * 150)

    # Find most/least local models
    most_local = max(model_keys, key=lambda mk: models_data[mk]['k=7 >80%'])
    least_local = min(model_keys, key=lambda mk: models_data[mk]['k=7 >80%'])

    p(f'\n  1. Attention Locality 排序:')
    p(f'     最局部: {all_results[most_local][0]["name"]} (k=7 >80% = {models_data[most_local]["k=7 >80%"]:.4f})')
    p(f'     最全局: {all_results[least_local][0]["name"]} (k=7 >80% = {models_data[least_local]["k=7 >80%"]:.4f})')

    p(f'\n  2. Reverse Locality (深层>浅层):')
    for mk in model_keys:
        config = all_results[mk][0]
        rl = models_data[mk]['reverse_locality']
        sr = models_data[mk]['shallow >80%@k=7']
        dr = models_data[mk]['deep >80%@k=7']
        p(f'     {config["name"]:<16} {"✅ reverse" if rl else "❌ normal"}: 浅={sr:.4f} 深={dr:.4f}')

    p(f'\n  3. 时间步依赖 (晚期>早期):')
    for mk in model_keys:
        config = all_results[mk][0]
        er = models_data[mk]['early >80%@k=7']
        lr = models_data[mk]['late >80%@k=7']
        delta = lr - er
        p(f'     {config["name"]:<16} early={er:.4f} late={lr:.4f} Δ={delta:+.4f} {"⚠ 有趋势" if abs(delta) > 0.02 else "✓ 稳定"}')

    p(f'\n  4. 图像内容依赖:')
    for mk in model_keys:
        config = all_results[mk][0]
        sp = models_data[mk]['img_spread']
        p(f'     {config["name"]:<16} spread={sp:.4f} {"⚠ 有依赖" if sp > 0.02 else "✓ 基本独立"}')

    p(f'\n  5. Head 多样性 (层内head间std):')
    for mk in model_keys:
        config = all_results[mk][0]
        nf = models_data[mk]['na_friendly_heads_pct']
        p(f'     {config["name"]:<16} NA-friendly heads: {nf:.1f}%')

    p(f'\n  6. Per-Head NA 理论 FLOPs 节省 (50% query覆盖):')
    for mk in model_keys:
        config = all_results[mk][0]
        fs = models_data[mk]['na_flops_saved_50pct']
        avg_k = models_data[mk]['na_avg_k_50pct']
        p(f'     {config["name"]:<16} avg_k={avg_k:.1f}, FLOPs节省={fs:.1f}%')

    # Check: FM vs DDPM
    fm_models = [mk for mk in model_keys if 'DDPM' not in all_results[mk][0]['framework']]
    ddpm_models = [mk for mk in model_keys if 'DDPM' in all_results[mk][0]['framework']]
    if fm_models and ddpm_models:
        fm_avg = np.mean([models_data[mk]['k=7 >80%'] for mk in fm_models])
        ddpm_avg = np.mean([models_data[mk]['k=7 >80%'] for mk in ddpm_models])
        p(f'\n  7. FM vs DDPM:')
        p(f'     FM 模型平均 k=7 >80%: {fm_avg:.4f}')
        p(f'     DDPM 模型平均 k=7 >80%: {ddpm_avg:.4f}')
        p(f'     {"DDPM更局部" if ddpm_avg > fm_avg else "FM更局部"} (Δ={ddpm_avg-fm_avg:+.4f})')

    # Check: Scale effect in SiT family
    sit_models = [mk for mk in model_keys if all_results[mk][0]['family'] == 'SiT']
    if len(sit_models) >= 2:
        p(f'\n  8. SiT 规模效应 (B→L→XL):')
        for mk in sit_models:
            config = all_results[mk][0]
            p(f'     {config["name"]:<16} layers={config["layers"]}, k=7 >80%={models_data[mk]["k=7 >80%"]:.4f}')

    p()
    p('█' * 130)
    p('█  跨模型综合对比完成')
    p('█' * 130)


def main():
    parser = argparse.ArgumentParser(description='统一 Attention Locality 分析')
    parser.add_argument('--model', type=str, default=None,
                        help=f'只分析指定模型: {", ".join(MODEL_CONFIGS.keys())}')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='报告输出目录 (默认: outputs/analysis/)')
    args = parser.parse_args()

    # Determine output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(project_dir, 'outputs', 'analysis')
    os.makedirs(output_dir, exist_ok=True)

    # Determine which models to analyze
    if args.model:
        if args.model not in MODEL_CONFIGS:
            print(f"Unknown model: {args.model}")
            print(f"Available: {list(MODEL_CONFIGS.keys())}")
            sys.exit(1)
        model_keys = [args.model]
    else:
        model_keys = list(MODEL_CONFIGS.keys())

    # Analyze each model
    all_results = {}
    for model_key in model_keys:
        config = MODEL_CONFIGS[model_key]
        print(f'\n{"="*130}')
        print(f'  Loading {config["name"]} ...')
        print(f'{"="*130}')

        data = load_data(config)
        if data is None:
            print(f'  SKIP {config["name"]}: no data found')
            continue

        masses, min_k_80, min_k_50, min_k_90, min_k_95, min_k_99, t_sched = data

        # Verify shape
        S, B, L, H, N, K = masses.shape
        expected_L = config['layers']
        expected_H = config['heads']
        if L != expected_L or H != expected_H:
            print(f'  WARNING: Expected [{expected_L} layers, {expected_H} heads], got [{L}, {H}]')
            # Update config
            config = dict(config)
            config['layers'] = L
            config['heads'] = H

        lines = analyze_model(model_key, config, masses, min_k_80, min_k_50, min_k_90, min_k_95, min_k_99, t_sched)

        # Save individual report
        report_path = os.path.join(output_dir, f'report_{model_key}.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(lines))
        print(f'\n  Report saved → {report_path}')

        all_results[model_key] = (config, masses, min_k_80, min_k_50, min_k_90, min_k_95, min_k_99, t_sched)

    # Cross-model comparison (if multiple models)
    if len(all_results) >= 2:
        # Redirect stdout to capture cross-model report
        import io
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        cross_model_comparison(all_results)
        sys.stdout = old_stdout
        cross_report = buffer.getvalue()
        print(cross_report)

        cross_path = os.path.join(output_dir, 'report_cross_model.txt')
        with open(cross_path, 'w') as f:
            f.write(cross_report)
        print(f'  Cross-model report saved → {cross_path}')

    print(f'\n所有报告已保存到: {output_dir}/')


if __name__ == '__main__':
    main()
