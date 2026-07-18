#!/usr/bin/env python3
"""PixelDiT Attention Locality 完整分析"""
import numpy as np, zipfile

with zipfile.ZipFile('../outputs/attention_locality_pixeldit/attention_locality_pixeldit.pt', 'r') as z:
    PREFIX = 'attention_locality_pixeldit'
    masses = np.frombuffer(z.read(f'{PREFIX}/data/0'), dtype=np.float16).reshape(25,4,30,16,256,8)
    min_k_80 = np.frombuffer(z.read(f'{PREFIX}/data/1'), dtype=np.uint8).reshape(25,4,30,16,256)
    min_k_50 = np.frombuffer(z.read(f'{PREFIX}/data/2'), dtype=np.uint8).reshape(25,4,30,16,256)
    min_k_90 = np.frombuffer(z.read(f'{PREFIX}/data/3'), dtype=np.uint8).reshape(25,4,30,16,256)
    min_k_95 = np.frombuffer(z.read(f'{PREFIX}/data/4'), dtype=np.uint8).reshape(25,4,30,16,256)
    min_k_99 = np.frombuffer(z.read(f'{PREFIX}/data/5'), dtype=np.uint8).reshape(25,4,30,16,256)
    t_sched = np.frombuffer(z.read(f'{PREFIX}/data/6'), dtype=np.float32)

k_values = [1,3,5,7,9,11,13,15]
S, B, L, H, N, K = masses.shape
gs = 16
total_pairs_per_layer = S * B * H * N

region = np.zeros((gs, gs), dtype=int)
for i in range(gs):
    for j in range(gs):
        d = min(i, j, gs-1-i, gs-1-j)
        if d <= 1: region[i,j] = 0
        elif d <= 3: region[i,j] = 1
        else: region[i,j] = 2
region_flat = region.reshape(-1)
ri_counts = np.bincount(region_flat, minlength=3)

print('='*130)
print('  PixDiT c2i — Flow Matching 推理 Attention Locality 完整分析')
print('='*130)
print(f'  数据维度: {S} ODE steps x {B} images x {L} layers (26 patch + 4 pixel) x {H} heads x {N} queries x {K} k-values')
print(f'  k values: {k_values}')
print(f'  t_schedule (t=0→1): {t_sched}')
print(f'  类别: 207(golden_retriever), 360(otter), 387(lesser_panda), 974(geyser)')
print(f'  每层测量对: {total_pairs_per_layer:,}')
print()

# ═══════════════════════════════════════════ 表1: 全局 ═══════════════════════════════════════════
print('='*130)
print('表1 — 全局 Summary: k×k 窗口内累积 attention mass 达标比例')
print('='*130)
hdr = f'{"k":<8} {"累计mass均值":>14} {">50%":>10} {">80%":>10} {">90%":>10} {">95%":>10} {">99%":>10}  {"推荐":>4}'
print(hdr)
print('-'*len(hdr))
for ki, k in enumerate(k_values):
    d = masses[:,:,:,:,:,ki].astype(float)
    avg_mass = d.mean()
    r50 = (d > 0.50).mean()
    r80 = (d > 0.80).mean()
    r90 = (d > 0.90).mean()
    r95 = (d > 0.95).mean()
    r99 = (d > 0.99).mean()
    flag = ' ★' if r80 >= 0.8 else ''
    print(f'k={k:<5} {avg_mass:>14.4f} {r50:>10.4f} {r80:>10.4f} {r90:>10.4f} {r95:>10.4f} {r99:>10.4f}  {flag}')

# ═══════════════════════════════════════════ 表2: 每层 >80% ═══════════════════════════════════════════
print()
print('='*130)
print('表2 — 每层 × 每个k: >80% attention mass 达标比例')
print('(avg over 25 steps x 4 imgs x 16 heads x 256 queries = 409,600 点/层)')
print('='*130)
hdr2 = f'{"L":<4} {"type":<6}' + ''.join(f'{"k="+str(k):>9}' for k in k_values) + '  {"推荐k":>8}  {"层均值":>8}'
print(hdr2)
print('-'*len(hdr2))
for l in range(L):
    ltype = 'patch' if l < 26 else 'pixel'
    row = f'{l:<4} {ltype:<6}'
    best_k = None
    ratios = []
    for ki, k in enumerate(k_values):
        r = (masses[:,:,l,:,:,ki].astype(float) > 0.80).mean()
        ratios.append(r)
        row += f'{r:>9.4f}'
        if best_k is None and r >= 0.8:
            best_k = k
    best_str = str(best_k) if best_k else '>15'
    row += f'  {best_str:>8}  {np.mean(ratios):>8.4f}'
    print(row)

# ═══════════════════════════════════════════ 表3: patch vs pixel 对比 ═══════════════════════════════════════════
print()
print('='*130)
print('表3 — Patch Blocks vs Pixel Blocks 对比')
print('='*130)
for name, sl in [('Patch blocks (L0-L25, 26层)', slice(0,26)), ('Pixel blocks (L26-L29, 4层)', slice(26,30))]:
    d = masses[:,:,sl,:,:,:].astype(float)
    d_mk80 = min_k_80[:,:,sl,:,:].astype(float)
    d_nz = d_mk80.flatten(); d_nz = d_nz[d_nz > 0]
    print(f'\n{name}:')
    print(f'  min_k_80: mean={d_nz.mean():.1f}, std={d_nz.std():.1f}, median={np.median(d_nz):.0f}')
    for ki, k in enumerate(k_values):
        r50 = (d[:,:,:,:,:,ki] > 0.50).mean()
        r80 = (d[:,:,:,:,:,ki] > 0.80).mean()
        r95 = (d[:,:,:,:,:,ki] > 0.95).mean()
        r99 = (d[:,:,:,:,:,ki] > 0.99).mean()
        print(f'  k={k:<4}  >50%={r50:.4f}  >80%={r80:.4f}  >95%={r95:.4f}  >99%={r99:.4f}')

# ═══════════════════════════════════════════ 表4: Head 差异 ═══════════════════════════════════════════
print()
print('='*130)
print('表4 — Head 差异: 每层16个head的 min_k_80 统计')
print('='*130)
hdr4 = f'{"L":<4} {"type":<6} {"层均值":>8} {"std":>8} {"min":>6} {"max":>6}  | 各 head min_k_80 均值'
print(hdr4)
print('-'*130)
for l in range(L):
    ltype = 'patch' if l < 26 else 'pixel'
    layer_data = min_k_80[:,:,l,:,:].astype(float).flatten()
    layer_nz = layer_data[layer_data > 0]
    if len(layer_nz) == 0:
        print(f'{l:<4} {ltype:<6} N/A')
        continue
    head_means = []
    for h in range(H):
        hd = min_k_80[:,:,l,h,:].astype(float).flatten()
        hd = hd[hd > 0]
        head_means.append(hd.mean() if len(hd) > 0 else 0)
    hdr = ' '.join(f'{hm:>5.0f}' for hm in head_means)
    print(f'{l:<4} {ltype:<6} {layer_nz.mean():>8.1f} {layer_nz.std():>8.1f} '
          f'{layer_nz.min():>6.0f} {layer_nz.max():>6.0f}  | {hdr}')

# ═══════════════════════════════════════════ 表5: 空间位置 ═══════════════════════════════════════════
print()
print('='*130)
print('表5 — 空间位置差异: corner/edge/interior min_k_80 均值')
print(f'      query分布: corner(d<=1)={ri_counts[0]}, edge(d2-3)={ri_counts[1]}, interior(d>=4)={ri_counts[2]}')
print('='*130)
region_names = ['corner', 'edge', 'interior']
hdr5 = f'{"L":<4} {"type":<6}' + ''.join(f'{n:>14}' for n in region_names) + '  {"int-corner Δ":>14}'
print(hdr5)
print('-'*len(hdr5))
for l in range(L):
    ltype = 'patch' if l < 26 else 'pixel'
    row = f'{l:<4} {ltype:<6}'
    vals = []
    for ri in range(3):
        q_mask = (region_flat == ri)
        rd = min_k_80[:,:,l,:,q_mask].astype(float).flatten()
        rd = rd[rd > 0]
        v = rd.mean() if len(rd) > 0 else 0
        vals.append(v)
        row += f'{v:>14.1f}'
    delta = vals[2] - vals[0]
    row += f'  {delta:>+14.1f}'
    print(row)

# ═══════════════════════════════════════════ 表6: min_k 分布 ═══════════════════════════════════════════
print()
print('='*130)
print('表6 — min_k_80 分布直方图: 各 k 值占比 (%)')
print('='*130)
hdr6 = f'{"L":<4} {"type":<6}' + ''.join(f'{"k="+str(k):>8}' for k in k_values) + '  {"未达标":>8}'
print(hdr6)
print('-'*len(hdr6))
for l in range(L):
    ltype = 'patch' if l < 26 else 'pixel'
    row = f'{l:<4} {ltype:<6}'
    layer_mk = min_k_80[:,:,l,:,:].flatten()
    total = len(layer_mk)
    for k in k_values:
        pct = (layer_mk == k).sum() / total * 100
        row += f'{pct:>8.1f}'
    pct_fail = (layer_mk == 0).sum() / total * 100
    row += f'  {pct_fail:>8.1f}'
    print(row)

# ═══════════════════════════════════════════ 表7: 五个阈值 ═══════════════════════════════════════════
print()
print('='*130)
print('表7 — 五个 min_k 阈值统计对比 (50%/80%/90%/95%/99%)')
print('='*130)
hdr7 = f'{"阈值":<10} {"全局mean":>10} {"std":>8} {"median":>8} {"Q25":>6} {"Q75":>6} {"达标率":>8}'
print(hdr7)
print('-'*len(hdr7))
for label, data in [('>50%', min_k_50), ('>80%', min_k_80), ('>90%', min_k_90),
                     ('>95%', min_k_95), ('>99%', min_k_99)]:
    d = data.flatten().astype(float)
    d_nz = d[d > 0]
    coverage = len(d_nz) / len(d) * 100
    print(f'{label:<10} {d_nz.mean():>10.1f} {d_nz.std():>8.1f} {np.median(d_nz):>8.0f} '
          f'{np.percentile(d_nz,25):>6.0f} {np.percentile(d_nz,75):>6.0f} {coverage:>7.1f}%')

# ═══════════════════════════════════════════ 表8: 深度趋势 ═══════════════════════════════════════════
print()
print('='*130)
print('表8 — Patch blocks 深度趋势: 浅/中/深 对比 (仅 patch blocks L0-L25)')
print('='*130)
depth_groups = [
    ('浅层 L0-L8  (9层)',  slice(0,9)),
    ('中层 L9-L17 (9层)', slice(9,18)),
    ('深层 L18-L25 (8层)', slice(18,26)),
]
for name, sl in depth_groups:
    d = masses[:,:,sl,:,:,:].astype(float)
    d_mk80 = min_k_80[:,:,sl,:,:].astype(float)
    d_nz = d_mk80.flatten(); d_nz = d_nz[d_nz > 0]
    print(f'\n{name}:')
    print(f'  min_k_80: mean={d_nz.mean():.1f}, std={d_nz.std():.1f}, median={np.median(d_nz):.0f}')
    for ki, k in enumerate(k_values):
        r50 = (d[:,:,:,:,:,ki] > 0.50).mean()
        r80 = (d[:,:,:,:,:,ki] > 0.80).mean()
        r95 = (d[:,:,:,:,:,ki] > 0.95).mean()
        r99 = (d[:,:,:,:,:,ki] > 0.99).mean()
        print(f'  k={k:<4}  >50%={r50:.4f}  >80%={r80:.4f}  >95%={r95:.4f}  >99%={r99:.4f}')

# ═══════════════════════════════════════════ 表9: ODE 步间 ═══════════════════════════════════════════
print()
print('='*130)
print('表9 — ODE 步间趋势: 每步全局 >80%@k=7')
print('='*130)
ki7 = 3
print(f'{"step":<6} {"t":>8}', end='')
for l in [0,6,12,18,24,26,28,29]:
    print(f'{"L="+str(l):>10}', end='')
print('  {"avg30L":>10}')
print('-'*90)
for s in range(S):
    print(f'{s:<6} {t_sched[s]:>8.3f}', end='')
    for l in [0,6,12,18,24,26,28,29]:
        r = (masses[s,:,l,:,:,ki7].astype(float) > 0.80).mean()
        print(f'{r:>10.4f}', end='')
    r_all = (masses[s,:,:,:,:,ki7].astype(float) > 0.80).mean()
    print(f'  {r_all:>10.4f}')

# ═══════════════════════════════════════════ 表10: 图像间 ═══════════════════════════════════════════
print()
print('='*130)
print('表10 — 图像间变异性: >80%@k=7')
print('='*130)
for b in range(B):
    r_all = (masses[:,b,:,:,:,ki7].astype(float) > 0.80).mean()
    print(f'  img{b}: {r_all:.4f}')

# ═══════════════════════════════════════════ 表11: Per-head NA ═══════════════════════════════════════════
print()
print('='*130)
print('表11 — Per-Head NA 可行性: 每个 head 满足不同覆盖率的 min_k')
print('='*130)
r80_per_head = (masses.astype(float) > 0.80).mean(axis=(0,1,4))  # [L, H, K]
for target_pct in [0.5, 0.8]:
    head_optimal_k = np.zeros((L, H), dtype=int)
    for l in range(L):
        for h in range(H):
            found = False
            for ki, k in enumerate(k_values):
                if r80_per_head[l, h, ki] >= target_pct:
                    head_optimal_k[l, h] = k
                    found = True
                    break
            if not found:
                head_optimal_k[l, h] = 99
    total_heads = L * H
    print(f'\nTarget: {target_pct*100:.0f}% queries per head reach >80% mass:')
    for k in k_values:
        n = (head_optimal_k == k).sum()
        print(f'  k={k:<4}: {n:>3} heads ({n/total_heads*100:>5.1f}%)')
    n = (head_optimal_k == 99).sum()
    print(f'  k>15  : {n:>3} heads ({n/total_heads*100:>5.1f}%)')
    w_avg = sum(k * (head_optimal_k == k).sum() for k in k_values) + 15 * n
    w_avg /= total_heads
    print(f'  Weighted avg k: {w_avg:.1f}')

# ═══════════════════════════════════════════ 表12: NA-friendly heads ═══════════════════════════════════════════
print()
print('='*130)
print('表12 — NA-Friendly Heads (mean min_k_80 <= 7)')
print('='*130)
na_friendly = []
for l in range(L):
    ltype = 'patch' if l < 26 else 'pixel'
    for h in range(H):
        hd = min_k_80[:,:,l,h,:].astype(float).flatten()
        hd = hd[hd > 0]
        mean_k = hd.mean() if len(hd) > 0 else 99
        if mean_k <= 7:
            na_friendly.append((l, ltype, h, mean_k))
            print(f'  L{l} ({ltype}) H{h}: mean_min_k={mean_k:.1f}')
print(f'  Total: {len(na_friendly)} / {L*H} heads ({len(na_friendly)/(L*H)*100:.1f}%)')

# ═══════════════════════════════════════════ 表13: SiT vs PixDiT 对比 ═══════════════════════════════════════════
print()
print('='*130)
print('表13 — SiT vs PixDiT 关键指标对比')
print('='*130)
# SiT numbers from previous analysis
sit_stats = {
    'k=1 avg mass': 0.0175, 'k=1 >80%': 0.0015,
    'k=3 >80%': 0.0871, 'k=5 >80%': 0.1069, 'k=7 >80%': 0.1301,
    'k=9 >80%': 0.1604, 'k=11 >80%': 0.2061, 'k=13 >80%': 0.3453, 'k=15 >80%': 0.9372,
    'shallow >80%@k=7': 0.0357, 'deep >80%@k=7': 0.1794,
}

# PixDiT numbers - compute fresh from data
d_all = masses.astype(float)
pix_stats = {
    'k=1 avg mass': d_all[:,:,:,:,:,0].mean(),
    'k=1 >80%': (d_all[:,:,:,:,:,0] > 0.80).mean(),
    'k=3 >80%': (d_all[:,:,:,:,:,1] > 0.80).mean(),
    'k=5 >80%': (d_all[:,:,:,:,:,2] > 0.80).mean(),
    'k=7 >80%': (d_all[:,:,:,:,:,3] > 0.80).mean(),
    'k=9 >80%': (d_all[:,:,:,:,:,4] > 0.80).mean(),
    'k=11 >80%': (d_all[:,:,:,:,:,5] > 0.80).mean(),
    'k=13 >80%': (d_all[:,:,:,:,:,6] > 0.80).mean(),
    'k=15 >80%': (d_all[:,:,:,:,:,7] > 0.80).mean(),
    'shallow >80%@k=7': (d_all[:,:,:9,:,:,3] > 0.80).mean(),   # L0-8 patch
    'deep >80%@k=7': (d_all[:,:,18:26,:,:,3] > 0.80).mean(),    # L18-25 patch
}

print(f'{"Metric":<25} {"SiT-XL/2":>12} {"PixDiT":>12} {"Diff":>12}')
print('-'*65)
for key in ['k=1 avg mass', 'k=1 >80%', 'k=3 >80%', 'k=5 >80%', 'k=7 >80%',
            'k=9 >80%', 'k=11 >80%', 'k=13 >80%', 'k=15 >80%',
            'shallow >80%@k=7', 'deep >80%@k=7']:
    sv = sit_stats[key]
    pv = pix_stats.get(key, 0)
    diff = pv - sv
    print(f'{key:<25} {sv:>12.4f} {pv:>12.4f} {diff:>+12.4f}')

print()
print('='*130)
print('  分析完成')
print('='*130)
