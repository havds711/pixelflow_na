#!/usr/bin/env python3
"""详细分析 attention_locality.pt 数据"""
import numpy as np, zipfile, json

with zipfile.ZipFile('outputs/attention_locality/attention_locality.pt', 'r') as z:
    masses = np.frombuffer(z.read('attention_locality/data/0'), dtype=np.float16).reshape(20,4,28,16,256,8)
    min_k_80 = np.frombuffer(z.read('attention_locality/data/1'), dtype=np.uint8).reshape(20,4,28,16,256)
    min_k_50 = np.frombuffer(z.read('attention_locality/data/2'), dtype=np.uint8).reshape(20,4,28,16,256)
    min_k_90 = np.frombuffer(z.read('attention_locality/data/3'), dtype=np.uint8).reshape(20,4,28,16,256)
    min_k_95 = np.frombuffer(z.read('attention_locality/data/4'), dtype=np.uint8).reshape(20,4,28,16,256)
    min_k_99 = np.frombuffer(z.read('attention_locality/data/5'), dtype=np.uint8).reshape(20,4,28,16,256)
    t_sched = np.frombuffer(z.read('attention_locality/data/6'), dtype=np.float32)

k_values = [1,3,5,7,9,11,13,15]
S, B, L, H, N, K = masses.shape
gs = 16
total_pairs_per_layer = S * B * H * N

# ── 空间区域 ──
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
print('  SiT-XL/2 ODE 推理 Attention Locality 完整分析')
print('='*130)
print(f'  数据维度: {S} ODE steps × {B} images × {L} layers × {H} heads × {N} queries × {K} k-values')
print(f'  k values: {k_values}')
print(f'  t_schedule (t=1→0): {t_sched}')
print(f'  类别标签: 207(golden_retriever), 360(otter), 387(lesser_panda), 974(geyser)')
print(f'  每层测量对: {total_pairs_per_layer:,} (20 steps × 4 imgs × 16 heads × 256 queries)')
print()

# ═══════════════════════════════════════════════════════════
# 表1: 全局 Summary
# ═══════════════════════════════════════════════════════════
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

# ═══════════════════════════════════════════════════════════
# 表2: 每层 k-matrix (>80%)
# ═══════════════════════════════════════════════════════════
print()
print('='*130)
print('表2 — 每层 × 每个k: >80% attention mass 的达标比例')
print('(avg over 20 steps × 4 imgs × 16 heads × 256 queries = 327,680 点/层)')
print('='*130)
hdr2 = f'{"L":<4}' + ''.join(f'{"k="+str(k):>9}' for k in k_values) + '  {"推荐k(>80%)":>12}  {"层均值":>8}  {"深/浅":>6}'
print(hdr2)
print('-'*len(hdr2))
for l in range(L):
    row = f'{l:<4}'
    best_k = None
    ratios = []
    for ki, k in enumerate(k_values):
        r = (masses[:,:,l,:,:,ki].astype(float) > 0.80).mean()
        ratios.append(r)
        row += f'{r:>9.4f}'
        if best_k is None and r >= 0.8:
            best_k = k
    best_str = str(best_k) if best_k else '>15'
    depth = '浅' if l < 9 else ('中' if l < 19 else '深')
    row += f'  {best_str:>12}  {np.mean(ratios):>8.4f}  {depth:>6}'
    print(row)

# ═══════════════════════════════════════════════════════════
# 表3: t 依赖性
# ═══════════════════════════════════════════════════════════
print()
print('='*130)
print('表3 — t 依赖性: ODE 早/中/晚期 k=7 >80% 达标比例')
print('='*130)
stages = [
    ('early  t=[1.0→0.75] steps 0-5',  slice(0,6)),
    ('mid    t=[0.70→0.35] steps 6-13', slice(6,14)),
    ('late   t=[0.30→0.00] steps 14-19', slice(14,20)),
]
ki7 = 3  # k=7
hdr3 = f'{"L":<4}' + ''.join(f'{name:>34}' for name,_ in stages) + '  {"early→late":>12}'
print(hdr3)
print('-'*len(hdr3))
for l in range(L):
    row = f'{l:<4}'
    vals = []
    for name, sl in stages:
        r = (masses[sl,:,l,:,:,ki7].astype(float) > 0.80).mean()
        vals.append(r)
        row += f'{r:>34.4f}'
    delta = vals[2] - vals[0]
    row += f'  {delta:>+12.4f}'
    print(row)
# 全局t平均
print()
for name, sl in stages:
    r = (masses[sl,:,:,:,:,ki7].astype(float) > 0.80).mean()
    print(f'  全局 {name}: {r:.4f}')

# ═══════════════════════════════════════════════════════════
# 表4: Head 差异
# ═══════════════════════════════════════════════════════════
print()
print('='*130)
print('表4 — Head 差异: 每层16个head的 min_k_80 统计 (只计已达标query, min_k=0表示k=15仍<80%)')
print('='*130)
hdr4 = f'{"L":<4} {"层均值":>8} {"std":>8} {"min":>6} {"max":>6}  | 各 head min_k_80 均值'
print(hdr4)
print('-'*130)
for l in range(L):
    layer_data = min_k_80[:,:,l,:,:].astype(float).flatten()
    layer_nz = layer_data[layer_data > 0]
    if len(layer_nz) == 0:
        print(f'{l:<4} {"N/A"}')
        continue
    head_means = []
    for h in range(H):
        hd = min_k_80[:,:,l,h,:].astype(float).flatten()
        hd = hd[hd > 0]
        head_means.append(hd.mean() if len(hd) > 0 else 0)
    hdr = ' '.join(f'{hm:>5.0f}' for hm in head_means)
    print(f'{l:<4} {layer_nz.mean():>8.1f} {layer_nz.std():>8.1f} '
          f'{layer_nz.min():>6.0f} {layer_nz.max():>6.0f}  | {hdr}')

# ═══════════════════════════════════════════════════════════
# 表5: 空间位置
# ═══════════════════════════════════════════════════════════
print()
print('='*130)
print('表5 — 空间位置差异: corner/edge/interior min_k_80 均值')
print(f'      query分布: corner(d<=1)={ri_counts[0]}, edge(d2-3)={ri_counts[1]}, interior(d>=4)={ri_counts[2]}')
print('='*130)
region_names = ['corner', 'edge', 'interior']
hdr5 = f'{"L":<4}' + ''.join(f'{n:>14}' for n in region_names) + '  {"int-corner Δ":>14}  {"corner/int":>12}'
print(hdr5)
print('-'*len(hdr5))
for l in range(L):
    row = f'{l:<4}'
    vals = []
    for ri in range(3):
        q_mask = (region_flat == ri)
        rd = min_k_80[:,:,l,:,q_mask].astype(float).flatten()
        rd = rd[rd > 0]
        v = rd.mean() if len(rd) > 0 else 0
        vals.append(v)
        row += f'{v:>14.1f}'
    delta = vals[2] - vals[0]
    ratio = vals[0] / vals[2] if vals[2] > 0 else 0
    row += f'  {delta:>+14.1f}  {ratio:>12.2f}'
    print(row)

# ═══════════════════════════════════════════════════════════
# 表6: min_k 分布直方图
# ═══════════════════════════════════════════════════════════
print()
print('='*130)
print('表6 — min_k_80 分布直方图: 各 k 值占比 (%)')
print('='*130)
hdr6 = f'{"L":<4}' + ''.join(f'{"k="+str(k):>8}' for k in k_values) + '  {">15(未达标)":>12}'
print(hdr6)
print('-'*len(hdr6))
for l in range(L):
    row = f'{l:<4}'
    layer_mk = min_k_80[:,:,l,:,:].flatten()
    total = len(layer_mk)
    for k in k_values:
        pct = (layer_mk == k).sum() / total * 100
        row += f'{pct:>8.1f}'
    pct_fail = (layer_mk == 0).sum() / total * 100
    row += f'  {pct_fail:>12.1f}'
    print(row)

# ═══════════════════════════════════════════════════════════
# 表7: 五个阈值对比
# ═══════════════════════════════════════════════════════════
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

# ═══════════════════════════════════════════════════════════
# 表8: 深层/中层/浅层 对比
# ═══════════════════════════════════════════════════════════
print()
print('='*130)
print('表8 — 深度趋势: 浅层(0-8) / 中层(9-18) / 深层(19-27) 对比')
print('='*130)
depth_groups = [
    ('浅层 L0-L8  (9层)',  slice(0,9)),
    ('中层 L9-L18 (10层)', slice(9,19)),
    ('深层 L19-L27 (9层)', slice(19,28)),
]
for name, sl in depth_groups:
    d = masses[sl,:,:,:,:].astype(float)
    d_mk80 = min_k_80[sl,:,:,:,:].astype(float)
    d_nz = d_mk80.flatten(); d_nz = d_nz[d_nz > 0]
    print(f'\n{name}:')
    print(f'  min_k_80: mean={d_nz.mean():.1f}, std={d_nz.std():.1f}, median={np.median(d_nz):.0f}')
    for ki, k in enumerate(k_values):
        r50 = (d[:,:,:,:,:,ki] > 0.50).mean()
        r80 = (d[:,:,:,:,:,ki] > 0.80).mean()
        r95 = (d[:,:,:,:,:,ki] > 0.95).mean()
        r99 = (d[:,:,:,:,:,ki] > 0.99).mean()
        print(f'  k={k:<4}  >50%={r50:.4f}  >80%={r80:.4f}  >95%={r95:.4f}  >99%={r99:.4f}')

# ═══════════════════════════════════════════════════════════
# 表9: 图像间差异
# ═══════════════════════════════════════════════════════════
print()
print('='*130)
print('表9 — 图像间变异性: 4张不同类别图像 >80%@k=7 比率')
print('类别: img0=207(golden_retriever), img1=360(otter), img2=387(lesser_panda), img3=974(geyser)')
print('='*130)
hdr9 = f'{"img":<6} {"类别":<20}' + ''.join(f'{"L="+str(l):>9}' for l in [0,4,9,14,19,23,27]) + '  {"avg28L":>9}'
print(hdr9)
print('-'*len(hdr9))
class_names = {0:'207-golden_retriever', 1:'360-otter', 2:'387-lesser_panda', 3:'974-geyser'}
for b in range(B):
    vals = []
    for l in [0,4,9,14,19,23,27]:
        r = (masses[:,b,l,:,:,ki7].astype(float) > 0.80).mean()
        vals.append(r)
    r_all = (masses[:,b,:,:,:,ki7].astype(float) > 0.80).mean()
    row = f'{b:<6} {class_names[b]:<20}'
    for v in vals:
        row += f'{v:>9.4f}'
    row += f'  {r_all:>9.4f}'
    print(row)
# 图像间std
print()
for l in [0,4,9,14,19,23,27]:
    vals = [(masses[:,b,l,:,:,ki7].astype(float) > 0.80).mean() for b in range(B)]
    print(f'  L{l}: 4图 max-min spread = {max(vals)-min(vals):.4f}, std = {np.std(vals):.4f}')

# ═══════════════════════════════════════════════════════════
# 表10: ODE 步间趋势 - 每一步的全局 >80%@k=7
# ═══════════════════════════════════════════════════════════
print()
print('='*130)
print('表10 — ODE 步间趋势: 每一步全局 >80%@k=7')
print('='*130)
print(f'{"step":<6} {"t":>8}', end='')
for l in [0,4,9,14,19,23,27]:
    print(f'{"L="+str(l):>10}', end='')
print('  {"avg28L":>10}')
print('-'*80)
for s in range(S):
    print(f'{s:<6} {t_sched[s]:>8.2f}', end='')
    vals_s = []
    for l in [0,4,9,14,19,23,27]:
        r = (masses[s,:,l,:,:,ki7].astype(float) > 0.80).mean()
        vals_s.append(r)
        print(f'{r:>10.4f}', end='')
    r_all = (masses[s,:,:,:,:,ki7].astype(float) > 0.80).mean()
    vals_s.append(r_all)
    print(f'  {r_all:>10.4f}')

# ═══════════════════════════════════════════════════════════
# 表11: 关键发现汇总
# ═══════════════════════════════════════════════════════════
print()
print('='*130)
print('表11 — 关键统计发现汇总')
print('='*130)

# 最佳k
best_k_for_80 = None
for ki, k in enumerate(k_values):
    if (masses.astype(float)[:,:,:,:,:,ki] > 0.80).mean() >= 0.8:
        best_k_for_80 = k
        break

print(f'\n  [Locality强度]')
# k=1 能否捕捉
k1_80 = (masses.astype(float)[:,:,:,:,:,0] > 0.80).mean()
print(f'    k=1(单点) >80% 达标率: {k1_80:.4f} ({k1_80*100:.1f}%)')
k3_80 = (masses.astype(float)[:,:,:,:,:,1] > 0.80).mean()
print(f'    k=3  >80% 达标率: {k3_80:.4f} ({k3_80*100:.1f}%)')
k5_80 = (masses.astype(float)[:,:,:,:,:,2] > 0.80).mean()
print(f'    k=5  >80% 达标率: {k5_80:.4f} ({k5_80*100:.1f}%)')
k7_80 = (masses.astype(float)[:,:,:,:,:,3] > 0.80).mean()
print(f'    k=7  >80% 达标率: {k7_80:.4f} ({k7_80*100:.1f}%)')
print(f'    全局推荐 k≥{best_k_for_80} 可达 80%+ attention mass')

print(f'\n  [层级差异]')
shallow = (masses.astype(float)[:,:,:9,:,:,3] > 0.80).mean()
deep = (masses.astype(float)[:,:,19:,:,:,3] > 0.80).mean()
print(f'    浅层 L0-8  >80%@k=7: {shallow:.4f}')
print(f'    深层 L19-27 >80%@k=7: {deep:.4f}')
print(f'    浅/深比: {shallow/deep:.2f}x')

print(f'\n  [t依赖]')
early = (masses.astype(float)[:6,:,:,:,:,3] > 0.80).mean()
late = (masses.astype(float)[14:,:,:,:,:,3] > 0.80).mean()
print(f'    早期 t=1.0→0.75 >80%@k=7: {early:.4f}')
print(f'    晚期 t=0.3→0.0  >80%@k=7: {late:.4f}')
print(f'    晚期/早期比: {late/early:.2f}x')

print(f'\n  [空间差异]')
corner = min_k_80[:,:,:,:,region_flat==0].astype(float).flatten()
edge = min_k_80[:,:,:,:,region_flat==1].astype(float).flatten()
interior = min_k_80[:,:,:,:,region_flat==2].astype(float).flatten()
corner_nz = corner[corner > 0]
edge_nz = edge[edge > 0]
interior_nz = interior[interior > 0]
print(f'    corner mean min_k_80: {corner_nz.mean():.1f}')
print(f'    edge mean min_k_80: {edge_nz.mean():.1f}')
print(f'    interior mean min_k_80: {interior_nz.mean():.1f}')

print(f'\n  [Head一致性]')
head_means_per_layer = []
for l in range(L):
    hm = []
    for h in range(H):
        hd = min_k_80[:,:,l,h,:].astype(float).flatten()
        hd = hd[hd > 0]
        hm.append(hd.mean() if len(hd) > 0 else 0)
    head_means_per_layer.append(np.std(hm))
print(f'    每层内16 head的mean min_k std: mean={np.mean(head_means_per_layer):.2f}, '
      f'min={np.min(head_means_per_layer):.2f}, max={np.max(head_means_per_layer):.2f}')

print(f'\n  [ODEstep稳定性]')
step_ratios = [(masses[s,:,:,:,:,3].astype(float) > 0.80).mean() for s in range(S)]
print(f'    20步 >80%@k=7 ratio: mean={np.mean(step_ratios):.4f}, std={np.std(step_ratios):.4f}, '
      f'min={np.min(step_ratios):.4f}, max={np.max(step_ratios):.4f}')

print(f'\n  [图像间一致性]')
img_ratios = [(masses[:,b,:,:,:,3].astype(float) > 0.80).mean() for b in range(B)]
print(f'    4图 >80%@k=7 ratio: mean={np.mean(img_ratios):.4f}, std={np.std(img_ratios):.4f}, '
      f'spread={max(img_ratios)-min(img_ratios):.4f}')

print()
print('='*130)
print('  分析完成')
print('='*130)
