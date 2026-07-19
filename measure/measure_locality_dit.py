#!/usr/bin/env python3
"""
DiT-XL/2 — DDPM 推理中 Attention Locality 全量测量

DiT 使用 DDPM 采样（预测 epsilon），t ∈ [0, 999] long integers。
Attention 是 timm 标准 Attention（情况 A: 手动 softmax），可直接 hook。

Usage:
  conda activate dit
  python measure/measure_locality_dit.py \
    --ckpt checkpoints/DiT-XL-2-256x256.pt \
    --num-steps 250 \
    --output outputs/attention_locality_dit_xl/attention_locality_dit_xl.pt
"""

import argparse, os, sys, math
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

import importlib.util
_dit_models_path = os.path.join(os.path.dirname(__file__), '..', 'dit_repo', 'models.py')
_dit_spec = importlib.util.spec_from_file_location("dit_models", _dit_models_path)
_dit_models = importlib.util.module_from_spec(_dit_spec)
_dit_spec.loader.exec_module(_dit_models)
DiT_XL_2 = _dit_models.DiT_XL_2

# DiT-XL/2 config
N_LAYERS = 28
N_HEADS = 16
GRID_SIZE = 16  # 32×32 latent, patch=2 → 16×16 grid
LATENT_SIZE = 32
IN_CHANNELS = 4


# ═════════════════════════════════════════════════════════════════════════════
# 1. Monkey-patch timm Attention: 抓取 post-softmax attention weights
# ═════════════════════════════════════════════════════════════════════════════
# DiT 使用 timm.models.vision_transformer.Attention，内部有手动 softmax
# 我们在每个 block 的 attn 上注册 hook 来抓 attention weights

_COLLECTED_ATTN = {}

def _make_attention_hook(layer_idx):
    """返回一个 hook，在 Attention forward 之后抓取 attention weights"""
    def hook(module, input, output):
        # 我们需要在 forward 中途抓，post-forward hook 拿不到中间 attn
        pass
    return hook


# 更好的方案：直接替换整个 forward 逻辑
# DiT 的 DiTBlock: x = x + gate * attn(modulate(norm1(x), shift, scale))
# 其中的 attn 是 timm.models.vision_transformer.Attention

_ORIG_ATTN_FORWARDS = {}

def _patched_attn_forward(self, x):
    """timm Attention forward + 抓取 attention weights"""
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    # 唯一改动：保存 attention weights
    _COLLECTED_ATTN[id(self)] = attn.detach()

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def patch_dit_attention(model):
    """替换所有 Attention 的 forward 为可抓取 weights 的版本"""
    _COLLECTED_ATTN.clear()
    _ORIG_ATTN_FORWARDS.clear()
    for block in model.blocks:
        attn = block.attn
        key = id(attn)
        _ORIG_ATTN_FORWARDS[key] = attn.forward
        attn.forward = _patched_attn_forward.__get__(attn, type(attn))
    print(f"  Patched {len(model.blocks)} attention modules")


def collect_attn_weights(model):
    """取出本次 forward 收集到的所有 attention weights（按层序）"""
    result = []
    for block in model.blocks:
        key = id(block.attn)
        result.append(_COLLECTED_ATTN.get(key, None))
    return result


# ═════════════════════════════════════════════════════════════════════════════
# 2. 预计算窗口索引（边界平移，完整 k×k）
# ═════════════════════════════════════════════════════════════════════════════

def precompute_window_indices(grid_size: int, k_values: list):
    H = W = grid_size
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

    gather_indices = {}
    for k in k_values:
        gather_indices[k] = indices[k].reshape(H * W, k * k)
    return gather_indices


# ═════════════════════════════════════════════════════════════════════════════
# 3. DDPM 推理 + 全量 attention 采集
# ═════════════════════════════════════════════════════════════════════════════

def get_ddpm_schedule(num_steps: int = 250, diffusion_steps: int = 1000):
    """
    构建 DDPM beta/alpha schedule (linear schedule)。
    返回预计算的 numpy 数组，用于采样。
    """
    # 与 create_diffusion 一致：linear schedule, 1000 steps
    scale = 1000 / diffusion_steps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    betas = np.linspace(beta_start, beta_end, diffusion_steps, dtype=np.float64)

    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

    # posterior variance
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    # coefficients for x_{t-1} = coef1 * pred_x0 + coef2 * x_t
    posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)

    # 降采样到 num_steps
    timesteps = np.linspace(diffusion_steps - 1, 0, num_steps).astype(int)

    return {
        'betas': betas,
        'alphas': alphas,
        'sqrt_alphas_cumprod': np.sqrt(alphas_cumprod),
        'sqrt_one_minus_alphas_cumprod': np.sqrt(1.0 - alphas_cumprod),
        'posterior_variance': posterior_variance,
        'posterior_mean_coef1': posterior_mean_coef1,
        'posterior_mean_coef2': posterior_mean_coef2,
        'timesteps': timesteps,
    }


@torch.no_grad()
def run_ddpm_and_collect(
    model,
    class_labels: list,
    num_steps: int = 250,
    k_values: list = None,
    device: str = "cuda",
    seed: int = 42,
):
    if k_values is None:
        k_values = [1, 3, 5, 7, 9, 11, 13, 15]

    torch.manual_seed(seed)
    model.eval()

    B = len(class_labels)
    labels = torch.tensor(class_labels, device=device)
    N = GRID_SIZE * GRID_SIZE  # 256
    n_layers = N_LAYERS
    n_heads = N_HEADS
    n_k = len(k_values)

    # 构建 DDPM schedule
    schedule = get_ddpm_schedule(num_steps)
    timesteps = schedule['timesteps']
    sqrt_alphas_cumprod = schedule['sqrt_alphas_cumprod']
    sqrt_one_minus_alphas_cumprod = schedule['sqrt_one_minus_alphas_cumprod']
    posterior_mean_coef1 = schedule['posterior_mean_coef1']
    posterior_mean_coef2 = schedule['posterior_mean_coef2']
    posterior_variance = schedule['posterior_variance']

    # 预计算窗口索引
    print("Precomputing window indices...")
    gather_idx = precompute_window_indices(GRID_SIZE, k_values)
    gather_idx_gpu = {k: v.to(device) for k, v in gather_idx.items()}

    # 存储
    masses = torch.zeros(num_steps, B, n_layers, n_heads, N, n_k, dtype=torch.float16)
    min_k_80 = torch.zeros(num_steps, B, n_layers, n_heads, N, dtype=torch.uint8)
    min_k_50 = torch.zeros(num_steps, B, n_layers, n_heads, N, dtype=torch.uint8)
    min_k_90 = torch.zeros(num_steps, B, n_layers, n_heads, N, dtype=torch.uint8)
    min_k_95 = torch.zeros(num_steps, B, n_layers, n_heads, N, dtype=torch.uint8)
    min_k_99 = torch.zeros(num_steps, B, n_layers, n_heads, N, dtype=torch.uint8)
    t_schedule = torch.zeros(num_steps, dtype=torch.float32)

    # 初始噪声 x_T
    x = torch.randn(B, IN_CHANNELS, LATENT_SIZE, LATENT_SIZE, device=device)

    for step_idx in tqdm(range(num_steps), desc="DDPM inference + attention collection"):
        t = timesteps[step_idx]
        t_schedule[step_idx] = float(t)
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

        # 模型预测 epsilon (noise)
        # DiT forward: output shape = (B, 8, 32, 32) with learn_sigma=True
        model_output = model.forward(x, t_tensor, labels)

        # 取前 4 通道作为 epsilon 预测
        if model.learn_sigma:
            epsilon = model_output[:, :model.in_channels]
        else:
            epsilon = model_output

        # 收集 attention
        attn_list = collect_attn_weights(model)

        # 计算 k×k 窗口 mass
        for layer_idx, attn in enumerate(attn_list):
            if attn is None:
                continue
            B2, H2, _, _ = attn.shape

            for ki, k in enumerate(k_values):
                idx = gather_idx_gpu[k].unsqueeze(0).unsqueeze(0)
                idx = idx.expand(B2, H2, -1, -1)
                gathered = attn.gather(dim=-1, index=idx)
                mass = gathered.sum(dim=-1)
                masses[step_idx, :, layer_idx, :, :, ki] = mass.half().cpu()

        # 向量化 min_k
        m_step = masses[step_idx]
        k_tensor = torch.tensor(k_values, dtype=torch.float32)
        for threshold, storage in [(0.50, min_k_50), (0.80, min_k_80),
                                    (0.90, min_k_90), (0.95, min_k_95),
                                    (0.99, min_k_99)]:
            above = m_step.float() >= threshold
            first_ki = above.float().argmax(dim=-1)
            any_above = above.any(dim=-1)
            storage[step_idx] = (k_tensor[first_ki] * any_above).byte()

        # DDPM 步进: x_{t-1} = coef1 * pred_x0 + coef2 * x_t (+ noise if t > 0)
        # pred_x0 = (x_t - sqrt(1-ᾱ_t) * ε) / sqrt(ᾱ_t)
        t_idx = t
        sqrt_alpha_bar = sqrt_alphas_cumprod[t_idx]
        sqrt_one_minus_alpha_bar = sqrt_one_minus_alphas_cumprod[t_idx]

        pred_x0 = (x - sqrt_one_minus_alpha_bar * epsilon) / sqrt_alpha_bar

        coef1 = posterior_mean_coef1[t_idx]
        coef2 = posterior_mean_coef2[t_idx]

        x_prev_mean = coef1 * pred_x0 + coef2 * x

        if t_idx > 0:
            noise = torch.randn_like(x)
            sigma = math.sqrt(posterior_variance[t_idx])
            x = x_prev_mean + sigma * noise
        else:
            x = x_prev_mean

    return {
        'masses': masses,
        'min_k_80': min_k_80, 'min_k_50': min_k_50,
        'min_k_90': min_k_90, 'min_k_95': min_k_95, 'min_k_99': min_k_99,
        't_schedule': t_schedule,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 4. 汇总分析 + 打印表格
# ═════════════════════════════════════════════════════════════════════════════

def analyze_and_print(data: dict, k_values: list):
    masses = data['masses']
    min_80 = data['min_k_80']

    S, B, L, H, N, K = masses.shape
    gs = int(N ** 0.5)

    # 空间区域划分
    region = np.zeros((gs, gs), dtype=int)
    for i in range(gs):
        for j in range(gs):
            d = min(i, j, gs - 1 - i, gs - 1 - j)
            if d <= 1:
                region[i, j] = 0
            elif d <= 3:
                region[i, j] = 1
            else:
                region[i, j] = 2
    region_flat = region.reshape(-1)
    region_counts = np.bincount(region_flat, minlength=3)
    region_names = ['corner', 'edge  ', 'interior']

    # 表1: 全局
    print("\n" + "=" * 110)
    print("表1 — 全局 Summary：k×k 窗口捕获 attention mass 的比例")
    print(f"     统计范围: {S} DDPM steps × {B} images × {L} layers × {H} heads × {N} queries")
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

    # 表2: 每层
    print("\n" + "=" * 110)
    print("表2 — 每层 k×k 窗口 >80% 达标比例")
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

    # 表3: Head 差异
    print("\n" + "=" * 110)
    print("表3 — Head 差异：每层的 min_k_80 统计")
    print("=" * 110)
    header4 = f"{'L':<4} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}  | 各 head 的 mean min_k"
    print(header4)
    print("-" * len(header4))
    for l in range(L):
        layer_data = min_80[:, :, l, :, :].float().flatten()
        layer_data = layer_data[layer_data > 0]
        if len(layer_data) == 0:
            print(f"{l:<4} {'N/A':>8}")
            continue
        head_means = []
        for h in range(H):
            h_data = min_80[:, :, l, h, :].float().flatten()
            h_data = h_data[h_data > 0]
            head_means.append(h_data.mean().item() if len(h_data) > 0 else 0)
        hdr = " ".join(f"{hm:>5.0f}" for hm in head_means)
        print(f"{l:<4} {layer_data.mean().item():>8.1f} {layer_data.std().item():>8.1f} "
              f"{layer_data.min().item():>8.0f} {layer_data.max().item():>8.0f}  | {hdr}")

    # 表4: min_k 分布
    print("\n" + "=" * 110)
    print("表4 — 每层 min_k_80 分布 (%)")
    print("=" * 110)
    header6 = f"{'L':<4}" + "".join(f"{'k='+str(k):>8}" for k in k_values) + f"  {'未达标':>8}"
    print(header6)
    print("-" * len(header6))
    for l in range(L):
        row = f"{l:<4}"
        layer_mk = min_80[:, :, l, :, :].flatten()
        total = len(layer_mk)
        for k in k_values:
            pct = (layer_mk == k).sum().item() / total * 100
            row += f"{pct:>8.1f}"
        pct_fail = (layer_mk == 0).sum().item() / total * 100
        row += f"  {pct_fail:>8.1f}"
        print(row)

    # 表5: 空间
    print("\n" + "=" * 110)
    print("表5 — 空间位置 min_k_80 均值")
    print("=" * 110)
    header5 = f"{'L':<4}" + "".join(f"{name:>12}" for name in region_names) + "  int-corner"
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
        row += f"  {vals[2]-vals[0]:>+12.1f}"
        print(row)


# ═════════════════════════════════════════════════════════════════════════════
# 5. Main
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str,
                   default=os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'DiT-XL-2-256x256.pt'))
    p.add_argument('--num_steps', type=int, default=250)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', type=str,
                   default=os.path.join(os.path.dirname(__file__), '..', 'outputs',
                                        'attention_locality_dit_xl', 'attention_locality_dit_xl.pt'))
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
    print("DiT-XL/2 DDPM Inference — Attention Locality Measurement")
    print(f"DDPM steps: {args.num_steps} | Batch: {args.batch_size} | Device: {device}")
    print(f"Grid: 16×16=256 tokens | Layers: {N_LAYERS} | Heads: {N_HEADS}")
    print(f"k values: {k_values}")
    print("=" * 60)

    # Load model
    print(f"\nLoading DiT-XL/2 from {args.ckpt}...")
    state_dict = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
    if "ema" in state_dict:
        state_dict = state_dict["ema"]
    model = DiT_XL_2(input_size=LATENT_SIZE, num_classes=1000, learn_sigma=True).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")

    # Patch attention
    patch_dit_attention(model)

    class_labels = [207, 360, 387, 974]

    print(f"\nRunning DDPM inference ({args.num_steps} steps × {len(class_labels)} images)...")
    data = run_ddpm_and_collect(
        model, class_labels,
        num_steps=args.num_steps,
        k_values=k_values,
        device=device,
        seed=args.seed,
    )

    analyze_and_print(data, k_values)

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
