#!/usr/bin/env python3
"""
MDTv2-XL/2 — DDPM 推理中 Attention Locality 全量测量

MDTv2 使用 DDPM 采样（预测 epsilon），t ∈ [0, 999] long integers。
Attention 是自定义 Attention 类（情况 A: 手动 softmax + relative position bias）。
推断时 ids_keep=None（不 masking）。

架构：en_inblocks(12) + en_outblocks(12) + de_blocks(4) + sideblocks(1) = 29 attention layers

Usage:
  conda activate mdt
  python measure/measure_locality_mdt.py \
    --ckpt checkpoints/mdt_xl2_v1_ckpt.pt \
    --num-steps 250 \
    --output outputs/attention_locality_mdtv2_xl/attention_locality_mdtv2_xl.pt
"""

import argparse, os, sys, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mdt_repo'))
from timm.models.vision_transformer import PatchEmbed
from masked_diffusion.models import MDTBlock, TimestepEmbedder, LabelEmbedder, FinalLayer, modulate

# MDTv1-XL/2 config (flat blocks + sideblocks, NOT encoder-decoder)
# Checkpoint has: blocks.0-27 (28 blocks) + sideblocks.0 (1 block)
N_BLOCKS = 28
N_SIDE = 1
N_LAYERS = N_BLOCKS + N_SIDE  # 29 total attention layers

N_HEADS = 16
GRID_SIZE = 16  # 32×32 latent, patch=2 → 16×16 grid
LATENT_SIZE = 32
IN_CHANNELS = 4


# ═════════════════════════════════════════════════════════════════════════════
# 0. MDTv1 model (flat blocks, matches checkpoint structure)
# ═════════════════════════════════════════════════════════════════════════════

class MDTv1(nn.Module):
    """MDTv1: flat DiT-style blocks + side interpolation block."""

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=True)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=True)

        self.blocks = nn.ModuleList([
            MDTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, num_patches=num_patches)
            for _ in range(depth)
        ])
        self.sideblocks = nn.ModuleList([
            MDTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, num_patches=num_patches)
            for _ in range(1)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        for block in self.sideblocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(shape=(x.shape[0], c, h * p, h * p))

    def forward(self, x, t, y, enable_mask=False):
        """Forward pass (inference mode: no masking)."""
        x = self.x_embedder(x) + self.pos_embed
        t = self.t_embedder(t)
        y = self.y_embedder(y, self.training)
        c = t + y

        for block in self.blocks:
            x = block(x, c, ids_keep=None)

        # Side interpolation (no masking during inference)
        x = x + self.decoder_pos_embed
        for sideblock in self.sideblocks:
            x = sideblock(x, c, ids_keep=None)

        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale=None, diffusion_steps=1000, scale_pow=4.0):
        if cfg_scale is not None:
            half = x[:len(x) // 2]
            combined = torch.cat([half, half], dim=0)
            model_out = self.forward(combined, t, y)
            eps, rest = model_out[:, :3], model_out[:, 3:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            scale_step = (1 - torch.cos(((1 - t / diffusion_steps) ** scale_pow) * math.pi)) * 1 / 2
            real_cfg_scale = (cfg_scale - 1) * scale_step + 1
            real_cfg_scale = real_cfg_scale[:len(x) // 2].view(-1, 1, 1, 1)
            half_eps = uncond_eps + real_cfg_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            return torch.cat([eps, rest], dim=1)
        else:
            return self.forward(x, t, y)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


# ═════════════════════════════════════════════════════════════════════════════
# 1. Monkey-patch MDT Attention: 抓取 post-softmax attention weights
# ═════════════════════════════════════════════════════════════════════════════
# MDT Attention 是 Case A：手动 softmax，可直接 hook

_COLLECTED_ATTN = {}
_ORIG_ATTN_FORWARDS = {}
_ATTN_ORDER = []  # 记录 attention module 被调用的顺序（按层序）


def _patched_mdt_attn_forward(self, x, ids_keep=None):
    """MDT Attention forward + 抓取 attention weights"""
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    attn = (q @ k.transpose(-2, -1)) * self.scale
    if ids_keep is not None:
        rp_bias = self.get_masked_rel_bias(B, ids_keep)
    else:
        rp_bias = self.rel_pos_bias()
    attn += rp_bias
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    # 唯一改动：保存 attention weights
    _COLLECTED_ATTN[id(self)] = attn.detach()

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def patch_mdt_attention(model):
    """替换所有 MDT Attention 的 forward 为可抓取 weights 的版本"""
    _COLLECTED_ATTN.clear()
    _ORIG_ATTN_FORWARDS.clear()
    _ATTN_ORDER.clear()

    all_blocks = list(model.blocks) + list(model.sideblocks)

    for block in all_blocks:
        attn = block.attn
        key = id(attn)
        if key not in _ORIG_ATTN_FORWARDS:
            _ORIG_ATTN_FORWARDS[key] = attn.forward
            attn.forward = _patched_mdt_attn_forward.__get__(attn, type(attn))
        _ATTN_ORDER.append(key)

    print(f"  Patched {len(all_blocks)} attention modules ({N_BLOCKS} blocks + {N_SIDE} side)")


def collect_attn_weights():
    """取出本次 forward 收集到的所有 attention weights（按层序）"""
    result = []
    for key in _ATTN_ORDER:
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
# 3. DDPM schedule（与 DiT 一致：linear schedule, 1000 steps）
# ═════════════════════════════════════════════════════════════════════════════

def get_ddpm_schedule(num_steps: int = 250, diffusion_steps: int = 1000):
    scale = 1000 / diffusion_steps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    betas = np.linspace(beta_start, beta_end, diffusion_steps, dtype=np.float64)

    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)

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


# ═════════════════════════════════════════════════════════════════════════════
# 4. DDPM 推理 + attention 采集
# ═════════════════════════════════════════════════════════════════════════════

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

    schedule = get_ddpm_schedule(num_steps)
    timesteps = schedule['timesteps']
    sqrt_alphas_cumprod = schedule['sqrt_alphas_cumprod']
    sqrt_one_minus_alphas_cumprod = schedule['sqrt_one_minus_alphas_cumprod']
    posterior_mean_coef1 = schedule['posterior_mean_coef1']
    posterior_mean_coef2 = schedule['posterior_mean_coef2']
    posterior_variance = schedule['posterior_variance']

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

        # 模型 forward（不用 forward_with_cfg，避免 CFG 混合 attention）
        # enable_mask=False → 不 masking，标准推理
        model_output = model.forward(x, t_tensor, labels, enable_mask=False)

        # 取前 IN_CHANNELS (4) 作为 epsilon 预测
        epsilon = model_output[:, :IN_CHANNELS]

        # 收集 attention
        attn_list = collect_attn_weights()

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

        # DDPM 步进
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
# 5. 汇总分析 + 打印表格
# ═════════════════════════════════════════════════════════════════════════════

def analyze_and_print(data: dict, k_values: list):
    masses = data['masses']
    min_80 = data['min_k_80']

    S, B, L, H, N, K = masses.shape
    gs = int(N ** 0.5)

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
    print(f"     层结构: blocks(0-{N_BLOCKS-1}) + side({N_BLOCKS})")
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
# 6. Main
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str,
                   default=os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'mdt_xl2_v1_ckpt.pt'))
    p.add_argument('--num_steps', type=int, default=250)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', type=str,
                   default=os.path.join(os.path.dirname(__file__), '..', 'outputs',
                                        'attention_locality_mdtv2_xl', 'attention_locality_mdtv2_xl.pt'))
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
    print("MDTv1-XL/2 DDPM Inference — Attention Locality Measurement")
    print(f"DDPM steps: {args.num_steps} | Batch: {args.batch_size} | Device: {device}")
    print(f"Grid: 16×16=256 tokens | Layers: {N_LAYERS} ({N_BLOCKS} blocks + {N_SIDE} side)")
    print(f"Heads: {N_HEADS} | k values: {k_values}")
    print("=" * 60)

    # Load model
    print(f"\nLoading MDTv1-XL/2 from {args.ckpt}...")
    model = MDTv1(input_size=LATENT_SIZE, depth=N_BLOCKS, num_heads=N_HEADS,
                  num_classes=1000, learn_sigma=True).to(device)
    state_dict = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict, strict=False)  # mask_token only used in training
    model.eval()
    print(f"Loaded: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")

    # Patch attention
    patch_mdt_attention(model)

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
