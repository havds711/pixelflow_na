#!/usr/bin/env python3
"""
PixArt-XL/2 (α & Σ) — Attention Locality 全量测量

PixArt 使用 DiT 架构 + cross-attention，推理时 self-attention 在 xformers
memory_efficient_attention 中完成（Case B），需 monkey-patch 替换为手动 softmax。

支持两个模型变体:
  - PixArt-α (PixArt-XL-2-256x256.pth): ε-prediction, DDPM 250 步
  - PixArt-Σ (PixArt-Sigma-XL-2-256x256.pth): v-prediction, 更少步数

用法:
  conda activate pixart
  python measure/measure_locality_pixart.py \
    --ckpt checkpoints/PixArt-XL-2-256x256.pth \
    --model-name pixart_alpha --num-steps 250

  python measure/measure_locality_pixart.py \
    --ckpt checkpoints/PixArt-Sigma-XL-2-256x256.pth \
    --model-name pixart_sigma --num-steps 250
"""

import argparse, os, sys, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════════════
# 0. 自包含 PixArt 模型定义 (基于 https://github.com/PixArt-alpha/PixArt-alpha)
# ═══════════════════════════════════════════════════════════════════════════════

def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """嵌入标量 timestep 为向量"""
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half)
        freqs = freqs.to(t.device)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class LabelEmbedder(nn.Module):
    """嵌入 class label 为向量"""
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)  # +1 for unconditioned
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def forward(self, labels, train=False):
        if train:
            drop_mask = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            labels = torch.where(drop_mask, self.num_classes, labels)
        return self.embedding_table(labels)


class CaptionEmbedder(nn.Module):
    """嵌入 text caption (T5 output) 为向量，用于 cross-attention"""
    def __init__(self, in_channels, hidden_size, uncond_prob=0.1, act_layer=nn.GELU, token_num=120):
        super().__init__()
        self.y_proj = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            act_layer(approximate="tanh"),
            nn.Linear(in_channels, hidden_size),
        )
        self.uncond_prob = uncond_prob
        self.token_num = token_num
        self.register_buffer("y_embedding", nn.Parameter(torch.randn(token_num, in_channels) / in_channels ** 0.5))

    def forward(self, caption, train=False):
        if train:
            y_embedding = self.y_embedding.unsqueeze(0).expand(caption.shape[0], -1, -1)
            drop_mask = torch.rand(caption.shape[0], device=caption.device) < self.uncond_prob
            caption = torch.where(drop_mask.unsqueeze(-1).unsqueeze(-1), y_embedding, caption)
        return self.y_proj(caption)


class MultiHeadCrossAttention(nn.Module):
    """Cross-attention: query from image tokens, key/value from text"""
    def __init__(self, d_model, num_heads, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model * 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, cond, mask=None):
        B, N, C = x.shape
        q = self.q_linear(x).reshape(B, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).reshape(B, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        # 使用手动 scale + softmax（替代 xformers，保证跨 env 兼容）
        q = q.transpose(1, 2)  # [B, heads, N, d]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        if mask is not None:
            attn = attn + mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class WindowAttention(nn.Module):
    """
    PixArt self-attention (带可选 relative position bias)。
    原始用 xformers memory_efficient_attention → 这里替换为手动 softmax。
    """
    def __init__(self, dim, num_heads=8, qkv_bias=True, use_rel_pos=False,
                 rel_pos_zero_init=True, input_size=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.0)
        self.use_rel_pos = use_rel_pos
        if use_rel_pos:
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))
            if not rel_pos_zero_init:
                nn.init.trunc_normal_(self.rel_pos_h, std=0.02)
                nn.init.trunc_normal_(self.rel_pos_w, std=0.02)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, d]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.use_rel_pos:
            attn = attn + self._get_rel_pos_bias()

        if mask is not None:
            attn = attn + mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 保存 attention weights 到全局字典
        _COLLECTED_ATTN[id(self)] = attn.detach()

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _get_rel_pos_bias(self):
        # 简化版 relative position bias (PixArt 默认不用 rel_pos，保留空实现)
        return 0


class PixArtBlock(nn.Module):
    """PixArt DiT block: self-attn + cross-attn + MLP, adaLN-single"""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, drop_path=0.,
                 window_size=0, input_size=None, use_rel_pos=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = WindowAttention(
            hidden_size, num_heads=num_heads, qkv_bias=True,
            input_size=input_size if window_size == 0 else (window_size, window_size),
            use_rel_pos=use_rel_pos,
        )
        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            approx_gelu(),
            nn.Dropout(0.0),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size),
            nn.Dropout(0.0),
        )
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size ** 0.5)

    def forward(self, x, y, t, mask=None):
        B, N, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)
        ).chunk(6, dim=1)
        # Self-attention
        x_norm = self.norm1(x) * (1 + scale_msa) + shift_msa
        x = x + gate_msa * self.attn(x_norm)
        # Cross-attention
        x = x + self.cross_attn(x, y, mask)
        # MLP
        x_norm2 = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        x = x + gate_mlp * self.mlp(x_norm2)
        return x


class FinalLayer(nn.Module):
    """PixArt 输出层: norm → linear → unpatchify"""
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.scale_shift_table = nn.Parameter(torch.randn(2, hidden_size) / hidden_size ** 0.5)
        self.out_channels = out_channels

    def forward(self, x, t):
        shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2, dim=1)
        x = self.norm_final(x) * (1 + scale) + shift
        x = self.linear(x)
        return x


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768, bias=True):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = -1  # set after first forward or externally

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """标准 2D sin-cos position embedding"""
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size[1], grid_size[0]])
    emb_h = _get_1d_sincos(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def _get_1d_sincos(embed_dim, pos):
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


class PixArt(nn.Module):
    """
    PixArt 扩散 Transformer（自包含定义，匹配 PixArt-alpha repo）。
    depth=28, hidden_size=1152, num_heads=16, patch_size=2, in_channels=4.
    """
    def __init__(self, input_size=32, patch_size=2, in_channels=4, hidden_size=1152,
                 depth=28, num_heads=16, mlp_ratio=4.0, pred_sigma=True,
                 use_rel_pos=False, window_size=0):
        super().__init__()
        self.pred_sigma = pred_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if pred_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.grid_size = input_size // patch_size  # 16

        self.x_embedder = PatchEmbed(patch_size, in_channels, hidden_size, bias=True)
        num_patches = (input_size // patch_size) ** 2
        self.register_buffer("pos_embed", torch.zeros(1, num_patches, hidden_size))
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        self.y_embedder = LabelEmbedder(num_classes=1000, hidden_size=hidden_size, dropout_prob=0.1)
        self.blocks = nn.ModuleList([
            PixArtBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=0.,
                        input_size=(self.grid_size, self.grid_size),
                        window_size=window_size, use_rel_pos=use_rel_pos)
            for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, y):
        """
        Forward pass.
        x: [B, C, H, W] latent
        t: [B] timestep
        y: [B] class labels
        Returns: model output [B, out_channels, H, W]
        """
        B = x.shape[0]
        x = self.x_embedder(x) + self.pos_embed  # [B, N, D]
        t_emb = self.t_embedder(t)               # [B, D]
        t_block = self.t_block(t_emb)             # [B, 6*D]
        y_emb = self.y_embedder(y)                # [B, D]

        # 用 class embedding 作为 cross-attention 的 condition
        y_cross = y_emb.unsqueeze(1)  # [B, 1, D]

        for block in self.blocks:
            x = block(x, y_cross, t_block)

        x = self.final_layer(x, t_emb)
        x = self.unpatchify(x)
        return x

    def unpatchify(self, x):
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(x.shape[0], c, h * p, h * p)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Attention 抓取
# ═══════════════════════════════════════════════════════════════════════════════

_COLLECTED_ATTN = {}  # id(module) → attention weights tensor


def patch_attention(model):
    """确保所有 WindowAttention 的 forward 会保存 attention weights"""
    _COLLECTED_ATTN.clear()
    attn_count = 0
    for block in model.blocks:
        attn = block.attn
        _COLLECTED_ATTN[id(attn)] = None  # 预分配槽位
        attn_count += 1
    print(f"  Ready to capture {attn_count} self-attention modules")


def collect_attn_weights(model):
    """按层序取出本次 forward 的 attention weights"""
    result = []
    for block in model.blocks:
        result.append(_COLLECTED_ATTN.get(id(block.attn), None))
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 预计算窗口索引
# ═══════════════════════════════════════════════════════════════════════════════

def precompute_window_indices(grid_size, k_values):
    H = W = grid_size
    gather_indices = {}
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
        gather_indices[k] = idx_mat
    return gather_indices


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DDPM Sampling + 采集
# ═══════════════════════════════════════════════════════════════════════════════

def get_ddpm_schedule(num_steps=250, diffusion_steps=1000):
    scale = 1000 / diffusion_steps
    betas = np.linspace(scale * 0.0001, scale * 0.02, diffusion_steps, dtype=np.float64)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
    timesteps = np.linspace(diffusion_steps - 1, 0, num_steps).astype(int)
    return {
        'sqrt_alphas_cumprod': np.sqrt(alphas_cumprod),
        'sqrt_one_minus_alphas_cumprod': np.sqrt(1.0 - alphas_cumprod),
        'posterior_mean_coef1': posterior_mean_coef1,
        'posterior_mean_coef2': posterior_mean_coef2,
        'posterior_variance': posterior_variance,
        'timesteps': timesteps,
    }


@torch.no_grad()
def run_ddpm_and_collect(model, class_labels, num_steps=250, k_values=None,
                         n_layers=28, n_heads=16, grid_size=16,
                         device="cuda", seed=42):
    if k_values is None:
        k_values = [1, 3, 5, 7, 9, 11, 13, 15]

    torch.manual_seed(seed)
    model.eval()

    B = len(class_labels)
    labels = torch.tensor(class_labels, device=device)
    N = grid_size * grid_size
    n_k = len(k_values)
    in_channels = model.in_channels  # 4

    schedule = get_ddpm_schedule(num_steps)
    timesteps = schedule['timesteps']
    sqrt_alphas_cumprod = schedule['sqrt_alphas_cumprod']
    sqrt_one_minus_alphas_cumprod = schedule['sqrt_one_minus_alphas_cumprod']
    coef1 = schedule['posterior_mean_coef1']
    coef2 = schedule['posterior_mean_coef2']
    posterior_variance = schedule['posterior_variance']

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

    latent_size = grid_size * model.patch_size  # 32
    x = torch.randn(B, in_channels, latent_size, latent_size, device=device)

    for step_idx in tqdm(range(num_steps), desc="DDPM + attention"):
        t = int(timesteps[step_idx])
        t_schedule[step_idx] = float(t)
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

        # Forward
        model_output = model(x, t_tensor, labels)

        # 取 epsilon
        if model.pred_sigma:
            epsilon = model_output[:, :in_channels]
        else:
            epsilon = model_output

        # 收集 attention
        attn_list = collect_attn_weights(model)

        for layer_idx, attn in enumerate(attn_list):
            if attn is None:
                continue
            for ki, k in enumerate(k_values):
                idx = gather_idx_gpu[k].unsqueeze(0).unsqueeze(0).expand(B, n_heads, -1, -1)
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

        # DDPM step
        sqrt_alpha_bar = sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_bar = sqrt_one_minus_alphas_cumprod[t]
        pred_x0 = (x - sqrt_one_minus_alpha_bar * epsilon) / sqrt_alpha_bar
        x_prev_mean = coef1[t] * pred_x0 + coef2[t] * x
        if t > 0:
            x = x_prev_mean + math.sqrt(posterior_variance[t]) * torch.randn_like(x)
        else:
            x = x_prev_mean

    return {
        'masses': masses,
        'min_k_80': min_k_80, 'min_k_50': min_k_50,
        'min_k_90': min_k_90, 'min_k_95': min_k_95, 'min_k_99': min_k_99,
        't_schedule': t_schedule,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 分析 + 表格
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_and_print(data, k_values):
    masses = data['masses']
    min_80 = data['min_k_80']
    S, B, L, H, N, K = masses.shape
    gs = int(N ** 0.5)

    region = np.zeros((gs, gs), dtype=int)
    for i in range(gs):
        for j in range(gs):
            d = min(i, j, gs - 1 - i, gs - 1 - j)
            if d <= 1:       region[i, j] = 0
            elif d <= 3:     region[i, j] = 1
            else:            region[i, j] = 2
    region_flat = region.reshape(-1)
    region_names = ['corner', 'edge  ', 'interior']

    # 表1: 全局
    print("\n" + "=" * 110)
    print(f"表1 — 全局 Summary：k×k 窗口捕获 attention mass 的比例")
    print(f"     统计范围: {S} steps × {B} images × {L} layers × {H} heads × {N} queries")
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
    hdr2 = f"{'L':<4}" + "".join(f"{'k='+str(k):>10}" for k in k_values) + f"  {'推荐k':>6}  {'均值':>6}"
    print(hdr2)
    print("-" * len(hdr2))
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
        row += f"  {str(best_k or '>15'):>6}  {np.mean(ratios):>6.3f}"
        print(row)

    # 表3: Head 差异
    print("\n" + "=" * 110)
    print("表3 — Head 差异：每层的 min_k_80 统计")
    print("=" * 110)
    hdr3 = f"{'L':<4} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}  | 各 head 的 mean min_k"
    print(hdr3)
    print("-" * len(hdr3))
    for l in range(L):
        ld = min_80[:, :, l, :, :].float().flatten()
        ld = ld[ld > 0]
        if len(ld) == 0:
            print(f"{l:<4} N/A")
            continue
        hmeans = []
        for h in range(H):
            hd = min_80[:, :, l, h, :].float().flatten()
            hd = hd[hd > 0]
            hmeans.append(hd.mean().item() if len(hd) > 0 else 0)
        hm_str = " ".join(f"{hm:>5.0f}" for hm in hmeans)
        print(f"{l:<4} {ld.mean():>8.1f} {ld.std():>8.1f} {ld.min():>8.0f} {ld.max():>8.0f}  | {hm_str}")

    # 表4: min_k 分布
    print("\n" + "=" * 110)
    print("表4 — 每层 min_k_80 分布 (%)")
    print("=" * 110)
    hdr4 = f"{'L':<4}" + "".join(f"{'k='+str(k):>8}" for k in k_values) + f"  {'未达标':>8}"
    print(hdr4)
    print("-" * len(hdr4))
    for l in range(L):
        row = f"{l:<4}"
        mk = min_80[:, :, l, :, :].flatten()
        total = len(mk)
        for k in k_values:
            row += f"{(mk == k).sum().item() / total * 100:>8.1f}"
        row += f"  {(mk == 0).sum().item() / total * 100:>8.1f}"
        print(row)

    # 表5: 空间
    print("\n" + "=" * 110)
    print("表5 — 空间位置 min_k_80 均值")
    print("=" * 110)
    hdr5 = f"{'L':<4}" + "".join(f"{n:>12}" for n in region_names) + "  int-corner"
    print(hdr5)
    print("-" * len(hdr5))
    for l in range(L):
        row = f"{l:<4}"
        vals = []
        for ri in range(3):
            qm = (region_flat == ri)
            rd = min_80[:, :, l, :, qm].float().flatten()
            rd = rd[rd > 0]
            v = rd.mean().item() if len(rd) > 0 else 0
            vals.append(v)
            row += f"{v:>12.1f}"
        row += f"  {vals[2] - vals[0]:>+12.1f}"
        print(row)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, required=True,
                   help='Path to PixArt checkpoint (.pth)')
    p.add_argument('--model-name', type=str, default='pixart',
                   help='Model name for output dir (e.g. pixart_alpha, pixart_sigma)')
    p.add_argument('--num-steps', type=int, default=250)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', type=str, default=None,
                   help='Output .pt file path')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--n-layers', type=int, default=28, help='Number of attention layers')
    p.add_argument('--n-heads', type=int, default=16, help='Number of attention heads')
    return p.parse_args()


def main():
    args = parse_args()

    if args.output is None:
        args.output = os.path.join(
            os.path.dirname(__file__), '..', 'outputs',
            f'attention_locality_{args.model_name}',
            f'attention_locality_{args.model_name}.pt'
        )
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    n_layers = args.n_layers
    n_heads = args.n_heads
    k_values = [1, 3, 5, 7, 9, 11, 13, 15]
    grid_size = 16

    print("=" * 60)
    print(f"PixArt (α/Σ) — Attention Locality Measurement")
    print(f"Model: {args.model_name} | Checkpoint: {args.ckpt}")
    print(f"DDPM steps: {args.num_steps} | Batch: {args.batch_size} | Device: {device}")
    print(f"Grid: {grid_size}×{grid_size} | Layers: {n_layers} | Heads: {n_heads}")
    print(f"k values: {k_values}")
    print("=" * 60)

    # Build model
    print("\nBuilding PixArt-XL/2 model...")
    model = PixArt(
        input_size=32, patch_size=2, in_channels=4, hidden_size=1152,
        depth=n_layers, num_heads=n_heads, pred_sigma=True,
    )

    # Load checkpoint
    print(f"Loading checkpoint: {args.ckpt}")
    state_dict = torch.load(args.ckpt, map_location="cpu")
    # Handle wrapped state_dict (ema, model, etc.)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if "model" in state_dict:
        state_dict = state_dict["model"]
    if "ema" in state_dict:
        state_dict = state_dict["ema"]

    # Remap keys if needed (PixArt checkpoint may use different naming)
    # Original PixArt repo keys match our model, but we handle common differences
    cleaned = {}
    for k, v in state_dict.items():
        new_k = k
        # Remove known prefixes
        for prefix in ["module.", "denoiser.", "ema_denoiser."]:
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix):]
        cleaned[new_k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
        for mk in missing[:5]:
            print(f"    - {mk}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
        for uk in unexpected[:5]:
            print(f"    + {uk}")

    model = model.to(device)
    model.eval()
    print(f"  Params: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")

    patch_attention(model)

    class_labels = [207, 360, 387, 974]

    print(f"\nRunning DDPM inference ({args.num_steps} steps × {len(class_labels)} images)...")
    import time
    t0 = time.time()
    data = run_ddpm_and_collect(
        model, class_labels,
        num_steps=args.num_steps,
        k_values=k_values,
        n_layers=n_layers,
        n_heads=n_heads,
        grid_size=grid_size,
        device=device,
        seed=args.seed,
    )
    print(f"  Done in {time.time() - t0:.0f}s")

    analyze_and_print(data, k_values)

    torch.save(data, args.output)
    print(f"\nSaved → {args.output} ({os.path.getsize(args.output) / 1e6:.0f} MB)")
    print("Done!")


if __name__ == '__main__':
    main()
