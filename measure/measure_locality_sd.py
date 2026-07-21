#!/usr/bin/env python3
"""
Stable Diffusion (1.5 & XL) — Self-Attention Locality 全量测量

SD UNet 在不同分辨率（32×32, 16×16, 8×8）上有 self-attention，使用
F.scaled_dot_product_attention（Case B），需 monkey-patch AttnProcessor 来抓取权重。

与 DiT 模型不同，SD 使用 text prompt 而非 class label 作为条件。
我们使用固定的 4 个 prompt，保持与 class-conditional 模型一致的评估方式。

用法:
  conda activate sd15
  python measure/measure_locality_sd.py \
    --model sd15 --ckpt checkpoints/sd/v1-5-pruned-emaonly.safetensors \
    --num-steps 50 --resolution 512

  python measure/measure_locality_sd.py \
    --model sdxl --ckpt checkpoints/sd/sd_xl_base_1.0.safetensors \
    --num-steps 50 --resolution 512
"""

import argparse, hashlib, os, sys, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 自定义 AttnProcessor: 捕获 self-attention weights
# ═══════════════════════════════════════════════════════════════════════════════

_COLLECTED_SELF_ATTN = {}   # module_key → attention weights
_SELF_ATTN_KEYS = []        # 按层序记录哪些 key 是 self-attention


class CaptureAttnProcessor:
    """
    替换默认 AttnProcessor2_0，使用手动 softmax 以捕获 self-attention weights。
    保留原始推理逻辑不变。
    """
    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, *args, **kwargs):
        # 只处理 self-attention（encoder_hidden_states is None）
        is_self_attn = (encoder_hidden_states is None)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            B, C, H, W = hidden_states.shape
            hidden_states = hidden_states.view(B, C, H * W).transpose(1, 2)

        B, N, C = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # QKV projection
        if encoder_hidden_states is not None:
            # Cross-attention: Q from hidden_states, KV from encoder_hidden_states
            query = attn.to_q(hidden_states)
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
        else:
            # Self-attention
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

        # Reshape to multi-head: [B, N, heads*dim] → [B, heads, N, dim]
        # 注意：用 query 的实际输出维度计算 head_dim，不是 hidden_states 的 C
        # SD XL 中 QKV projection 输出维度可能和输入维度不同
        head_dim = query.shape[-1] // attn.heads
        query = query.view(B, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(B, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(B, -1, attn.heads, head_dim).transpose(1, 2)

        # Manual attention (replaces F.scaled_dot_product_attention)
        scale = head_dim ** -0.5
        attn_weights = query @ key.transpose(-2, -1) * scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = attn_weights.softmax(dim=-1)

        # 捕获 self-attention weights
        if is_self_attn:
            module_key = _ATTN_ID_MAP.get(id(attn), None)
            if module_key is not None:
                _COLLECTED_SELF_ATTN[module_key] = attn_weights.detach()

        # Attention dropout
        attn_weights = F.dropout(attn_weights, p=0.0, training=False)

        hidden_states = attn_weights @ value
        # 用实际 dim 而非原始 hidden_states 的 C（SD XL 可能不同）
        hidden_states = hidden_states.transpose(1, 2).reshape(B, N, -1)

        # Output projection
        hidden_states = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:
            hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(B, C, H, W)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        return hidden_states / attn.rescale_output_factor


_ATTN_ID_MAP = {}  # id(attn_module) → "down_32x32_block0_attn0"


def patch_unet_attention(unet, model_type="sd15"):
    """
    遍历 UNet，找到所有 self-attention 模块，替换 AttnProcessor。
    记录每个 self-attention 的位置（分辨率、块索引）用于后续分析。
    """
    global _ATTN_ID_MAP, _SELF_ATTN_KEYS
    _COLLECTED_SELF_ATTN.clear()
    _ATTN_ID_MAP.clear()
    _SELF_ATTN_KEYS = []

    attn_count = 0

    # 递归搜索所有 Attention 模块
    def _scan_and_patch(module, prefix=""):
        nonlocal attn_count
        from diffusers.models.attention import Attention as DiffAttn

        for name, child in module.named_children():
            child_prefix = f"{prefix}.{name}" if prefix else name
            if isinstance(child, DiffAttn):
                # 判断是 self-attn (name == "attn1") 还是 cross-attn (name == "attn2")
                if "attn1" in name:
                    key = child_prefix
                    _ATTN_ID_MAP[id(child)] = key
                    _SELF_ATTN_KEYS.append(key)
                    # 替换 processor
                    child.set_processor(CaptureAttnProcessor())
                    attn_count += 1
            else:
                _scan_and_patch(child, child_prefix)

    _scan_and_patch(unet)
    print(f"  Patched {attn_count} self-attention modules")
    for k in _SELF_ATTN_KEYS:
        print(f"    {k}")


def collect_self_attn_weights():
    """按层序取出本次 forward 的 self-attention weights"""
    result = []
    for key in _SELF_ATTN_KEYS:
        result.append(_COLLECTED_SELF_ATTN.get(key, None))
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 预计算窗口索引
# ═══════════════════════════════════════════════════════════════════════════════

def precompute_window_indices(grid_size, k_values):
    H = W = grid_size
    gather_indices = {}
    for k in k_values:
        if k > grid_size:
            continue  # k 不能大于 grid
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
# 3. DDIM 采样 + 采集
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_ddim_and_collect(
    pipeline,
    prompts,
    num_steps=50,
    k_values_per_grid=None,
    device="cuda",
    seed=42,
    resolution=512,
):
    """
    运行 DDIM 采样，在每一步捕获所有 self-attention weights。
    输出按 grid_size 分组的数据。
    """
    if k_values_per_grid is None:
        k_values_per_grid = {}

    torch.manual_seed(seed)

    # 设置 scheduler 步数
    pipeline.scheduler.set_timesteps(num_steps, device=device)
    timesteps = pipeline.scheduler.timesteps

    B = len(prompts)

    # 编码 text prompts
    text_embeddings = pipeline.encode_prompt(
        prompts, device, 1, do_classifier_free_guidance=False
    )[0]  # [B, 77, 768] for SD 1.5

    # 初始噪声
    latents_shape = (B, pipeline.unet.config.in_channels, resolution // 8, resolution // 8)
    latents = torch.randn(latents_shape, device=device, generator=torch.Generator(device).manual_seed(seed))

    # 先跑一步确认 attention 层信息和 grid sizes
    print("  Dry run to discover attention layers...")
    _COLLECTED_SELF_ATTN.clear()
    t0 = timesteps[0]
    latent_model_input = pipeline.scheduler.scale_model_input(latents, t0)
    added_kw = getattr(pipeline, '_added_cond_kwargs', None)
    unet_kwargs = {'added_cond_kwargs': added_kw} if added_kw else {}
    pipeline.unet(
        latent_model_input, t0, encoder_hidden_states=text_embeddings,
        return_dict=False, **unet_kwargs,
    )
    # 此时 _SELF_ATTN_KEYS 已填充，_COLLECTED_SELF_ATTN 有数据

    # 解析每层的 grid_size 和 heads（SD XL 不同分辨率可能 head 数不同）
    layer_grid_sizes = []
    layer_head_counts = []
    for key in _SELF_ATTN_KEYS:
        attn = _COLLECTED_SELF_ATTN.get(key)
        if attn is not None:
            layer_grid_sizes.append(int(attn.shape[-1] ** 0.5))
            layer_head_counts.append(attn.shape[1])
        else:
            layer_grid_sizes.append(None)
            layer_head_counts.append(None)

    # 按 grid_size 分组
    grids_in_model = sorted(set(gs for gs in layer_grid_sizes if gs is not None))
    print(f"  Found grid sizes: {grids_in_model}")
    for gs in grids_in_model:
        count = sum(1 for s in layer_grid_sizes if s == gs)
        print(f"    {gs}×{gs}: {count} layers")

    # 为每个 grid size 确定 k-values
    final_k_values_per_grid = {}
    for gs in grids_in_model:
        if gs in k_values_per_grid:
            final_k_values_per_grid[gs] = k_values_per_grid[gs]
        else:
            # 默认：从 1 到 grid_size-1 的奇数
            max_k = min(gs, 31)
            final_k_values_per_grid[gs] = [k for k in range(1, max_k + 1, 2)]

    # 预计算每个 grid size 的窗口索引
    gather_idx_gpu = {}
    for gs in grids_in_model:
        k_vals = final_k_values_per_grid[gs]
        gather_idx = precompute_window_indices(gs, k_vals)
        gather_idx_gpu[gs] = {k: v.to(device) for k, v in gather_idx.items()}

    # 为每个 grid size 创建存储
    n_heads = pipeline.unet.config.attention_head_dim
    # 对于标准 SD 1.5: head_dim = 8, but each attention module has `heads` attribute

    # 先获取 heads 数
    _COLLECTED_SELF_ATTN.clear()
    added_kw = getattr(pipeline, '_added_cond_kwargs', None)
    unet_kwargs = {'added_cond_kwargs': added_kw} if added_kw else {}
    pipeline.unet(
        latent_model_input, t0, encoder_hidden_states=text_embeddings,
        return_dict=False, **unet_kwargs,
    )
    sample_attn = _COLLECTED_SELF_ATTN.get(_SELF_ATTN_KEYS[0])
    if sample_attn is not None:
        n_heads = sample_attn.shape[1]
    else:
        n_heads = 8  # SD 1.5 default

    # 初始化存储（按 grid_size 分组，每个 grid 可能 head 数不同）
    # 确定每个 grid 的 max heads
    grid_heads = {}
    for gs in grids_in_model:
        heads_in_grid = [h for h, s in zip(layer_head_counts, layer_grid_sizes) if s == gs and h is not None]
        grid_heads[gs] = max(heads_in_grid) if heads_in_grid else 8
    print(f"  Grid head counts: {grid_heads}")

    storage = {}
    for gs in grids_in_model:
        n_layers_gs = sum(1 for s in layer_grid_sizes if s == gs)
        n_k_gs = len(final_k_values_per_grid[gs])
        N_gs = gs * gs
        n_h_gs = grid_heads[gs]
        storage[gs] = {
            'masses': torch.zeros(num_steps, B, n_layers_gs, n_h_gs, N_gs, n_k_gs, dtype=torch.float16),
            'min_k_80': torch.zeros(num_steps, B, n_layers_gs, n_h_gs, N_gs, dtype=torch.uint8),
            'min_k_50': torch.zeros(num_steps, B, n_layers_gs, n_h_gs, N_gs, dtype=torch.uint8),
            'min_k_90': torch.zeros(num_steps, B, n_layers_gs, n_h_gs, N_gs, dtype=torch.uint8),
            'min_k_95': torch.zeros(num_steps, B, n_layers_gs, n_h_gs, N_gs, dtype=torch.uint8),
            'min_k_99': torch.zeros(num_steps, B, n_layers_gs, n_h_gs, N_gs, dtype=torch.uint8),
            'k_values': final_k_values_per_grid[gs],
            'layer_indices': [i for i, s in enumerate(layer_grid_sizes) if s == gs],
            'layer_heads': [h for h, s in zip(layer_head_counts, layer_grid_sizes) if s == gs],
        }

    t_schedule = torch.zeros(num_steps, dtype=torch.float32)

    # 主循环
    for step_idx in tqdm(range(num_steps), desc="DDIM + attention"):
        t = timesteps[step_idx]
        t_schedule[step_idx] = t.item()

        latent_model_input = pipeline.scheduler.scale_model_input(latents, t)
        _COLLECTED_SELF_ATTN.clear()

        added_kw = getattr(pipeline, '_added_cond_kwargs', None)
        unet_kwargs = {'added_cond_kwargs': added_kw} if added_kw else {}
        noise_pred = pipeline.unet(
            latent_model_input, t, encoder_hidden_states=text_embeddings,
            return_dict=False, **unet_kwargs,
        )[0]

        # 收集 attention
        attn_list = collect_self_attn_weights()

        # 按 grid_size 分组处理
        for gs, st in storage.items():
            kv = st['k_values']
            layer_indices = st['layer_indices']
            layer_heads_list = st['layer_heads']
            n_h_gs = st['masses'].shape[3]  # 此 grid 分配的 head 维度
            for gs_layer_idx, global_layer_idx in enumerate(layer_indices):
                attn = attn_list[global_layer_idx]
                if attn is None:
                    continue
                _, H2, N2, _ = attn.shape
                H_use = min(H2, n_h_gs)  # SD XL 可能不同层 head 数不同
                for ki, k in enumerate(kv):
                    idx = gather_idx_gpu[gs][k].unsqueeze(0).unsqueeze(0).expand(B, H2, -1, -1)
                    gathered = attn.gather(dim=-1, index=idx)
                    mass = gathered.sum(dim=-1)  # [B, H2, N2]
                    st['masses'][step_idx, :, gs_layer_idx, :H_use, :N2, ki] = mass[:, :H_use, :].half().cpu()

            # 向量化 min_k
            m_step = st['masses'][step_idx]
            k_tensor = torch.tensor(kv, dtype=torch.float32)
            for threshold, sname in [(0.50, 'min_k_50'), (0.80, 'min_k_80'),
                                      (0.90, 'min_k_90'), (0.95, 'min_k_95'),
                                      (0.99, 'min_k_99')]:
                above = m_step.float() >= threshold
                first_ki = above.float().argmax(dim=-1)
                any_above = above.any(dim=-1)
                st[sname][step_idx] = (k_tensor[first_ki] * any_above).byte()

        # DDIM step
        latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    return {
        'by_grid': storage,
        't_schedule': t_schedule,
        'layer_grid_sizes': layer_grid_sizes,
        'layer_names': _SELF_ATTN_KEYS,
        'prompts': prompts,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 分析 + 表格
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_and_print(data):
    by_grid = data['by_grid']
    print("\n" + "=" * 70)
    print("SD Self-Attention Locality — Multi-Resolution Analysis")
    print("=" * 70)

    for gs in sorted(by_grid.keys()):
        st = by_grid[gs]
        masses = st['masses']
        min_80 = st['min_k_80']
        kv = st['k_values']
        S, B, L, H, N, K = masses.shape

        print(f"\n{'─' * 70}")
        print(f"  Grid: {gs}×{gs} ({N} tokens) — {L} self-attention layers")
        print(f"  k values: {kv}")
        print(f"{'─' * 70}")

        # 全局摘要
        header = f"{'k':<8} {'>50%':>10} {'>80%':>10} {'>90%':>10} {'>95%':>10} {'>99%':>10}  {'推荐':>6}"
        print(header)
        print("-" * len(header))
        for ki, k in enumerate(kv):
            r50 = (masses[:, :, :, :, :, ki] > 0.50).float().mean().item()
            r80 = (masses[:, :, :, :, :, ki] > 0.80).float().mean().item()
            r90 = (masses[:, :, :, :, :, ki] > 0.90).float().mean().item()
            r95 = (masses[:, :, :, :, :, ki] > 0.95).float().mean().item()
            r99 = (masses[:, :, :, :, :, ki] > 0.99).float().mean().item()
            flag = " ←" if r80 >= 0.8 else ""
            print(f"k={k:<5} {r50:>10.3f} {r80:>10.3f} {r90:>10.3f} {r95:>10.3f} {r99:>10.3f}  {flag}")

        # 每层
        if L > 1:
            print(f"\n  Per-layer k >80% rate:")
            hdr2 = f"  {'L':<4}" + "".join(f"{'k='+str(k):>8}" for k in kv) + f"  {'推荐k':>6}"
            print(hdr2)
            print("  " + "-" * (len(hdr2) - 2))
            for l in range(L):
                row = f"  {l:<4}"
                best_k = None
                for ki, k in enumerate(kv):
                    r = (masses[:, :, l, :, :, ki] > 0.80).float().mean().item()
                    row += f"{r:>8.3f}"
                    if best_k is None and r >= 0.8:
                        best_k = k
                row += f"  {str(best_k or '>' + str(kv[-1])):>6}"
                print(row)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, required=True, choices=['sd15', 'sdxl'],
                   help='Model variant: sd15 or sdxl')
    p.add_argument('--ckpt', type=str, required=True,
                   help='Path to model checkpoint (.safetensors)')
    p.add_argument('--num-steps', type=int, default=50,
                   help='DDIM sampling steps')
    p.add_argument('--resolution', type=int, default=512,
                   help='Image resolution (must be multiple of 8)')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', type=str, default=None,
                   help='Output .pt file path')
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()


def main():
    args = parse_args()

    if args.output is None:
        args.output = os.path.join(
            os.path.dirname(__file__), '..', 'outputs',
            f'attention_locality_{args.model}',
            f'attention_locality_{args.model}.pt'
        )
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    print("=" * 60)
    print(f"Stable Diffusion — Self-Attention Locality Measurement")
    print(f"Model: {args.model} | Resolution: {args.resolution} | Steps: {args.num_steps}")
    print(f"Batch: {args.batch_size} | Device: {device}")
    print("=" * 60)

    # 加载 pipeline (offline — 不依赖 HuggingFace Hub)
    from diffusers import UNet2DConditionModel, DDIMScheduler
    from diffusers.pipelines.stable_diffusion.convert_from_ckpt import convert_ldm_unet_checkpoint
    from safetensors import safe_open

    # ── SD 1.5 UNet 配置 ──
    SD15_UNET_CONFIG = {
        'act_fn': 'silu', 'addition_embed_type': None, 'addition_embed_type_num_heads': 64,
        'addition_time_embed_dim': None, 'attention_head_dim': 8, 'attention_type': 'default',
        'block_out_channels': [320, 640, 1280, 1280], 'center_input_sample': False,
        'class_embed_type': None, 'class_embeddings_concat': False, 'conv_in_kernel': 3,
        'conv_out_kernel': 3, 'cross_attention_dim': 768, 'cross_attention_norm': None,
        'down_block_types': ['CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'DownBlock2D'],
        'downsample_padding': 1, 'dropout': 0.0, 'dual_cross_attention': False,
        'encoder_hid_dim': None, 'encoder_hid_dim_type': None, 'flip_sin_to_cos': True,
        'freq_shift': 0, 'in_channels': 4, 'layers_per_block': 2,
        'mid_block_only_cross_attention': None, 'mid_block_scale_factor': 1,
        'mid_block_type': 'UNetMidBlock2DCrossAttn', 'norm_eps': 1e-05, 'norm_num_groups': 32,
        'num_attention_heads': None, 'num_class_embeds': None, 'only_cross_attention': False,
        'out_channels': 4, 'projection_class_embeddings_input_dim': None,
        'resnet_out_scale_factor': 1.0, 'resnet_skip_time_act': False,
        'resnet_time_scale_shift': 'default', 'reverse_transformer_layers_per_block': None,
        'sample_size': 64, 'time_cond_proj_dim': None, 'time_embedding_act_fn': None,
        'time_embedding_dim': None, 'time_embedding_type': 'positional',
        'timestep_post_act': None, 'transformer_layers_per_block': 1,
        'up_block_types': ['UpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D'],
        'upcast_attention': False, 'use_linear_projection': False,
    }

    # ── SD XL UNet 配置 ──
    SDXL_UNET_CONFIG = {
        'act_fn': 'silu', 'addition_embed_type': 'text_time',
        'addition_embed_type_num_heads': 64, 'addition_time_embed_dim': 256,
        'attention_head_dim': [5, 10, 20], 'attention_type': 'default',
        'block_out_channels': [320, 640, 1280], 'center_input_sample': False,
        'class_embed_type': None, 'class_embeddings_concat': False, 'conv_in_kernel': 3,
        'conv_out_kernel': 3, 'cross_attention_dim': 2048, 'cross_attention_norm': None,
        'down_block_types': ['DownBlock2D', 'CrossAttnDownBlock2D', 'CrossAttnDownBlock2D'],
        'downsample_padding': 1, 'dropout': 0.0, 'dual_cross_attention': False,
        'encoder_hid_dim': None, 'encoder_hid_dim_type': None, 'flip_sin_to_cos': True,
        'freq_shift': 0, 'in_channels': 4, 'layers_per_block': 2,
        'mid_block_only_cross_attention': None, 'mid_block_scale_factor': 1,
        'mid_block_type': 'UNetMidBlock2DCrossAttn', 'norm_eps': 1e-05, 'norm_num_groups': 32,
        'num_attention_heads': None, 'num_class_embeds': None, 'only_cross_attention': False,
        'out_channels': 4, 'projection_class_embeddings_input_dim': 2816,
        'resnet_out_scale_factor': 1.0, 'resnet_skip_time_act': False,
        'resnet_time_scale_shift': 'default', 'reverse_transformer_layers_per_block': None,
        'sample_size': 128, 'time_cond_proj_dim': None, 'time_embedding_act_fn': None,
        'time_embedding_dim': None, 'time_embedding_type': 'positional',
        'timestep_post_act': None, 'transformer_layers_per_block': [1, 2, 10],
        'up_block_types': ['CrossAttnUpBlock2D', 'CrossAttnUpBlock2D', 'UpBlock2D'],
        'upcast_attention': False, 'use_linear_projection': True,
    }

    class SimplePipeline:
        """最小 pipeline wrapper，提供 run_ddim_and_collect 需要的接口。

        使用 hash-based text embeddings 替代真实 text encoder。
        对 self-attention locality 测量来说足够 — 因为 spatial patterns 主要由位置决定，
        text condition 影响什么特征出现而非空间注意范围。
        """
        def __init__(self, unet, scheduler, device, B, resolution):
            self.unet = unet
            self.scheduler = scheduler
            self._device = device
            self._B = B
            self._resolution = resolution
            self._cross_attn_dim = unet.config.cross_attention_dim
            self._max_seq_len = 77
            # SDXL 需要 added_cond_kwargs（text_embeds + time_ids）
            self._is_sdxl = (getattr(unet.config, 'addition_embed_type', None) == 'text_time')
            self._added_cond_kwargs = None

        def encode_prompt(self, prompts, device, num_images_per_prompt,
                          do_classifier_free_guidance=False):
            """用 hash-based 确定性 embeddings 替代真实 text encoding。

            每个 prompt 得到唯一的、固定嵌入。不同 runs 可复现。
            """
            B = len(prompts)
            embeds = torch.zeros(B, self._max_seq_len, self._cross_attn_dim,
                                 device=device, dtype=self.unet.dtype)
            for i, prompt in enumerate(prompts):
                seed = int(hashlib.md5(prompt.encode()).hexdigest(), 16) % (2**31)
                g = torch.Generator(device=device).manual_seed(seed)
                embeds[i] = torch.randn(self._max_seq_len, self._cross_attn_dim,
                                        generator=g, device=device, dtype=self.unet.dtype) * 0.02

            if self._is_sdxl:
                # SDXL: 需要 pooled text embeddings [B, 1280] + time_ids [B, 6]
                pooled_dim = 1280  # SDXL pooled text embedding dim (OpenCLIP-G)
                pooled = torch.zeros(B, pooled_dim, device=device, dtype=self.unet.dtype)
                for i, prompt in enumerate(prompts):
                    seed = int(hashlib.md5((prompt + '_pooled').encode()).hexdigest(), 16) % (2**31)
                    g = torch.Generator(device=device).manual_seed(seed)
                    pooled[i] = torch.randn(pooled_dim, generator=g, device=device,
                                            dtype=self.unet.dtype) * 0.02
                # time_ids: [original_H, original_W, crop_top, crop_left, target_H, target_W]
                time_ids = torch.tensor(
                    [[self._resolution, self._resolution, 0, 0, self._resolution, self._resolution]],
                    device=device, dtype=self.unet.dtype
                ).repeat(B, 1)
                self._added_cond_kwargs = {'text_embeds': pooled, 'time_ids': time_ids}
                return embeds, pooled
            else:
                self._added_cond_kwargs = None
                return embeds, None

    print(f"\nLoading {args.model} UNet from safetensors (offline)...")

    # 加载 safetensors checkpoint
    ckpt = {}
    with safe_open(args.ckpt, framework='pt') as f:
        for key in f.keys():
            ckpt[key] = f.get_tensor(key)
    print(f"  Loaded {len(ckpt)} keys")

    # 选择配置
    if args.model == 'sd15':
        unet_cfg = SD15_UNET_CONFIG
    else:
        unet_cfg = SDXL_UNET_CONFIG

    # 转换 compvis → diffusers weights，创建 UNet
    converted_unet = convert_ldm_unet_checkpoint(ckpt, unet_cfg)
    unet = UNet2DConditionModel(**unet_cfg)
    unet.load_state_dict(converted_unet, strict=False)
    unet.to(device)
    unet.eval()
    print(f"  UNet params: {sum(p.numel() for p in unet.parameters()) / 1e6:.0f}M")

    # DDIM scheduler
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule='scaled_linear',
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )

    # 构建 minimal pipeline
    pipe = SimplePipeline(unet, scheduler, device, args.batch_size, args.resolution)

    # Patch attention
    patch_unet_attention(unet, args.model)

    # 固定 prompts（模拟 ImageNet 4 类）
    prompts = [
        "a photo of a golden retriever",
        "a photo of an otter",
        "a photo of an African elephant",
        "a photo of a jellyfish",
    ][:args.batch_size]

    # k-values per grid: 按分辨率设置合适范围
    k_values_per_grid = {
        32: [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31],
        16: [1, 3, 5, 7, 9, 11, 13, 15],
        8:  [1, 3, 5, 7],
    }

    print(f"\nRunning DDIM inference ({args.num_steps} steps × {args.batch_size} images)...")
    import time
    t0 = time.time()
    data = run_ddim_and_collect(
        pipe, prompts,
        num_steps=args.num_steps,
        k_values_per_grid=k_values_per_grid,
        device=device,
        seed=args.seed,
        resolution=args.resolution,
    )
    print(f"  Done in {time.time() - t0:.0f}s")

    analyze_and_print(data)

    torch.save(data, args.output)
    print(f"\nSaved → {args.output} ({os.path.getsize(args.output) / 1e6:.0f} MB)")
    print("Done!")


if __name__ == '__main__':
    main()
