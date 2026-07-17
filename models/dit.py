"""
DiT (Diffusion Transformer) adapted for pixel-space Flow Matching.

Architecture follows DiT (Peebles & Xie, ICCV 2023) with adaLN-Zero modulation.
Key differences from the original:
  - Operates in pixel space (3×H×W) instead of latent space
  - Predicts velocity field v = x_1 - x_0 (flow matching) instead of noise ε
  - Supports both full attention and neighborhood attention (NA)

Reference:
  - DiT: "Scalable Diffusion Models with Transformers" (arxiv:2212.09748)
  - SiT: "Exploring Flow Matching for Diffusion Models" (arxiv:2401.08740)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Literal

from .attention import make_attention, FullAttention, NeighborAttention


@dataclass
class DiTConfig:
    """Configuration for DiT model."""
    # Image
    img_size: int = 64
    patch_size: int = 2
    in_channels: int = 3

    # Architecture
    dim: int = 384
    depth: int = 12
    heads: int = 6
    mlp_ratio: float = 4.0

    # Attention
    attn_type: Literal["full", "na"] = "full"
    na_kernel_size: int = 7
    attn_dropout: float = 0.0

    # Conditioning
    num_classes: int = 1000  # for class-conditional generation (ImageNet)
    use_cfg: bool = True     # classifier-free guidance during sampling
    cfg_drop_prob: float = 0.1  # probability of dropping class label during training

    @property
    def num_patches(self) -> int:
        return (self.img_size // self.patch_size) ** 2

    @property
    def grid_size(self) -> int:
        return self.img_size // self.patch_size


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Convert image to patch tokens."""
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.proj = nn.Conv2d(
            config.in_channels, config.dim,
            kernel_size=config.patch_size, stride=config.patch_size,
        )
        self.num_patches = config.num_patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W] → [B, dim, H/p, W/p] → [B, N, dim]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding → MLP."""
    def __init__(self, dim: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000):
        """Create sinusoidal timestep embeddings (same as DiT/SiT)."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
        ).to(t.device)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_emb)


class LabelEmbedder(nn.Module):
    """Class label embedding (for classifier-free guidance)."""
    def __init__(self, num_classes: int, dim: int, drop_prob: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.drop_prob = drop_prob
        self.embed = nn.Embedding(num_classes + 1, dim)  # +1 for "unconditional" token
        self.null_token = num_classes

    def forward(self, labels: torch.Tensor, force_drop: bool = False) -> torch.Tensor:
        if force_drop or (self.training and torch.rand(1).item() < self.drop_prob):
            labels = torch.full_like(labels, self.null_token)
        return self.embed(labels)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN modulation: scale * norm(x) + shift."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AdaLNZero(nn.Module):
    """
    Adaptive Layer Norm with zero-initialized modulation.
    Each DiT block has its own adaLN parameters.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, 6 * dim)  # 2×(shift, scale, gate)

        # Zero-initialize (important for training stability)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, c: torch.Tensor) -> tuple:
        """
        Args:
            c: conditioning vector [B, dim] (timestep + optionally label embedding)
        Returns:
            (shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp)
            each [B, dim]
        """
        params = self.linear(self.silu(c))
        return tuple(params.chunk(6, dim=-1))


class DiTBlock(nn.Module):
    """
    Single DiT transformer block with adaLN-Zero + attention + MLP.

    Operations:
      x = x + gate_attn * attn(modulate(ln1(x), shift_attn, scale_attn))
      x = x + gate_mlp * mlp(modulate(ln2(x), shift_mlp, scale_mlp))
    """

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.dim = config.dim
        self.grid_size = config.grid_size

        self.adaLN = AdaLNZero(config.dim)
        self.ln1 = nn.LayerNorm(config.dim, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(config.dim, elementwise_affine=False)

        self.attn = make_attention(
            attn_type=config.attn_type,
            dim=config.dim,
            num_heads=config.heads,
            kernel_size=config.na_kernel_size,
            dropout=config.attn_dropout,
        )

        self.mlp = nn.Sequential(
            nn.Linear(config.dim, int(config.dim * config.mlp_ratio)),
            nn.GELU(approximate='tanh'),
            nn.Linear(int(config.dim * config.mlp_ratio), config.dim),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.adaLN(c)

        # Attention sub-block
        x_norm = modulate(self.ln1(x), shift_attn, scale_attn)
        x = x + gate_attn.unsqueeze(1) * self.attn(x_norm)

        # MLP sub-block
        x_norm = modulate(self.ln2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)

        return x


class FinalLayer(nn.Module):
    """Final projection: tokens → pixel patches."""
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.ln = nn.LayerNorm(config.dim, elementwise_affine=False)
        self.adaLN = AdaLNZero(config.dim)
        self.linear = nn.Linear(config.dim, config.patch_size ** 2 * config.in_channels)
        # Zero-initialize final projection
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale, gate, _, _, _ = self.adaLN(c)
        x = modulate(self.ln(x), shift, scale)
        x = self.linear(x)
        return x


# ---------------------------------------------------------------------------
# Main DiT model
# ---------------------------------------------------------------------------

class DiT(nn.Module):
    """
    Diffusion Transformer for pixel-space Flow Matching.

    Predicts velocity field v(x, t) given noisy image x and timestep t.
    Architecture: patch embed → token sequence → N× DiTBlock(s) → final proj → un-patchify
    """

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config

        # Input
        self.patch_embed = PatchEmbed(config)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.num_patches, config.dim)
        )

        # Conditioning
        self.t_embedder = TimestepEmbedder(config.dim)
        self.label_embedder = LabelEmbedder(
            config.num_classes, config.dim, config.cfg_drop_prob
        ) if config.use_cfg else None

        # Transformer blocks
        self.blocks = nn.ModuleList([DiTBlock(config) for _ in range(config.depth)])

        # Output
        self.final_layer = FinalLayer(config)

        self._init_weights()

    def _init_weights(self):
        # pos_embed init
        nn.init.normal_(self.pos_embed, std=0.02)
        # Patch embed
        nn.init.xavier_uniform_(self.patch_embed.proj.weight.flatten(1))

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: noisy pixel image [B, 3, H, W]
            t: timestep [B] in [0, 1]
            labels: class labels [B] or None

        Returns:
            velocity prediction [B, 3, H, W]
        """
        B, C, H, W = x.shape
        N = self.config.num_patches

        # Patchify
        tokens = self.patch_embed(x)  # [B, N, dim]
        tokens = tokens + self.pos_embed[:, :N, :]

        # Conditioning
        t_emb = self.t_embedder(t)  # [B, dim]
        if self.label_embedder is not None and labels is not None:
            c = t_emb + self.label_embedder(labels)
        else:
            c = t_emb
        if labels is None and self.label_embedder is not None:
            # Null conditioning for CFG
            null_labels = torch.full(
                (B,), self.label_embedder.null_token, device=x.device, dtype=torch.long
            )
            c = t_emb + self.label_embedder(null_labels)

        # Transformer blocks
        for block in self.blocks:
            tokens = block(tokens, c)

        # Final projection
        tokens = self.final_layer(tokens, c)  # [B, N, patch² * 3]

        # Un-patchify
        tokens = tokens.reshape(
            B, H // self.config.patch_size, W // self.config.patch_size,
            self.config.patch_size, self.config.patch_size, C
        )
        tokens = tokens.permute(0, 5, 1, 3, 2, 4).contiguous()
        out = tokens.reshape(B, C, H, W)

        return out

    def get_attention_weights(self, x: torch.Tensor, t: torch.Tensor) -> list[torch.Tensor]:
        """
        Hook-based extraction of attention weights from all layers.
        Used for ERF and distance distribution measurements.

        Args:
            x: input image [B, 3, H, W]
            t: timestep [B]

        Returns:
            List of attention weight tensors, one per layer.
            Each: [B, heads, N, N] (post-softmax probabilities).
            Empty entries for blocks that don't support extraction.
        """
        B, C, H, W = x.shape
        N = self.config.num_patches

        tokens = self.patch_embed(x)
        tokens = tokens + self.pos_embed[:, :N, :]

        t_emb = self.t_embedder(t)
        if self.label_embedder is not None:
            null_labels = torch.full(
                (B,), self.label_embedder.null_token, device=x.device, dtype=torch.long
            )
            c = t_emb + self.label_embedder(null_labels)
        else:
            c = t_emb

        all_attn = []
        for block in self.blocks:
            # Get attention weights before the block processes (from input to this block)
            shift_attn, scale_attn, gate_attn, _, _, _ = block.adaLN(c)
            x_norm = modulate(block.ln1(tokens), shift_attn, scale_attn)

            if hasattr(block.attn, 'extract_attention_weights'):
                attn_weights = block.attn.extract_attention_weights(x_norm)
                all_attn.append(attn_weights)
            else:
                all_attn.append(None)

            # Actually run the block to get the next layer's input
            tokens = block(tokens, c)

        return all_attn

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def DiT_T(**kwargs) -> DiT:
    """DiT-Tiny: dim=192, depth=6, heads=3 (~21M params)."""
    defaults = dict(img_size=64, patch_size=2, dim=192, depth=6, heads=3)
    defaults.update(kwargs)
    return DiT(DiTConfig(**defaults))

def DiT_S(**kwargs) -> DiT:
    """DiT-Small: dim=384, depth=12, heads=6 (~127M params)."""
    defaults = dict(img_size=64, patch_size=2, dim=384, depth=12, heads=6)
    defaults.update(kwargs)
    return DiT(DiTConfig(**defaults))

def DiT_B(**kwargs) -> DiT:
    """DiT-Base: dim=768, depth=12, heads=12 (~460M params)."""
    defaults = dict(img_size=64, patch_size=2, dim=768, depth=12, heads=12)
    defaults.update(kwargs)
    return DiT(DiTConfig(**defaults))
