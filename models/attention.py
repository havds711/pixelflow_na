"""
Attention variants for DiT: Full Attention and Neighborhood Attention (NA).

NA restricts each query token to attend only to keys within a k×k spatial window
centered at the query's 2D grid position, reducing complexity from O(N²) to O(N·k²).

Reference:
  - NAT (Hassani et al., CVPR 2023): NA definition
  - HDiT (Crowson et al., ICML 2024): NA in pixel-space diffusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_na_mask(h: int, w: int, kernel_size: int, device: torch.device) -> torch.Tensor:
    """
    Build a 2D neighborhood attention mask.

    For a grid of shape (h, w), each query at position (i, j) can only attend
    to keys within a kernel_size × kernel_size window centered at (i, j).
    Chebyshev distance (L∞) is used: max(|Δi|, |Δj|) ≤ kernel_size // 2.

    Args:
        h, w: grid height and width
        kernel_size: window size (must be odd)
        device: torch device

    Returns:
        mask: [h*w, h*w] — 0.0 for allowed positions, -inf for masked positions
    """
    N = h * w
    half = kernel_size // 2

    # 2D coordinates for each token
    rows = torch.arange(h, device=device)
    cols = torch.arange(w, device=device)
    grid_i, grid_j = torch.meshgrid(rows, cols, indexing='ij')
    coords = torch.stack([grid_i.reshape(-1), grid_j.reshape(-1)], dim=-1)  # [N, 2]

    # Pairwise Chebyshev distance
    diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # [N, N, 2]
    dist = diff.abs().max(dim=-1).values  # [N, N]

    mask = torch.where(dist <= half, 0.0, float('-inf'))
    return mask


class FullAttention(nn.Module):
    """Standard multi-head self-attention (all-to-all)."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, C] where N = H*W tokens
        Returns:
            [B, N, C]
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each [B, N, heads, head_dim]

        # [B, heads, N, head_dim]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, N, N]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v  # [B, heads, N, head_dim]
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class NeighborAttention(nn.Module):
    """
    Neighborhood Attention: each query attends only to keys within a k×k spatial window.

    Uses a mask-based implementation: computes full QK^T then applies spatial mask.
    Memory is O(N²) but for small grids (1024 tokens at 64×64 with patch=2) this is fine.
    A production implementation would use the NAT CUDA kernel for O(N·k²).

    Args:
        dim: token dimension
        num_heads: number of attention heads
        kernel_size: spatial window size (must be odd)
        dropout: attention dropout rate
    """

    def __init__(
        self, dim: int, num_heads: int = 8, kernel_size: int = 7, dropout: float = 0.0
    ):
        super().__init__()
        assert kernel_size % 2 == 1, f"kernel_size must be odd, got {kernel_size}"
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.kernel_size = kernel_size
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # Mask cache: keyed by (h, w)
        self._mask_cache: dict[tuple, torch.Tensor] = {}

    def _get_mask(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        key = (h, w, self.kernel_size)
        if key not in self._mask_cache:
            self._mask_cache[key] = _build_na_mask(h, w, self.kernel_size, device)
        mask = self._mask_cache[key]
        if mask.device != device:
            mask = mask.to(device)
            self._mask_cache[key] = mask
        return mask

    def forward(self, x: torch.Tensor, grid_h: int = None, grid_w: int = None) -> torch.Tensor:
        """
        Args:
            x: [B, N, C]
            grid_h, grid_w: spatial grid dimensions. If None, assumes square grid.
        Returns:
            [B, N, C]
        """
        B, N, C = x.shape
        if grid_h is None:
            grid_h = grid_w = int(N ** 0.5)
        if grid_w is None:
            grid_w = grid_h
        assert grid_h * grid_w == N, f"grid {grid_h}×{grid_w} ≠ {N} tokens"

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        q = q.permute(0, 2, 1, 3)  # [B, heads, N, head_dim]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, N, N]

        # Apply 2D spatial mask
        mask = self._get_mask(grid_h, grid_w, attn.device)  # [N, N]
        attn = attn + mask[None, None, :, :]

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)

    def extract_attention_weights(
        self, x: torch.Tensor, grid_h: int = None, grid_w: int = None
    ) -> torch.Tensor:
        """
        Extract raw attention weights (before softmax or after — returns post-softmax).
        Used for ERF and distance distribution measurements.

        Returns:
            attn_weights: [B, heads, N, N] — post-softmax attention probabilities
        """
        B, N, C = x.shape
        if grid_h is None:
            grid_h = grid_w = int(N ** 0.5)
        if grid_w is None:
            grid_w = grid_h

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = self._get_mask(grid_h, grid_w, attn.device)
        attn = attn + mask[None, None, :, :]
        attn = F.softmax(attn, dim=-1)
        return attn


def make_attention(
    attn_type: str,
    dim: int,
    num_heads: int = 8,
    kernel_size: int = 7,
    dropout: float = 0.0,
) -> nn.Module:
    """
    Factory for attention modules.

    Args:
        attn_type: "full" | "na"
        dim: token dimension
        num_heads: number of heads
        kernel_size: NA window size (ignored for "full")
        dropout: attention dropout
    """
    if attn_type == "full":
        return FullAttention(dim, num_heads, dropout)
    elif attn_type == "na":
        return NeighborAttention(dim, num_heads, kernel_size, dropout)
    else:
        raise ValueError(f"Unknown attention type: {attn_type}. Use 'full' or 'na'.")
