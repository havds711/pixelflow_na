"""
Attention variants for DiT: Full Attention and Neighborhood Attention (NA).

NA restricts each query token to attend only to keys within a k×k spatial window
centered at the query's 2D grid position, reducing complexity from O(N²) to O(N·k²).

Backend priority:
  1. NATTEN CUDA kernel (if available) — true O(N·k²) memory + speed
  2. PyTorch online-softmax fallback — O(N·k²) memory, slower but no compile needed

Reference:
  - NAT (Hassani et al., CVPR 2023): NA definition
  - HDiT (Crowson et al., ICML 2024): NA in pixel-space diffusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# NATTEN availability check
# ---------------------------------------------------------------------------
_NATTEN_AVAILABLE = False
try:
    import natten
    _NATTEN_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# NA implementation: NATTEN (preferred) + PyTorch online-softmax (fallback)
# ---------------------------------------------------------------------------

def _na_natten(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kernel_size: int, dilation: int = 1
) -> torch.Tensor:
    """
    NATTEN CUDA neighborhood attention.
    q,k,v: [B, nH, N, hD]
    Returns: [B, nH, N, hD]
    """
    B, nH, N, hD = q.shape
    H = W = int(N ** 0.5)
    scale = hD ** -0.5

    q_s = q.reshape(B, nH, H, W, hD)
    k_s = k.reshape(B, nH, H, W, hD)
    v_s = v.reshape(B, nH, H, W, hD)

    fn = natten.functional
    attn = fn.na2d_qk(q_s * scale, k_s, kernel_size=kernel_size, dilation=dilation)
    attn = attn.softmax(dim=-1)
    out = fn.na2d_av(attn, v_s, kernel_size=kernel_size, dilation=dilation)
    return out.reshape(B, nH, N, hD)


def _na_torch_online(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kernel_size: int, dilation: int = 1
) -> torch.Tensor:
    """
    Pure PyTorch NA using online-softmax (O(N·k²) memory, boundary-safe).
    Adapted from PixelDiT-vae/pixdit_core/neighbor_attn.py.
    """
    B, nH, N, hD = q.shape
    H = W = int(N ** 0.5)
    d = dilation
    half = kernel_size // 2
    p = half * d
    scale = hD ** -0.5
    C = B * nH
    NEG = -1e4

    # Reshape to spatial: [B*nH, hD, H, W]
    def to_spatial(x):
        return x.transpose(2, 3).reshape(C, hD, H, W).float()

    q_s = to_spatial(q)
    k_p = F.pad(to_spatial(k), (p, p, p, p))
    v_p = F.pad(to_spatial(v), (p, p, p, p))
    valid = F.pad(torch.ones(1, 1, H, W, device=q.device), (p, p, p, p))

    m = q_s.new_full((C, 1, H, W), NEG)
    l = q_s.new_zeros(C, 1, H, W)
    acc = q_s.new_zeros(C, hD, H, W)

    for a in range(kernel_size):
        ys = a * d
        for b in range(kernel_size):
            xs = b * d
            k_ab = k_p[:, :, ys:ys + H, xs:xs + W]
            v_ab = v_p[:, :, ys:ys + H, xs:xs + W]
            va = valid[:, :, ys:ys + H, xs:xs + W]

            logit = (q_s * k_ab).sum(dim=1, keepdim=True) * scale
            logit = torch.where(va > 0, logit, torch.full_like(logit, NEG))

            new_m = torch.maximum(m, logit)
            alpha = torch.exp(m - new_m)
            beta = torch.exp(logit - new_m)

            l = l * alpha + beta
            acc = acc * alpha + beta * v_ab
            m = new_m

    out = (acc / l).reshape(B, nH, hD, N).transpose(2, 3)
    return out.to(q.dtype)


def neighbor_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    kernel_size: int, dilation: int = 1,
) -> torch.Tensor:
    """
    Unified NA interface. Uses NATTEN if available, falls back to PyTorch.

    Args:
        q, k, v: [B, nH, N, hD] tensors
        kernel_size: spatial window size (odd)
        dilation: dilation factor (1 = contiguous window)

    Returns:
        [B, nH, N, hD]
    """
    if _NATTEN_AVAILABLE:
        try:
            return _na_natten(q, k, v, kernel_size, dilation)
        except Exception:
            # Fall through to PyTorch
            pass
    return _na_torch_online(q, k, v, kernel_size, dilation)


# ---------------------------------------------------------------------------
# NA mask (used ONLY for measurement — attention weight extraction)
# ---------------------------------------------------------------------------

def _build_na_mask(h: int, w: int, kernel_size: int, device: torch.device) -> torch.Tensor:
    """
    Build a 2D neighborhood attention mask (Chebyshev distance).

    Used for extract_attention_weights() — measurement path only.
    Not used during training/sampling.
    """
    N = h * w
    half = kernel_size // 2
    rows = torch.arange(h, device=device)
    cols = torch.arange(w, device=device)
    grid_i, grid_j = torch.meshgrid(rows, cols, indexing='ij')
    coords = torch.stack([grid_i.reshape(-1), grid_j.reshape(-1)], dim=-1)
    diff = coords.unsqueeze(1) - coords.unsqueeze(0)
    dist = diff.abs().max(dim=-1).values
    return torch.where(dist <= half, 0.0, float('-inf'))


# ---------------------------------------------------------------------------
# Attention modules
# ---------------------------------------------------------------------------

class FullAttention(nn.Module):
    """Standard multi-head self-attention (all-to-all)."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)

    def extract_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Extract post-softmax attention weights [B, heads, N, N]."""
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, _ = qkv.unbind(dim=2)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        return F.softmax(attn, dim=-1)


class NeighborAttention(nn.Module):
    """
    Neighborhood Attention with NATTEN backend + dilation support.

    Training/sampling uses NATTEN (or PyTorch online-softmax) — O(N·k²) memory.
    Measurement (extract_attention_weights) falls back to mask-based path
    since we need the full attention matrix for ERF/distance analysis.

    Args:
        dim: token dimension
        num_heads: number of attention heads
        kernel_size: spatial window size (odd)
        dilation: dilation factor for sparse sampling (1 = contiguous)
        dropout: attention dropout (only used in mask-based measurement path)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        kernel_size: int = 7,
        dilation: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, f"kernel_size must be odd, got {kernel_size}"
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # Mask cache for measurement path only
        self._mask_cache: dict[tuple, torch.Tensor] = {}

    def forward(
        self, x: torch.Tensor, grid_h: int = None, grid_w: int = None
    ) -> torch.Tensor:
        """
        Forward pass using NATTEN/online-softmax (O(N·k²)).

        Args:
            x: [B, N, C]
            grid_h, grid_w: spatial grid dims (inferred if None)
        Returns:
            [B, N, C]
        """
        B, N, C = x.shape
        if grid_h is None:
            grid_h = grid_w = int(N ** 0.5)

        # Project QKV
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # [B, N, nH, hD]
        q = q.permute(0, 2, 1, 3).contiguous()  # [B, nH, N, hD]
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()

        # NA forward
        out = neighbor_attention(q, k, v, self.kernel_size, self.dilation)

        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)

    def extract_attention_weights(
        self, x: torch.Tensor, grid_h: int = None, grid_w: int = None
    ) -> torch.Tensor:
        """
        Extract post-softmax attention weights for measurement.

        Uses mask-based O(N²) path since NATTEN doesn't expose attention weights.
        For N=1024 tokens, this is acceptable for measurement only.

        Returns:
            [B, heads, N, N] post-softmax attention probabilities
        """
        B, N, C = x.shape
        if grid_h is None:
            grid_h = grid_w = int(N ** 0.5)

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, _ = qkv.unbind(dim=2)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply dilated NA mask
        effective_ks = (self.kernel_size - 1) * self.dilation + 1
        mask = self._get_mask(grid_h, grid_w, effective_ks, attn.device)
        attn = attn + mask[None, None, :, :]

        return F.softmax(attn, dim=-1)

    def _get_mask(self, h: int, w: int, kernel_size: int, device: torch.device) -> torch.Tensor:
        key = (h, w, kernel_size)
        if key not in self._mask_cache:
            self._mask_cache[key] = _build_na_mask(h, w, kernel_size, device)
        mask = self._mask_cache[key]
        if mask.device != device:
            mask = mask.to(device)
            self._mask_cache[key] = mask
        return mask


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_attention(
    attn_type: str,
    dim: int,
    num_heads: int = 8,
    kernel_size: int = 7,
    dilation: int = 1,
    dropout: float = 0.0,
) -> nn.Module:
    """
    Factory for attention modules.

    Args:
        attn_type: "full" | "na"
        dim: token dimension
        num_heads: number of heads
        kernel_size: NA window size (ignored for "full")
        dilation: NA dilation factor (ignored for "full")
        dropout: attention dropout
    """
    if attn_type == "full":
        return FullAttention(dim, num_heads, dropout)
    elif attn_type == "na":
        return NeighborAttention(dim, num_heads, kernel_size, dilation, dropout)
    else:
        raise ValueError(f"Unknown attention type: {attn_type}. Use 'full' or 'na'.")
