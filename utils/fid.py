"""
FID computation utilities using torchvision InceptionV3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import linalg
from torch.utils.data import DataLoader
from typing import Optional
from tqdm import tqdm


def get_inception_features(
    images: torch.Tensor,
    model: nn.Module,
    batch_size: int = 64,
    device: str = "cuda",
) -> np.ndarray:
    """
    Extract InceptionV3 features (pool3, 2048-dim).

    Args:
        images: [N, 3, H, W] in [0, 1] range
        model: InceptionV3 with fc=Identity
        batch_size: processing batch size
        device: torch device
    """
    all_feats = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size].to(device)
        # Inception expects [-1, 1]
        batch = batch * 2 - 1
        batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
        with torch.no_grad():
            feats = model(batch)
        all_feats.append(feats.cpu().numpy())
    return np.concatenate(all_feats, axis=0)


def compute_fid(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Compute Fréchet Inception Distance between two feature sets.

    Args:
        real_features: [N_real, D] Inception features
        fake_features: [N_fake, D] Inception features
        eps: numerical stability
    """
    mu_real = real_features.mean(axis=0)
    sigma_real = np.cov(real_features, rowvar=False)

    mu_fake = fake_features.mean(axis=0)
    sigma_fake = np.cov(fake_features, rowvar=False)

    diff = mu_real - mu_fake

    # sqrtm of product
    covmean = linalg.sqrtm(sigma_real @ sigma_fake, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_real) + np.trace(sigma_fake) - 2 * np.trace(covmean)
    return float(np.maximum(fid, 0.0))
