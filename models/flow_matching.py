"""
Flow Matching training and sampling for pixel-space DiT.

Uses linear interpolant (Rectified Flow / SiT linear path):
  x_t = (1-t)·x_0 + t·x_1
  v_target = x_1 - x_0
  Loss = MSE(v_θ(x_t, t), v_target)

Reference:
  - SiT (Ma et al., ECCV 2024): arxiv:2401.08740
  - Flow Matching (Lipman et al., ICLR 2023): arxiv:2210.02747
  - Rectified Flow (Liu et al., ICLR 2023): arxiv:2209.03003
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional
from tqdm import tqdm
import math

from .dit import DiT, DiTConfig


class FlowMatchingTrainer:
    """
    Trainer for Flow Matching on pixel-space DiT.

    Usage:
        trainer = FlowMatchingTrainer(model, lr=1e-4)
        trainer.train(train_loader, epochs=200)
    """

    def __init__(
        self,
        model: DiT,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        ema_decay: float = 0.9999,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.device = device
        self.ema_decay = ema_decay

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999)
        )

        # EMA model for inference
        self.ema_model = None
        if ema_decay > 0:
            self.ema_model = DiT(model.config).to(device)
            self.ema_model.load_state_dict(model.state_dict())
            for p in self.ema_model.parameters():
                p.requires_grad_(False)

    def _update_ema(self):
        """Update exponential moving average of model weights."""
        if self.ema_model is None:
            return
        with torch.no_grad():
            for ema_p, p in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_p.data.mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)

    def _compute_loss(
        self, x_0: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Flow matching loss with linear interpolant.
        x_t = (1-t)·x_0 + t·x_1,  target v = x_1 - x_0
        """
        B = x_0.shape[0]
        # Sample noise and timesteps
        x_1 = torch.randn_like(x_0)
        t = torch.rand(B, device=x_0.device)

        # Interpolate
        t_reshaped = t.view(-1, 1, 1, 1)
        x_t = (1 - t_reshaped) * x_0 + t_reshaped * x_1
        v_target = x_1 - x_0

        # Predict velocity
        v_pred = self.model(x_t, t, labels)

        return F.mse_loss(v_pred, v_target)

    def train(
        self,
        train_loader: DataLoader,
        epochs: int = 200,
        log_every: int = 50,
        save_every: int = 50,
        save_path: str = "checkpoints/dit_flow.pt",
    ) -> list[float]:
        """
        Main training loop.

        Returns:
            loss_history: list of average losses per epoch
        """
        self.model.train()
        loss_history = []

        for epoch in range(epochs):
            epoch_losses = []
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

            for batch in pbar:
                if isinstance(batch, (tuple, list)):
                    x_0, labels = batch
                    x_0 = x_0.to(self.device)
                    labels = labels.to(self.device) if labels is not None else None
                else:
                    x_0 = batch.to(self.device)
                    labels = None

                loss = self._compute_loss(x_0, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self._update_ema()

                epoch_losses.append(loss.item())
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            loss_history.append(avg_loss)
            print(f"Epoch {epoch+1}: avg_loss = {avg_loss:.6f}")

            if (epoch + 1) % save_every == 0:
                self.save(save_path)

        return loss_history

    def save(self, path: str):
        """Save model and optimizer state."""
        checkpoint = {
            'model': self.model.state_dict(),
            'ema_model': self.ema_model.state_dict() if self.ema_model else None,
            'optimizer': self.optimizer.state_dict(),
            'config': self.model.config,
        }
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    def load(self, path: str):
        """Load model and optimizer state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model'])
        if self.ema_model and checkpoint.get('ema_model'):
            self.ema_model.load_state_dict(checkpoint['ema_model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Loaded checkpoint from {path}")


@torch.no_grad()
def sample_ode(
    model: DiT,
    n_samples: int,
    img_size: int = 64,
    steps: int = 20,
    labels: Optional[torch.Tensor] = None,
    cfg_scale: float = 1.0,
    device: str = "cuda",
    return_trajectory: bool = False,
) -> torch.Tensor:
    """
    Sample images using Euler ODE integration (reverse flow).

    Starting from noise, solve dx/dt = v_θ(x, t) backward from t=1 to t=0.

    Args:
        model: trained DiT model
        n_samples: number of images to generate
        img_size: image size
        steps: number of Euler steps
        labels: class labels [n_samples] (optional, for CFG)
        cfg_scale: classifier-free guidance scale (1.0 = no guidance)
        device: torch device
        return_trajectory: if True, return all intermediate states

    Returns:
        images: [n_samples, 3, img_size, img_size] in [0, 1] range
        or if return_trajectory: [steps+1, n_samples, 3, img_size, img_size]
    """
    model.eval()
    x = torch.randn(n_samples, 3, img_size, img_size, device=device)
    dt = 1.0 / steps

    trajectory = [x.clone()] if return_trajectory else None

    for i in range(steps):
        t_current = 1.0 - i * dt  # going backward: 1 → 0
        t = torch.full((n_samples,), t_current, device=device)

        if cfg_scale != 1.0 and labels is not None:
            # CFG: v = v_uncond + scale * (v_cond - v_uncond)
            v_uncond = model(x, t, None)
            v_cond = model(x, t, labels)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v = model(x, t, labels)

        x = x + v * dt

        if return_trajectory:
            trajectory.append(x.clone())

    # Clamp to valid pixel range
    x = x.clamp(0, 1)

    if return_trajectory:
        return torch.stack(trajectory)

    return x


def compute_fid_stats(
    model: DiT,
    n_samples: int,
    img_size: int,
    steps: int,
    batch_size: int = 64,
    labels: Optional[list[int]] = None,
    labels_per_class: Optional[int] = None,
    cfg_scale: float = 1.0,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Generate samples and compute Inception features for FID.

    Args:
        model: trained DiT model
        n_samples: total number of samples to generate
        img_size: image size
        steps: ODE sampling steps
        batch_size: generation batch size
        labels: list of class labels (for class-conditional generation)
        labels_per_class: generate this many per class (overrides n_samples)
        cfg_scale: CFG scale
        device: torch device

    Returns:
        features: [n_samples, 2048] InceptionV3 features
    """
    from torchvision.models import inception_v3
    import numpy as np

    # Setup InceptionV3
    inception = inception_v3(pretrained=True, transform_input=False).to(device)
    inception.eval()
    # Use the pool3 layer (2048-dim features, before final FC)
    inception.fc = nn.Identity()

    all_features = []
    model.eval()

    if labels_per_class is not None and labels is not None:
        # Class-balanced generation
        n_samples = labels_per_class * len(labels)
    elif labels is not None:
        labels = labels * ((n_samples + len(labels) - 1) // len(labels))
        labels = labels[:n_samples]

    n_batches = (n_samples + batch_size - 1) // batch_size

    for i in tqdm(range(n_batches), desc="Computing FID stats"):
        n = min(batch_size, n_samples - i * batch_size)
        batch_labels = labels[i*batch_size:i*batch_size+n] if labels is not None else None
        if batch_labels is not None:
            batch_labels = torch.tensor(batch_labels, device=device)

        images = sample_ode(
            model, n, img_size, steps, batch_labels, cfg_scale, device
        )

        # Resize to 299×299 for InceptionV3
        images_299 = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)

        with torch.no_grad():
            feats = inception(images_299)
        all_features.append(feats.cpu().numpy())

    return np.concatenate(all_features, axis=0)
