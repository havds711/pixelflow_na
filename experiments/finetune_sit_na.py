#!/usr/bin/env python3
"""
Fine-tune SiT-XL/2 with Neighborhood Attention on ImageNet (latent space).

Loads pretrained SiT-XL/2 → replaces all attention blocks with NA → fine-tune
with Flow Matching loss. Saves validation samples every N steps.

Usage:
  # Quick test: 500 steps with SiT-B/2 on 4080
  python finetune_sit_na.py --model SiT-B/2 --kernel_size 7 --max_steps 500 --device cuda:3

  # Full fine-tune: SiT-XL/2 on 3090
  python finetune_sit_na.py --model SiT-XL/2 --kernel_size 7 --max_steps 5000 --device cuda:2
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import io
import glob
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from collections import OrderedDict
from copy import deepcopy
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'SiT'))

from SiT.models import SiT_XL_2, SiT_B_2, SiT_S_2, SiTBlock
from SiT.download import find_model
from SiT.attention import make_attention


# ---------------------------------------------------------------------------
# VAE wrapper for latent encoding/decoding
# ---------------------------------------------------------------------------

class VAEWrapper:
    """Lazy-loaded VAE for encode/decode between pixel and latent space."""

    def __init__(self, vae_path: str = None, device: str = "cuda"):
        self.device = device
        self._vae = None
        self.vae_path = vae_path or os.path.join(os.path.dirname(__file__), '..', 'vae')

    @property
    def vae(self):
        if self._vae is None:
            from diffusers.models import AutoencoderKL
            self._vae = AutoencoderKL.from_pretrained(self.vae_path).to(self.device)
            self._vae.eval()
            for p in self._vae.parameters():
                p.requires_grad_(False)
        return self._vae

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """images: [B,3,256,256] in [0,1] → latent: [B,4,32,32]"""
        images = images * 2 - 1  # [0,1] → [-1,1]
        latent = self.vae.encode(images).latent_dist.sample().mul_(0.18215)
        return latent

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """latent: [B,4,32,32] → images: [B,3,256,256] in [0,1]"""
        latent = latent / 0.18215
        images = self.vae.decode(latent).sample
        images = (images + 1) / 2  # [-1,1] → [0,1]
        return images.clamp(0, 1)


# ---------------------------------------------------------------------------
# ImageNet Parquet → latent dataset (online encode or cached)
# ---------------------------------------------------------------------------

class ImageNetLatentDataset(Dataset):
    """Load ImageNet images, encode to latents on-the-fly via VAE."""

    def __init__(self, parquet_dir: str, vae: VAEWrapper, img_size: int = 256):
        self.vae = vae
        self.img_size = img_size

        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ])

        # Index parquet files
        train_files = sorted(glob.glob(os.path.join(parquet_dir, 'train-*.parquet')))
        if not train_files:
            raise FileNotFoundError(f"No train-*.parquet found in {parquet_dir}")
        print(f"Found {len(train_files)} parquet files")

        self._samples = []  # (file_idx, row_idx)
        self._labels = []
        self._label_set = set()
        self._tables = {}   # lazy-loaded parquet tables

        for file_idx, fpath in enumerate(train_files):
            meta = pq.read_metadata(fpath)
            n_rows = meta.num_rows
            table = pq.read_table(fpath, columns=['label'])
            file_labels = table['label'].to_pylist()
            for row_idx in range(n_rows):
                self._samples.append((file_idx, row_idx))
                self._labels.append(file_labels[row_idx])
                self._label_set.add(file_labels[row_idx])

        self._file_paths = train_files
        self.num_classes = len(self._label_set)
        print(f"  {len(self._samples):,} images, {self.num_classes} classes")

    def _load_image(self, file_idx: int, row_idx: int):
        if file_idx not in self._tables:
            self._tables[file_idx] = pq.read_table(self._file_paths[file_idx])

        table = self._tables[file_idx]
        img_data = table['image'][row_idx]
        if hasattr(img_data, 'as_py'):
            img_data = img_data.as_py()
        if isinstance(img_data, dict):
            img_bytes = img_data['bytes']
        else:
            img_bytes = img_data

        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        return self.transform(img)

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        file_idx, row_idx = self._samples[idx]
        label = self._labels[idx]

        img_tensor = self._load_image(file_idx, row_idx)
        latent = self.vae.encode(img_tensor.unsqueeze(0).to(self.vae.device)).squeeze(0).cpu()
        return latent, label


# ---------------------------------------------------------------------------
# Precomputed latent dataset (fast, no VAE in loop)
# ---------------------------------------------------------------------------

class PrecomputedLatentDataset(Dataset):
    """Load precomputed VAE latents from .pt shard files."""

    def __init__(self, latent_dir: str):
        self.latent_dir = latent_dir
        meta = torch.load(os.path.join(latent_dir, 'metadata.pt'), map_location='cpu',
                         weights_only=False)
        shard_files = sorted(glob.glob(os.path.join(latent_dir, 'latents_*.pt')))
        self._shards = shard_files
        self._loaded_shard = None
        self._loaded_shard_idx = -1
        self._items_per_shard = meta['shard_size']
        self.total = meta['total_images']
        print(f"  {self.total:,} precomputed latents in {len(shard_files)} shards")

    def __len__(self):
        return self.total

    def _load_shard(self, shard_idx):
        if self._loaded_shard_idx == shard_idx:
            return
        if shard_idx >= len(self._shards):
            return
        data = torch.load(self._shards[shard_idx], map_location='cpu', weights_only=False)
        self._loaded_shard = data['latents'], data['labels']
        self._loaded_shard_idx = shard_idx

    def __getitem__(self, idx):
        shard_idx = idx // self._items_per_shard
        local_idx = idx % self._items_per_shard
        self._load_shard(shard_idx)
        latents, labels = self._loaded_shard
        if local_idx >= len(labels):
            local_idx = len(labels) - 1
        return latents[local_idx].float(), int(labels[local_idx])


# ---------------------------------------------------------------------------
# NA model builder
# ---------------------------------------------------------------------------

def build_na_model(model_type: str, kernel_size: int, device: str):
    """Load pretrained SiT, replace attention with NA."""
    model_map = {
        'SiT-XL/2': lambda: SiT_XL_2(input_size=32, num_classes=1000, learn_sigma=True),
        'SiT-B/2': lambda: SiT_B_2(input_size=32, num_classes=1000, learn_sigma=True),
        'SiT-S/2': lambda: SiT_S_2(input_size=32, num_classes=1000, learn_sigma=True),
    }
    if model_type not in model_map:
        raise ValueError(f"Unknown model: {model_type}. Choose from {list(model_map.keys())}")

    model = model_map[model_type]().to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    # Load pretrained weights
    ckpt_path = os.path.join(os.path.dirname(__file__), '..', 'SiT/checkpoints/SiT-XL-2-256.pt')
    state_dict = find_model(ckpt_path)

    # Handle key mismatch: pretrained model has full attention, we're replacing with NA
    # → load all weights except attention qkv/proj
    model_state = model.state_dict()
    matched = {}
    skipped = []
    for k, v in state_dict.items():
        if k in model_state and v.shape == model_state[k].shape:
            matched[k] = v
        else:
            skipped.append(k)

    model.load_state_dict(matched, strict=False)
    print(f"Loaded {len(matched)}/{len(state_dict)} keys from pretrained checkpoint")
    if skipped:
        print(f"  Skipped (will train from scratch): {len(skipped)} keys")

    # Replace attention with NA
    for i, block in enumerate(model.blocks):
        block.attn = make_attention(
            'na', dim=block.attn.dim, num_heads=block.attn.num_heads,
            kernel_size=kernel_size, dilation=1,
        ).to(device)

    # Freeze all non-attention parameters to save VRAM
    # Only fine-tune: attention qkv/proj + final_layer + label embedder
    for name, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if 'attn' in name or 'final_layer' in name or 'y_embedder' in name:
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Model: {model_type}, params: {n_params:.0f}M, trainable: {trainable:.0f}M, NA k={kernel_size}")
    print(f"  Grid: {model.grid_size}x{model.grid_size} = {model.grid_size**2} tokens")

    return model


# ---------------------------------------------------------------------------
# Flow Matching loss
# ---------------------------------------------------------------------------

def flow_matching_loss(model, x_0, labels):
    """Linear interpolant: x_t = (1-t)·x_0 + t·x_1, target v = x_1 - x_0."""
    B = x_0.shape[0]
    x_1 = torch.randn_like(x_0)
    t = torch.rand(B, device=x_0.device)
    t_reshaped = t.view(-1, 1, 1, 1)
    x_t = (1 - t_reshaped) * x_0 + t_reshaped * x_1
    v_target = x_1 - x_0
    v_pred = model(x_t, t, labels)
    return F.mse_loss(v_pred, v_target)


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(decay).add_(p.data, alpha=1 - decay)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def finetune(
    model,
    train_loader,
    vae: VAEWrapper,
    steps: int = 5000,
    lr: float = 1e-5,
    ema_decay: float = 0.9999,
    sample_every: int = 500,
    save_every: int = 1000,
    log_every: int = 50,
    output_dir: str = "../outputs/sit_finetune_na",
    device: str = "cuda",
):
    os.makedirs(os.path.join(output_dir, 'samples'), exist_ok=True)

    # EMA
    ema = deepcopy(model)
    ema.eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    update_ema(ema, model, decay=0)

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0, betas=(0.9, 0.999))

    # Fixed labels for validation samples
    val_labels = torch.tensor([207, 360, 387, 974, 88, 979, 417, 279], device=device)
    n_val = len(val_labels)

    model.train()
    running_loss = 0.0
    log_steps = 0
    data_iter = iter(train_loader)

    pbar = tqdm(range(1, steps + 1), desc="Fine-tuning")
    for step in pbar:
        # Get batch (restart iterator if exhausted)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        x_0, labels = batch
        x_0 = x_0.to(device)
        labels = labels.to(device)

        # Flow matching loss
        loss = flow_matching_loss(model, x_0, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        update_ema(ema, model, ema_decay)

        running_loss += loss.item()
        log_steps += 1

        if step % log_every == 0:
            avg_loss = running_loss / log_steps
            pbar.set_postfix(loss=f"{avg_loss:.4f}")
            running_loss = 0.0
            log_steps = 0

        # Save validation samples
        if step % sample_every == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                # Generate via ODE sampling
                n = n_val
                z = torch.randn(n, model.in_channels, model.grid_size * 2, model.grid_size * 2,
                               device=device)
                dt = 1.0 / 20  # 20 ODE steps for quick visualization
                x = z
                for i_s in range(20):
                    t_current = 1.0 - i_s * dt
                    t_tensor = torch.full((n,), t_current, device=device)
                    v = ema(x, t_tensor, val_labels)
                    x = x + v * dt
                x = x.clamp(-3, 3)  # rough clamp for latent space

            # Decode to pixel space
            samples = vae.decode(x)
            from torchvision.utils import save_image
            save_path = os.path.join(output_dir, 'samples', f'step_{step:05d}.png')
            save_image(samples, save_path, nrow=4, normalize=False)
            print(f"\n  [Step {step}] Validation samples saved → {save_path}")
            model.train()

        # Save checkpoint
        if step % save_every == 0:
            ckpt_path = os.path.join(output_dir, f'checkpoint_step{step}.pt')
            torch.save({
                'step': step,
                'model': model.state_dict(),
                'ema': ema.state_dict(),
                'opt': opt.state_dict(),
            }, ckpt_path)
            print(f"  [Step {step}] Checkpoint saved → {ckpt_path}")

    # Final save
    final_path = os.path.join(output_dir, 'checkpoint_final.pt')
    torch.save({
        'step': steps,
        'model': model.state_dict(),
        'ema': ema.state_dict(),
        'opt': opt.state_dict(),
    }, final_path)
    print(f"\nFinal checkpoint → {final_path}")

    return model, ema


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune SiT with Neighborhood Attention")
    parser.add_argument('--model', type=str, default='SiT-B/2',
                        choices=['SiT-XL/2', 'SiT-B/2', 'SiT-S/2'])
    parser.add_argument('--kernel_size', type=int, default=7, help='NA kernel size (odd)')
    parser.add_argument('--max_steps', type=int, default=5000, help='Fine-tune steps')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--sample_every', type=int, default=500, help='Save samples every N steps')
    parser.add_argument('--save_every', type=int, default=1000, help='Save checkpoint every N steps')
    parser.add_argument('--data_dir', type=str,
                        default=os.path.expanduser('~/PixelDiT-vae/c2i/imagenet_parquet'))
    parser.add_argument('--latent_dir', type=str,
                        default=os.path.expanduser('~/PixelDiT-vae/c2i/imagenet_latents'),
                        help='Precomputed latents directory')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limit dataset size for quick testing')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda / cuda:0 / cuda:2 etc.)')
    return parser.parse_args()


def main():
    args = parse_args()

    device = args.device
    if 'cuda' in device and not torch.cuda.is_available():
        device = 'cpu'

    if args.output_dir is None:
        args.output_dir = f"../outputs/sit_finetune_na{args.kernel_size}"

    print("=" * 60)
    print(f"SiT NA Fine-tune — {args.model}, k={args.kernel_size}")
    print(f"Steps: {args.max_steps} | LR: {args.lr} | Batch: {args.batch_size}")
    print(f"Device: {device} | Output: {args.output_dir}")
    print("=" * 60)

    # VAE
    print("\nLoading VAE...")
    vae = VAEWrapper(device=device)

    # Model
    print("Building NA model...")
    model = build_na_model(args.model, args.kernel_size, device)

    # Data
    if args.latent_dir and os.path.exists(os.path.join(args.latent_dir, 'metadata.pt')):
        print("\nLoading precomputed latents...")
        dataset = PrecomputedLatentDataset(args.latent_dir)
        if args.max_samples is not None and args.max_samples < len(dataset):
            from torch.utils.data import Subset
            dataset = Subset(dataset, range(args.max_samples))
            print(f"  Limited to {args.max_samples} samples")
    else:
        print("\nLoading dataset (online VAE encode)...")
        dataset = ImageNetLatentDataset(args.data_dir, vae)
        if args.max_samples is not None and args.max_samples < len(dataset):
            from torch.utils.data import Subset
            dataset = Subset(dataset, range(args.max_samples))
            print(f"  Limited to {args.max_samples} samples")

    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    print(f"Batches per epoch: {len(train_loader)}")

    # Fine-tune
    print(f"\nStarting fine-tune ({args.max_steps} steps)...\n")
    model, ema = finetune(
        model, train_loader, vae,
        steps=args.max_steps,
        lr=args.lr,
        sample_every=args.sample_every,
        save_every=args.save_every,
        output_dir=args.output_dir,
        device=device,
    )

    print("\nDone!")


if __name__ == '__main__':
    main()
