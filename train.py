#!/usr/bin/env python3
"""
Train a pixel-space DiT with Flow Matching.

Usage:
  # Tiny model, ImageNet-64 (parquet), quick test
  python train.py --model DiT_T --dataset imagenet_parquet \\
    --data_dir ~/PixelDiT-vae/c2i/imagenet_parquet --epochs 50 --batch_size 16

  # Small model, full attention baseline
  python train.py --model DiT_S --dataset imagenet_parquet \\
    --data_dir ~/PixelDiT-vae/c2i/imagenet_parquet --attn_type full --epochs 200

  # NA training
  python train.py --model DiT_S --dataset imagenet_parquet \\
    --data_dir ~/PixelDiT-vae/c2i/imagenet_parquet --attn_type na --na_kernel_size 7

  # CIFAR-10 (smoke test)
  python train.py --model DiT_T --dataset cifar10 --data_dir ~/PixelDiT/data --epochs 5
"""

import argparse
import torch
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from models.dit import DiT, DiTConfig, DiT_T, DiT_S, DiT_B, LabelEmbedder
from models.flow_matching import FlowMatchingTrainer
from models.attention import make_attention
from data.dataset import get_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Train pixel-space DiT with Flow Matching")

    # Model
    parser.add_argument('--model', type=str, default='DiT_T',
                        choices=['DiT_T', 'DiT_S', 'DiT_B'],
                        help='Model size preset')
    parser.add_argument('--img_size', type=int, default=64, help='Image size')
    parser.add_argument('--patch_size', type=int, default=2, help='Patch size')
    parser.add_argument('--dim', type=int, default=None, help='Override model dim')
    parser.add_argument('--depth', type=int, default=None, help='Override model depth')
    parser.add_argument('--heads', type=int, default=None, help='Override heads count')

    # Attention
    parser.add_argument('--attn_type', type=str, default='full',
                        choices=['full', 'na'], help='Attention type')
    parser.add_argument('--na_kernel_size', type=int, default=7,
                        help='NA kernel size (odd, e.g. 3,5,7,11,15)')
    parser.add_argument('--na_dilation', type=int, default=1,
                        help='NA dilation factor (1=contiguous, >1=sparse sampling)')

    # Data
    parser.add_argument('--dataset', type=str, default='imagenet_parquet',
                        choices=['imagenet_parquet', 'cifar10', 'imagenet64', 'imagefolder'],
                        help='Dataset name')
    parser.add_argument('--data_dir', type=str,
                        default=os.path.expanduser('~/PixelDiT-vae/c2i/imagenet_parquet'),
                        help='Dataset directory')
    parser.add_argument('--num_classes', type=int, default=None,
                        help='Number of classes (auto-detected if not specified)')

    # Training
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limit dataset size (for quick testing)')

    # Logging
    parser.add_argument('--log_every', type=int, default=50, help='Log every N batches')
    parser.add_argument('--save_every', type=int, default=50, help='Save checkpoint every N epochs')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Checkpoint directory')

    # Hardware
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cuda:0/cuda:1/cpu)')

    return parser.parse_args()


def build_model(args, num_classes: int) -> DiT:
    """Build DiT model from args."""
    if args.model == 'DiT_T':
        model_fn = DiT_T
    elif args.model == 'DiT_S':
        model_fn = DiT_S
    else:
        model_fn = DiT_B

    kwargs = dict(img_size=args.img_size, patch_size=args.patch_size)
    if args.dim is not None:
        kwargs['dim'] = args.dim
    if args.depth is not None:
        kwargs['depth'] = args.depth
    if args.heads is not None:
        kwargs['heads'] = args.heads

    model = model_fn(**kwargs)

    config = model.config
    config.attn_type = args.attn_type
    config.na_kernel_size = args.na_kernel_size
    config.na_dilation = args.na_dilation
    config.num_classes = num_classes
    config.use_cfg = True

    # Rebuild attention layers
    for block in model.blocks:
        block.attn = make_attention(
            config.attn_type, config.dim, config.heads,
            config.na_kernel_size, config.na_dilation,
        )

    # Rebuild label embedder for the correct number of classes
    model.label_embedder = LabelEmbedder(config.num_classes, config.dim)

    return model


def main():
    args = parse_args()

    device = args.device
    if 'cuda' in device and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'

    # Data (returns (loader, num_classes))
    train_loader, num_classes = get_dataloader(
        dataset_name=args.dataset,
        data_dir=os.path.expanduser(args.data_dir),
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
    )

    if args.num_classes is not None:
        num_classes = args.num_classes

    print("=" * 60)
    print(f"PixelFlow NA — Train {args.model} ({args.attn_type}, k={args.na_kernel_size}, "
          f"dilation={args.na_dilation})")
    print(f"Dataset: {args.dataset} | Classes: {num_classes} | "
          f"Image size: {args.img_size} | Patch: {args.patch_size}")
    print(f"Batch: {args.batch_size} | Epochs: {args.epochs} | LR: {args.lr}")
    print(f"Device: {device}")
    print("=" * 60)

    # Build model
    model = build_model(args, num_classes)
    n_params = model.get_num_trainable_params()
    print(f"Model parameters: {n_params / 1e6:.1f}M")
    print(f"Grid: {model.config.grid_size}×{model.config.grid_size} = {model.config.num_patches} tokens")
    print(f"Dataset size: {len(train_loader.dataset)}")

    # Trainer
    trainer = FlowMatchingTrainer(
        model, lr=args.lr, weight_decay=args.weight_decay, device=device,
    )

    # Save path
    os.makedirs(args.save_dir, exist_ok=True)
    save_name = f"dit_{args.model}_{args.attn_type}_k{args.na_kernel_size}_d{args.na_dilation}_{args.dataset}.pt"
    save_path = os.path.join(args.save_dir, save_name)

    # Train
    loss_history = trainer.train(
        train_loader, epochs=args.epochs, log_every=args.log_every,
        save_every=args.save_every, save_path=save_path,
    )

    trainer.save(save_path)
    print(f"\nTraining complete. Model saved to {save_path}")


if __name__ == '__main__':
    main()
