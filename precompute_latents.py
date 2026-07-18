#!/usr/bin/env python3
"""
Precompute ImageNet latents for fast SiT training.

Encodes all images through VAE once, saves as .pt files.
160K images → ~1.3 GB on disk (fp16, 4×32×32 latents).

Usage:
  python precompute_latents.py --device cuda:0 --max_images 20000
"""

import argparse
import torch
import os
import sys
import io
import glob
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from diffusers.models import AutoencoderKL

sys.path.insert(0, os.path.dirname(__file__))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default=os.path.expanduser('~/PixelDiT-vae/c2i/imagenet_parquet'))
    parser.add_argument('--output_dir', type=str,
                        default=os.path.expanduser('~/PixelDiT-vae/c2i/imagenet_latents'))
    parser.add_argument('--vae_path', type=str, default='vae')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Limit number of images (default: all)')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


def main():
    args = parse_args()

    device = args.device
    if 'cuda' in device and not torch.cuda.is_available():
        device = 'cpu'

    os.makedirs(args.output_dir, exist_ok=True)

    # Load VAE
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    # Load parquet metadata
    import pyarrow.parquet as pq
    train_files = sorted(glob.glob(os.path.join(args.data_dir, 'train-*.parquet')))
    print(f"Found {len(train_files)} parquet files")

    samples = []
    labels_list = []
    for file_idx, fpath in enumerate(train_files):
        meta = pq.read_metadata(fpath)
        n_rows = meta.num_rows
        table = pq.read_table(fpath, columns=['label'])
        file_labels = table['label'].to_pylist()
        for row_idx in range(n_rows):
            samples.append((file_idx, row_idx))
            labels_list.append(file_labels[row_idx])

    if args.max_images is not None:
        samples = samples[:args.max_images]
        labels_list = labels_list[:args.max_images]

    print(f"Total images: {len(samples):,}")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])

    # Precompute in batches
    all_latents = []
    all_labels = []
    tables_cache = {}  # file_idx → parquet table

    n_batches = (len(samples) + args.batch_size - 1) // args.batch_size
    for batch_idx in tqdm(range(n_batches), desc="Precomputing latents"):
        start = batch_idx * args.batch_size
        end = min(start + args.batch_size, len(samples))
        batch_samples = samples[start:end]

        images = []
        for file_idx, row_idx in batch_samples:
            if file_idx not in tables_cache:
                tables_cache[file_idx] = pq.read_table(train_files[file_idx])
            table = tables_cache[file_idx]
            img_data = table['image'][row_idx]
            if hasattr(img_data, 'as_py'):
                img_data = img_data.as_py()
            if isinstance(img_data, dict):
                img_bytes = img_data['bytes']
            else:
                img_bytes = img_data
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            images.append(transform(img))

        images = torch.stack(images).to(device)
        batch_labels = labels_list[start:end]

        with torch.no_grad():
            images_norm = images * 2 - 1  # [0,1] → [-1,1]
            latent = vae.encode(images_norm).latent_dist.sample().mul_(0.18215)
            all_latents.append(latent.half().cpu())
            all_labels.extend(batch_labels)

    # Save as sharded .pt files
    shard_size = 5000
    latents_cat = torch.cat(all_latents, dim=0)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)

    n_shards = (len(all_labels) + shard_size - 1) // shard_size
    print(f"Saving {len(all_labels):,} latents in {n_shards} shards...")

    for i in tqdm(range(n_shards), desc="Saving shards"):
        s = i * shard_size
        e = min(s + shard_size, len(all_labels))
        torch.save({
            'latents': latents_cat[s:e],
            'labels': labels_tensor[s:e],
        }, os.path.join(args.output_dir, f'latents_{i:04d}.pt'))

    # Save metadata
    torch.save({
        'total_images': len(all_labels),
        'shard_size': shard_size,
        'n_shards': n_shards,
        'latent_shape': list(latents_cat.shape[1:]),
    }, os.path.join(args.output_dir, 'metadata.pt'))

    size_gb = sum(os.path.getsize(os.path.join(args.output_dir, f))
                  for f in os.listdir(args.output_dir)) / 1e9
    print(f"\nDone! {len(all_labels):,} latents → {args.output_dir} ({size_gb:.1f} GB)")


if __name__ == '__main__':
    main()
