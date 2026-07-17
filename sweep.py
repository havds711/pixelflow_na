#!/usr/bin/env python3
"""
Kernel size sweep: train & evaluate NA models with different kernel sizes.

Runs the full experiment matrix:
  attention types: full | NA(k=3,5,7,11,15)
  metrics: FID, GFLOPs, ERF (per-t), distance distribution (per-t)

Usage:
  # Full sweep (train all variants)
  python sweep.py --dataset imagenet_parquet \\
    --data_dir ~/PixelDiT-vae/c2i/imagenet_parquet --model DiT_T --epochs 100

  # Quick sweep: only measure (use existing checkpoints)
  python sweep.py --dataset imagenet_parquet --model DiT_T --measure_only \\
    --ckpt_dir checkpoints/sweep/
"""

import argparse
import torch
import os
import sys
import json
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from models.dit import DiT, DiTConfig, DiT_T, DiT_S, LabelEmbedder
from models.attention import make_attention
from models.flow_matching import FlowMatchingTrainer, sample_ode
from data.dataset import get_dataloader
from measure import measure_erf, measure_distance_distribution, load_model


SWEEP_KERNELS = {
    'full': None,
    'na3': 3,
    'na5': 5,
    'na7': 7,
    'na11': 11,
    'na15': 15,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Kernel size sweep experiment")

    parser.add_argument('--dataset', type=str, default='imagenet_parquet')
    parser.add_argument('--data_dir', type=str,
                        default=os.path.expanduser('~/PixelDiT-vae/c2i/imagenet_parquet'))
    parser.add_argument('--model', type=str, default='DiT_T', choices=['DiT_T', 'DiT_S'])
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--patch_size', type=int, default=2)
    parser.add_argument('--num_classes', type=int, default=None,
                        help='Num classes (auto-detected if not set)')

    parser.add_argument('--epochs', type=int, default=100, help='Training epochs per variant')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--kernels', type=str, nargs='+',
                        default=['full', 'na3', 'na5', 'na7', 'na11', 'na15'],
                        help='Kernel sizes to sweep')

    parser.add_argument('--measure_only', action='store_true',
                        help='Skip training, only measure existing checkpoints')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints/sweep',
                        help='Checkpoint directory for sweep')

    parser.add_argument('--n_measure_samples', type=int, default=64,
                        help='Samples for ERF/distance measurement')
    parser.add_argument('--n_fid_samples', type=int, default=5000,
                        help='Samples for FID computation')

    parser.add_argument('--output', type=str, default='outputs/sweep_results.json')
    parser.add_argument('--device', type=str, default='cuda')

    return parser.parse_args()


def count_gflops(model: DiT, attn_type: str, kernel_size: int) -> float:
    N = model.config.num_patches
    d = model.config.dim
    depth = model.config.depth
    heads = model.config.heads
    head_dim = d // heads

    gflops_qkv = 3 * N * d * (3 * d) / 1e9

    if attn_type == 'full':
        gflops_attn = 2 * heads * N * N * head_dim / 1e9
    else:
        k = kernel_size
        gflops_attn = 2 * heads * N * (k * k) * head_dim / 1e9

    gflops_proj = N * d * d / 1e9
    mlp_ratio = 4.0
    gflops_mlp = 2 * N * d * (mlp_ratio * d) / 1e9

    return gflops_qkv + depth * (gflops_attn + gflops_proj + gflops_mlp)


def compute_fid_fast(
    model, real_loader, n_fake=5000, img_size=64, steps=20, batch_size=64,
    cfg_scale=1.0, device="cuda",
) -> float:
    from torchvision.models import inception_v3
    from scipy import linalg

    inception = inception_v3(pretrained=True, transform_input=False).to(device)
    inception.eval()
    inception.fc = torch.nn.Identity()

    def get_features(images):
        images = torch.nn.functional.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        images = images * 2 - 1
        with torch.no_grad():
            return inception(images).cpu().numpy()

    real_features = []
    n_collected = 0
    for batch in tqdm(real_loader, desc="Real features"):
        imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
        imgs = imgs.to(device)
        real_features.append(get_features(imgs))
        n_collected += imgs.shape[0]
        if n_collected >= n_fake:
            break
    real_features = np.concatenate(real_features, axis=0)[:n_fake]

    fake_features = []
    n_batches = (n_fake + batch_size - 1) // batch_size
    for i in tqdm(range(n_batches), desc="Fake features"):
        n = min(batch_size, n_fake - i * batch_size)
        fake = sample_ode(model, n, img_size, steps, None, cfg_scale, device)
        fake_features.append(get_features(fake))
    fake_features = np.concatenate(fake_features, axis=0)

    mu_r, sig_r = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu_f, sig_f = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)

    diff = mu_r - mu_f
    covmean = np.real(linalg.sqrtm(sig_r @ sig_f)).trace()
    fid = diff @ diff + np.trace(sig_r) + np.trace(sig_f) - 2 * covmean
    return float(max(fid, 0.0))


def main():
    args = parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    device = args.device
    if 'cuda' in device and not torch.cuda.is_available():
        device = 'cpu'

    # Data loaders
    real_loader, num_classes = get_dataloader(
        args.dataset, os.path.expanduser(args.data_dir),
        args.img_size, args.batch_size, num_workers=2, train=False,
    )
    if args.num_classes is not None:
        num_classes = args.num_classes

    print(f"Dataset: {args.dataset}, {num_classes} classes")
    print(f"Kernels: {args.kernels}")
    print(f"Device: {device}")

    results = {}

    for name in args.kernels:
        attn_type = 'na' if name.startswith('na') else 'full'
        kernel_size = SWEEP_KERNELS[name]

        print(f"\n{'='*60}")
        print(f"Running: {name} (attn={attn_type}, k={kernel_size})")
        print(f"{'='*60}")

        ckpt_path = os.path.join(args.ckpt_dir, f"dit_{name}.pt")

        if not args.measure_only:
            if args.model == 'DiT_T':
                model = DiT_T(img_size=args.img_size, patch_size=args.patch_size,
                              num_classes=num_classes)
            else:
                model = DiT_S(img_size=args.img_size, patch_size=args.patch_size,
                              num_classes=num_classes)

            config = model.config
            config.attn_type = attn_type
            config.na_kernel_size = kernel_size or 7

            for block in model.blocks:
                block.attn = make_attention(attn_type, block.dim, block.attn.num_heads,
                                            kernel_size or 7)
            model.label_embedder = LabelEmbedder(num_classes, config.dim)

            train_loader, _ = get_dataloader(
                args.dataset, os.path.expanduser(args.data_dir),
                args.img_size, args.batch_size, num_workers=4, train=True,
            )

            trainer = FlowMatchingTrainer(model, lr=args.lr, device=device)
            trainer.train(train_loader, epochs=args.epochs, log_every=20,
                          save_every=args.epochs, save_path=ckpt_path)
            eval_model = trainer.ema_model if trainer.ema_model else model
        else:
            eval_model = load_model(ckpt_path, args.model, attn_type,
                                    kernel_size or 7, num_classes,
                                    args.img_size, device)

        # --- Measurements ---
        print("  Measuring ERF...")
        erf = measure_erf(eval_model, n_samples=args.n_measure_samples,
                          img_size=args.img_size, device=device)

        print("  Measuring Distance Distribution...")
        dist = measure_distance_distribution(eval_model, n_samples=args.n_measure_samples,
                                             img_size=args.img_size, device=device)

        gflops = count_gflops(eval_model, attn_type, kernel_size or 7)

        print("  Computing FID...")
        fid = compute_fid_fast(eval_model, real_loader, n_fake=args.n_fid_samples,
                                img_size=args.img_size, steps=20, batch_size=args.batch_size,
                                device=device)

        results[name] = {
            'attn_type': attn_type,
            'kernel_size': kernel_size,
            'gflops': gflops,
            'fid': fid,
            'erf_mean': erf['mean_erf'],
            'per_t_erf': erf.get('per_t_erf', {}),
            'erf_per_layer': {str(k): v['mean'] for k, v in erf['layer_erf'].items()},
            'distance_mean': dist['mean_distance'],
            'distance_p99': dist['p99_distance'],
            'distance_p95': dist['p95_distance'],
            'per_t_distance': dist.get('per_t_distance', {}),
        }

        print(f"\n  {name}: FID={fid:.2f} GFLOPs={gflops:.2f} ERF={erf['mean_erf']:.2f} "
              f"Dist: mean={dist['mean_distance']:.2f} P99={dist['p99_distance']:.2f}")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {args.output}")

    # Summary
    print(f"\n{'='*80}")
    print(f"{'Variant':<10} {'FID':>8} {'GFLOPs':>8} {'ERF':>8} {'Dist Mean':>10} {'Dist P99':>10}")
    print(f"{'-'*60}")
    for name in args.kernels:
        r = results[name]
        print(f"{name:<10} {r['fid']:>8.2f} {r['gflops']:>8.2f} {r['erf_mean']:>8.2f} "
              f"{r['distance_mean']:>10.2f} {r['distance_p99']:>10.2f}")


if __name__ == '__main__':
    main()
