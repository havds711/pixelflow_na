"""
Data loading for pixel-space Flow Matching training.

Supports:
  - ImageNet (via torchvision ImageFolder, 64×64)
  - Any image directory
  - CIFAR-10 (resized)
"""

import os
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from typing import Optional, Tuple


def get_default_transforms(img_size: int = 64, augment: bool = True):
    """Default image transforms for pixel-space generation."""
    if augment:
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # scales to [0, 1]
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])
    return transform


class ImageFolderDataset(Dataset):
    """Dataset from a directory of images (any size, auto-resized)."""

    def __init__(self, root: str, img_size: int = 64, augment: bool = True):
        self.dataset = datasets.ImageFolder(
            root=root,
            transform=get_default_transforms(img_size, augment),
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label


def get_dataloader(
    dataset_name: str = "imagenet64",
    data_dir: str = "./data",
    img_size: int = 64,
    batch_size: int = 64,
    num_workers: int = 4,
    train: bool = True,
    max_samples: Optional[int] = None,
) -> DataLoader:
    """
    Get a DataLoader for the specified dataset.

    Args:
        dataset_name: "imagenet64", "cifar10", or "imagefolder"
        data_dir: path to dataset directory
        img_size: resize images to this size
        batch_size: batch size
        num_workers: dataloader workers
        train: True for training split, False for validation
        max_samples: limit dataset size (for quick testing)

    Returns:
        DataLoader yielding (images, labels) tuples
    """
    transform = get_default_transforms(img_size, augment=train)

    if dataset_name == "cifar10":
        # Try existing CIFAR-10 paths first
        search_paths = [
            data_dir,
            os.path.expanduser('~/PixelDiT/data'),
            os.path.expanduser('~/data'),
            './data',
        ]
        ds = None
        for path in search_paths:
            try:
                ds = datasets.CIFAR10(root=path, train=train, download=False, transform=transform)
                if len(ds) > 0:
                    print(f"Found CIFAR-10 at: {path}")
                    break
            except Exception:
                continue
        if ds is None:
            print(f"CIFAR-10 not found locally, downloading to {data_dir}...")
            ds = datasets.CIFAR10(root=data_dir, train=train, download=True, transform=transform)

    elif dataset_name == "imagenet64":
        # ImageNet 64×64 — downloaded via torchvision or pre-prepared
        # Standard location: data/imagenet64/train, data/imagenet64/val
        split = 'train' if train else 'val'
        ds_path = os.path.join(data_dir, 'imagenet64', split)
        if os.path.exists(ds_path):
            ds = datasets.ImageFolder(root=ds_path, transform=transform)
        else:
            # Try ImageNet-1k (full size, will be resized)
            ds_path = os.path.join(data_dir, 'imagenet', split)
            if os.path.exists(ds_path):
                ds = datasets.ImageFolder(root=ds_path, transform=transform)
            else:
                raise FileNotFoundError(
                    f"ImageNet not found at {ds_path}. "
                    "Download ImageNet-64 from https://image-net.org/ or use 'cifar10'."
                )

    elif dataset_name == "imagefolder":
        ds = datasets.ImageFolder(root=data_dir, transform=transform)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if max_samples is not None and max_samples < len(ds):
        ds = Subset(ds, range(max_samples))

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=train,
        num_workers=num_workers, pin_memory=True, drop_last=train,
    )
    return loader
