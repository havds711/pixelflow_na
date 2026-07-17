"""
Data loading for pixel-space Flow Matching training.

Supports:
  - ImageNet-64 parquet (native 256→64 downsample, 124/1000 classes)
  - ImageNet (via torchvision ImageFolder)
  - CIFAR-10 (resized from 32, for smoke testing)
  - Any image directory
"""

import io
import os
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from typing import Optional, Tuple


def get_default_transforms(img_size: int = 64, augment: bool = True):
    """Default image transforms for pixel-space generation."""
    if augment:
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
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


class ImageNetParquetDataset(Dataset):
    """
    ImageNet dataset stored in parquet format.

    Each parquet file has columns:
      - image: struct{bytes: binary, path: string} — JPEG-encoded image bytes
      - label: int64 — ImageNet class label (0-indexed)

    Args:
        parquet_dir: directory containing train-*.parquet / validation-*.parquet
        img_size: resize images to this size
        train: True for training split, False for validation
        augment: apply random horizontal flip (train only)
    """

    def __init__(
        self,
        parquet_dir: str,
        img_size: int = 64,
        train: bool = True,
        augment: bool = True,
    ):
        import pyarrow.parquet as pq

        self.img_size = img_size
        self.train = train
        self.augment = augment and train
        self.transform = get_default_transforms(img_size, augment=self.augment)

        # Find parquet files
        prefix = 'train' if train else 'validation'
        pattern = os.path.join(parquet_dir, f'{prefix}-*.parquet')
        self.parquet_files = sorted(glob.glob(pattern))
        if not self.parquet_files:
            raise FileNotFoundError(f"No parquet files found at {pattern}")

        # Load metadata: (file_idx, row_in_file) for each sample
        self._samples = []  # list of (file_idx, row_in_file)
        self._labels = []
        self._label_set = set()

        print(f"Loading {prefix} parquet metadata...")
        for file_idx, fpath in enumerate(self.parquet_files):
            meta = pq.read_metadata(fpath)
            n_rows = meta.num_rows
            # Read labels only (fast — no image decoding)
            table = pq.read_table(fpath, columns=['label'])
            file_labels = table['label'].to_pylist()

            for row_idx in range(n_rows):
                self._samples.append((file_idx, row_idx))
                self._labels.append(file_labels[row_idx])
                self._label_set.add(file_labels[row_idx])

        self._parquet_cache = {}  # file_idx -> pyarrow table (lazy loaded)
        self.num_classes = len(self._label_set)

        print(f"  {len(self._samples):,} images, {self.num_classes} classes "
              f"(labels {min(self._label_set)}-{max(self._label_set)})")

    def _get_parquet_table(self, file_idx: int):
        """Lazy-load parquet file."""
        import pyarrow.parquet as pq
        if file_idx not in self._parquet_cache:
            self._parquet_cache[file_idx] = pq.read_table(self.parquet_files[file_idx])
        return self._parquet_cache[file_idx]

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        file_idx, row_idx = self._samples[idx]
        table = self._get_parquet_table(file_idx)

        # Decode JPEG bytes
        img_data = table['image'][row_idx]
        if hasattr(img_data, 'as_py'):
            img_data = img_data.as_py()
        if isinstance(img_data, dict):
            img_bytes = img_data['bytes']
        else:
            img_bytes = img_data

        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # Apply transforms
        if self.transform:
            img = self.transform(img)

        label = self._labels[idx]
        return img, label


def get_dataloader(
    dataset_name: str = "imagenet_parquet",
    data_dir: str = "./data",
    img_size: int = 64,
    batch_size: int = 64,
    num_workers: int = 4,
    train: bool = True,
    max_samples: Optional[int] = None,
) -> Tuple[DataLoader, int]:
    """
    Get a DataLoader for the specified dataset.

    Args:
        dataset_name: "imagenet_parquet", "imagenet64", "cifar10", or "imagefolder"
        data_dir: path to dataset directory
        img_size: resize images to this size
        batch_size: batch size
        num_workers: dataloader workers
        train: True for training split, False for validation
        max_samples: limit dataset size (for quick testing)

    Returns:
        (dataloader, num_classes)
    """
    transform = get_default_transforms(img_size, augment=train)

    if dataset_name == "imagenet_parquet":
        ds = ImageNetParquetDataset(
            parquet_dir=data_dir,
            img_size=img_size,
            train=train,
            augment=train,
        )
        num_classes = ds.num_classes

    elif dataset_name == "cifar10":
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
        num_classes = 10

    elif dataset_name == "imagenet64":
        split = 'train' if train else 'val'
        ds_path = os.path.join(data_dir, 'imagenet64', split)
        if os.path.exists(ds_path):
            ds = datasets.ImageFolder(root=ds_path, transform=transform)
        else:
            ds_path = os.path.join(data_dir, 'imagenet', split)
            if os.path.exists(ds_path):
                ds = datasets.ImageFolder(root=ds_path, transform=transform)
            else:
                raise FileNotFoundError(
                    f"ImageNet not found at {ds_path}. "
                    "Download ImageNet-64 from https://image-net.org/ or use 'cifar10'."
                )
        num_classes = len(ds.classes)

    elif dataset_name == "imagefolder":
        ds = datasets.ImageFolder(root=data_dir, transform=transform)
        num_classes = len(ds.classes)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if max_samples is not None and max_samples < len(ds):
        ds = Subset(ds, range(max_samples))

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=train,
        num_workers=num_workers, pin_memory=True, drop_last=train,
    )
    return loader, num_classes
