"""Datasets, transforms and DataLoaders.

Three datasets are supported so the project can be pointed at YOUR data later:
  - oxford_pet : Oxford-IIIT Pets (37 cat/dog breeds). torchvision downloads it if missing.
  - cifar10    : tiny 32x32 images, handy for very fast experiments.
  - folder     : any folder of images laid out like  root/<class_name>/*.jpg
                 (optionally root/train/... and root/val/...). This is how you plug in your own project.

`subset_frac` keeps only a fraction of the data. That is the key trick for working on a weak PC:
debug the whole pipeline on ~2% of the images (seconds), then run the full dataset on a free GPU.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

# Pretrained torchvision models were trained on ImageNet with these channel statistics,
# so we must normalise our images the same way for transfer learning to work well.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DATASETS = ("oxford_pet", "cifar10", "folder")


def build_transforms(img_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    """Train transforms add randomness (augmentation); val transforms are deterministic."""
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),  # random crop + resize
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # PIL image -> float tensor in [0, 1], shape (3, H, W)
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.14)),  # 256 for 224: resize, then centre-crop
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tf, val_tf


def _subset(ds: Dataset, frac: float, seed: int, min_items: int = 64) -> Dataset:
    """Keep a random `frac` of `ds` (at least `min_items`, so tiny fractions still make a usable val set)."""
    if frac >= 1.0:
        return ds
    n = len(ds)
    k = min(n, max(min_items, int(n * frac)))
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g)[:k].tolist()
    return Subset(ds, idx)


def _load_folder(root: Path, train_tf, val_tf, seed: int):
    """ImageFolder with either explicit train/ + val/ folders, or one folder split 80/20."""
    if (root / "train").is_dir() and (root / "val").is_dir():
        train_ds = datasets.ImageFolder(root / "train", transform=train_tf)
        val_ds = datasets.ImageFolder(root / "val", transform=val_tf)
        return train_ds, val_ds, train_ds.classes

    # One folder -> two ImageFolder views of the same files (different transforms), split by index.
    full_train = datasets.ImageFolder(root, transform=train_tf)
    full_val = datasets.ImageFolder(root, transform=val_tf)
    n_val = int(0.2 * len(full_train))
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(full_train), generator=g).tolist()
    return Subset(full_train, perm[n_val:]), Subset(full_val, perm[:n_val]), full_train.classes


def get_loaders(
    dataset: str,
    root: str | Path,
    img_size: int = 224,
    batch_size: int = 32,
    subset_frac: float = 1.0,
    num_workers: int = 0,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader, int, list[str]]:
    """Return (train_loader, val_loader, num_classes, class_names)."""
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset '{dataset}', choose one of {DATASETS}")
    root = Path(root)
    train_tf, val_tf = build_transforms(img_size)

    if dataset == "oxford_pet":
        # 'trainval' (3680 imgs) for training, 'test' (3669 imgs) for validation.
        train_ds = datasets.OxfordIIITPet(root, split="trainval", target_types="category", transform=train_tf, download=True)
        val_ds = datasets.OxfordIIITPet(root, split="test", target_types="category", transform=val_tf, download=True)
        classes = train_ds.classes
    elif dataset == "cifar10":
        train_ds = datasets.CIFAR10(root, train=True, transform=train_tf, download=True)
        val_ds = datasets.CIFAR10(root, train=False, transform=val_tf, download=True)
        classes = train_ds.classes
    else:  # folder
        train_ds, val_ds, classes = _load_folder(root, train_tf, val_tf, seed)

    train_ds = _subset(train_ds, subset_frac, seed)
    val_ds = _subset(val_ds, subset_frac, seed)

    pin = torch.cuda.is_available()  # faster host->GPU copies; pointless on CPU
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader, len(classes), list(classes)
