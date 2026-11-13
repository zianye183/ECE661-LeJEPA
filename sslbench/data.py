"""Datasets, augmentations and class-balanced subsets shared by every method."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
from torchvision.transforms import v2

STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    "fake": ((0.5, 0.5, 0.5), (0.29, 0.29, 0.29)),
}
NUM_CLASSES = {"cifar10": 10, "cifar100": 100, "fake": 10}


class FakeImages(Dataset):
    """Random 32x32 RGB images with a `targets` list, for tests without downloads."""

    def __init__(self, size: int = 512, num_classes: int = 10, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.images = rng.integers(0, 256, size=(size, 32, 32, 3), dtype=np.uint8)
        self.targets = rng.integers(0, num_classes, size=size).tolist()

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, i):
        return Image.fromarray(self.images[i]), self.targets[i]


def base_dataset(name: str, root: str, train: bool) -> Dataset:
    """Raw dataset returning (PIL image, label). Downloads CIFAR on first use."""
    if name == "cifar10":
        return datasets.CIFAR10(root, train=train, download=True)
    if name == "cifar100":
        return datasets.CIFAR100(root, train=train, download=True)
    if name == "fake":
        return FakeImages(size=512 if train else 256, seed=0 if train else 1)
    raise ValueError(f"Unknown dataset {name!r}")


def eval_transform(name: str) -> v2.Compose:
    mean, std = STATS[name]
    return v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)])


def ssl_transform(name: str, aug: dict) -> v2.Compose:
    """One random view. Every method uses this same pipeline (fair comparison)."""
    mean, std = STATS[name]
    return v2.Compose(
        [
            v2.RandomResizedCrop(32, scale=tuple(aug["crop_scale"]), antialias=True),
            v2.RandomHorizontalFlip(),
            v2.RandomApply([v2.ColorJitter(*aug["color_jitter"])], p=aug["color_jitter_p"]),
            v2.RandomGrayscale(p=aug["grayscale_p"]),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean, std),
        ]
    )


class MultiView(Dataset):
    """Wraps a (PIL, label) dataset and returns (views[V, C, H, W], label)."""

    def __init__(self, base: Dataset, transform, num_views: int):
        self.base = base
        self.transform = transform
        self.num_views = num_views

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i):
        img, label = self.base[i]
        return torch.stack([self.transform(img) for _ in range(self.num_views)]), label


class Transformed(Dataset):
    def __init__(self, base: Dataset, transform):
        self.base = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i):
        img, label = self.base[i]
        return self.transform(img), label


def targets_of(ds: Dataset) -> np.ndarray:
    if isinstance(ds, Subset):
        return targets_of(ds.dataset)[np.asarray(ds.indices)]
    return np.asarray(ds.targets)


def class_balanced_indices(targets, fraction: float, seed: int = 0) -> np.ndarray:
    """Indices of a class-balanced subset holding `fraction` of each class.

    Subsets are nested: with the same seed, the 1% subset is inside the 10% subset,
    which is inside the 50% subset, and so on.
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    targets = np.asarray(targets)
    rng = np.random.default_rng(seed)
    chosen = []
    for cls in np.unique(targets):
        idx = np.flatnonzero(targets == cls)
        idx = idx[rng.permutation(len(idx))]
        chosen.append(idx[: max(1, round(fraction * len(idx)))])
    return np.sort(np.concatenate(chosen))


def pretrain_loader(cfg: dict) -> DataLoader:
    d = cfg["data"]
    base = base_dataset(d["dataset"], d["root"], train=True)
    if d.get("fraction", 1.0) < 1.0:
        base = Subset(base, class_balanced_indices(targets_of(base), d["fraction"], seed=0).tolist())
    ds = MultiView(base, ssl_transform(d["dataset"], d["augment"]), cfg["method"]["num_views"])
    return DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=d["num_workers"],
        pin_memory=torch.cuda.is_available(),
        persistent_workers=d["num_workers"] > 0,
    )


def eval_datasets(cfg: dict) -> tuple[Dataset, Dataset]:
    """Un-augmented full train and test sets, for probes and k-NN."""
    d = cfg["data"]
    tf = eval_transform(d["dataset"])
    train = Transformed(base_dataset(d["dataset"], d["root"], train=True), tf)
    test = Transformed(base_dataset(d["dataset"], d["root"], train=False), tf)
    return train, test
