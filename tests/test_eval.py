import numpy as np
import torch

from sslbench.data import class_balanced_indices
from sslbench.eval import effective_rank, knn_accuracy, linear_probe


def clusters(n_per_class=50, num_classes=4, dim=16, seed=0):
    centers = torch.randn(num_classes, dim, generator=torch.Generator().manual_seed(123)) * 5
    g = torch.Generator().manual_seed(seed)  # same centers, different noise per split
    y = torch.arange(num_classes).repeat_interleave(n_per_class)
    return centers[y] + torch.randn(len(y), dim, generator=g), y


def test_knn_separable_clusters():
    xtr, ytr = clusters(seed=0)
    xte, yte = clusters(seed=1)
    assert knn_accuracy(xtr, ytr, xte, yte, num_classes=4, k=5) > 0.95


def test_linear_probe_separable_clusters():
    xtr, ytr = clusters(seed=0)
    xte, yte = clusters(seed=1)
    assert linear_probe(xtr, ytr, xte, yte, num_classes=4, epochs=30, lr=0.1, batch_size=64) > 0.95


def test_effective_rank_bounds():
    torch.manual_seed(0)
    iso = torch.randn(4096, 16)
    collapsed = torch.randn(4096, 1) @ torch.randn(1, 16)
    assert 14 < effective_rank(iso) <= 16
    assert effective_rank(collapsed) < 1.5


def test_class_balanced_subsets_are_balanced_and_nested():
    targets = np.repeat(np.arange(10), 100)
    small = class_balanced_indices(targets, 0.1, seed=0)
    big = class_balanced_indices(targets, 0.5, seed=0)
    assert len(small) == 100 and np.all(np.bincount(targets[small]) == 10)
    assert set(small) <= set(big)
