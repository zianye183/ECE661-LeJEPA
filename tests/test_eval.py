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


def test_lars_and_lr_scaling():
    from sslbench.optim import LARS, build_optimizer, resolve_lr

    assert resolve_lr({"base_lr": 0.03}, 512) == 0.06
    assert resolve_lr({"lr": 1.0, "base_lr": 0.03}, 512) == 1.0
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(8, 8))
    target = torch.randn(8, 8)
    opt = LARS([w], lr=1.0, momentum=0.9, weight_decay=0.0)
    first = None
    for _ in range(200):
        loss = (w - target).square().sum()
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5 * first
    cfg = {"optim": {"name": "lars", "lr": 1.0, "momentum": 0.9, "weight_decay": 1e-6}, "train": {"batch_size": 256}}
    model = torch.nn.Linear(8, 8)
    opt = build_optimizer(model, [torch.nn.Parameter(torch.zeros(3, 3))], cfg)
    assert opt.param_groups[1]["lars_exclude"] and opt.param_groups[1]["lr"] == 0.1


def test_fix_pred_lr_puts_predictor_in_its_own_group():
    from sslbench.backbones import build_backbone
    from sslbench.methods.simsiam import SimSiam
    from sslbench.optim import build_optimizer

    backbone, dim = build_backbone("resnet18_cifar")
    model = SimSiam(backbone, dim, {"proj_hidden": 64, "proj_dim": 32, "pred_hidden": 16, "proj_layers": 2})
    cfg = {"optim": {"name": "sgd", "base_lr": 0.03, "momentum": 0.9, "weight_decay": 5e-4, "fix_pred_lr": True},
           "train": {"batch_size": 256}}
    opt = build_optimizer(model, [], cfg)
    assert len(opt.param_groups) == 2 and opt.param_groups[1].get("fixed_lr")
    n_model = sum(1 for _ in model.parameters())
    assert sum(len(g["params"]) for g in opt.param_groups) == n_model
