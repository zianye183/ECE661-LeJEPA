"""Frozen-backbone evaluation shared by every method: k-NN, linear probes, effective rank."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data import class_balanced_indices


@torch.no_grad()
def extract_features(model: nn.Module, ds: Dataset, device, batch_size: int = 512, num_workers: int = 4):
    """Run model.embed over a dataset. Returns (features[N, F] float32 on CPU, labels[N])."""
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
    feats, labels = [], []
    for x, y in loader:
        feats.append(model.embed(x.to(device, non_blocking=True)).float().cpu())
        labels.append(torch.as_tensor(y))
    return torch.cat(feats), torch.cat(labels)


def knn_accuracy(train_f, train_y, test_f, test_y, num_classes: int, k: int = 20, temperature: float = 0.07,
                 chunk: int = 1024) -> float:
    """Weighted k-NN on cosine similarity (Wu et al., 2018): each neighbour votes exp(sim / T)."""
    train_f = F.normalize(train_f, dim=1)
    test_f = F.normalize(test_f, dim=1)
    correct = 0
    for start in range(0, len(test_f), chunk):
        sims = test_f[start:start + chunk] @ train_f.T  # (B, N_train)
        top_sim, top_idx = sims.topk(k, dim=1)
        votes = torch.zeros(len(top_idx), num_classes)
        votes.scatter_add_(1, train_y[top_idx], (top_sim / temperature).exp())
        correct += (votes.argmax(1) == test_y[start:start + chunk]).sum().item()
    return correct / len(test_f)


def linear_probe(train_f, train_y, test_f, test_y, num_classes: int, epochs: int = 100, lr: float = 0.1,
                 batch_size: int = 1024, device="cpu", seed: int = 0) -> float:
    """Train a linear classifier on frozen, standardized features; return test top-1."""
    torch.manual_seed(seed)
    mean, std = train_f.mean(0), train_f.std(0) + 1e-6
    xtr, xte = ((train_f - mean) / std).to(device), ((test_f - mean) / std).to(device)
    ytr, yte = train_y.to(device), test_y.to(device)
    clf = nn.Linear(xtr.size(1), num_classes).to(device)
    opt = torch.optim.SGD(clf.parameters(), lr=lr, momentum=0.9, weight_decay=0.0)
    steps_per_epoch = max(1, -(-len(xtr) // batch_size))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch)
    for _ in range(epochs):
        perm = torch.randperm(len(xtr), device=device)
        for start in range(0, len(xtr), batch_size):
            idx = perm[start:start + batch_size]
            loss = F.cross_entropy(clf(xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    with torch.no_grad():
        return (clf(xte).argmax(1) == yte).float().mean().item()


def effective_rank(feats: torch.Tensor, eps: float = 1e-12) -> float:
    """RankMe (Garrido et al., 2023): exp(entropy of normalized singular values).

    Ranges from 1 (complete collapse) to min(N, D) (isotropic). Our collapse check.
    """
    s = torch.linalg.svdvals(feats.double() - feats.double().mean(0))
    p = s / (s.sum() + eps) + eps
    return float(torch.exp(-(p * p.log()).sum()))


def evaluate(model: nn.Module, train_ds: Dataset, test_ds: Dataset, num_classes: int, cfg_eval: dict, device,
             num_workers: int = 4) -> dict[str, float]:
    """Every downstream metric we report, computed the same way for every method."""
    train_f, train_y = extract_features(model, train_ds, device, num_workers=num_workers)
    test_f, test_y = extract_features(model, test_ds, device, num_workers=num_workers)
    results = {
        "knn_top1": knn_accuracy(train_f, train_y, test_f, test_y, num_classes, cfg_eval["knn_k"],
                                 cfg_eval["knn_temperature"]),
        "effective_rank": effective_rank(test_f),
    }
    for frac in cfg_eval["label_fractions"]:
        idx = torch.as_tensor(class_balanced_indices(train_y.numpy(), frac, seed=cfg_eval["label_subset_seed"]))
        acc = linear_probe(train_f[idx], train_y[idx], test_f, test_y, num_classes, cfg_eval["linear_epochs"],
                           cfg_eval["linear_lr"], cfg_eval["linear_batch_size"], device)
        results[f"linear_top1_{round(frac * 100)}pct"] = acc
    return results
