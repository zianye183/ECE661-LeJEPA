"""The interface every SSL method implements. train.py only talks to this."""

from __future__ import annotations

import torch
from torch import nn


class SSLMethod(nn.Module):
    """Backbone + method-specific heads + loss.

    forward(views) takes a batch of augmented views shaped (N, V, C, H, W) and returns
    (loss, logs, feats): logs is a dict of floats to record, feats are the detached
    backbone features of view 0, shape (N, F), used by the online probe. embed(x)
    returns backbone features for (N, C, H, W) images; evaluation uses only embed().
    """

    def __init__(self, backbone: nn.Module, feat_dim: int):
        super().__init__()
        self.backbone = backbone
        self.feat_dim = feat_dim

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, views: torch.Tensor) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
        raise NotImplementedError


def mlp(dims: list[int], last_bn: bool = False, last_bn_affine: bool = True) -> nn.Sequential:
    """Linear-BN-ReLU blocks; the final layer is Linear (optionally followed by BN)."""
    layers: list[nn.Module] = []
    for i in range(len(dims) - 2):
        layers += [nn.Linear(dims[i], dims[i + 1], bias=False), nn.BatchNorm1d(dims[i + 1]), nn.ReLU(inplace=True)]
    layers.append(nn.Linear(dims[-2], dims[-1], bias=not last_bn))
    if last_bn:
        layers.append(nn.BatchNorm1d(dims[-1], affine=last_bn_affine))
    return nn.Sequential(*layers)
