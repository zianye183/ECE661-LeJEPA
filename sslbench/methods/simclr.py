"""SimCLR with the NT-Xent loss (Chen et al., 2020)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .base import SSLMethod, mlp


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """z1, z2: (N, K) projections of two views. Each sample's positive is its other
    view; the other 2N - 2 samples in the batch are negatives."""
    n = z1.size(0)
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    logits = z @ z.T / temperature
    self_mask = torch.eye(2 * n, dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(self_mask, float("-inf"))
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(logits, targets)


class SimCLR(SSLMethod):
    def __init__(self, backbone: nn.Module, feat_dim: int, cfg: dict):
        super().__init__(backbone, feat_dim)
        self.temperature = cfg["temperature"]
        self.projector = mlp([feat_dim, cfg["proj_hidden"], cfg["proj_dim"]])

    def forward(self, views: torch.Tensor):
        n = views.size(0)
        emb = self.backbone(views[:, :2].transpose(0, 1).flatten(0, 1))  # view-major: (2N, F)
        z = self.projector(emb)
        with torch.autocast(device_type=views.device.type, enabled=False):
            loss = nt_xent(z[:n].float(), z[n:].float(), self.temperature)
        return loss, {"loss": loss.item()}, emb[:n].detach()
