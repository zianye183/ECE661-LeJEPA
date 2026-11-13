"""SimSiam (Chen & He, 2021): predictor + stop-gradient, no negatives."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .base import SSLMethod, mlp


def simsiam_loss(p1: torch.Tensor, p2: torch.Tensor, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Symmetric negative cosine similarity; targets z are detached (stop-gradient)."""
    # TODO(3b): see DIRECTIONS.md, Task 3. Remember the stop-gradient.
    raise NotImplementedError("simsiam_loss")


class SimSiam(SSLMethod):
    def __init__(self, backbone: nn.Module, feat_dim: int, cfg: dict):
        super().__init__(backbone, feat_dim)
        h, d = cfg["proj_hidden"], cfg["proj_dim"]
        self.projector = mlp([feat_dim, h, h, d], last_bn=True, last_bn_affine=False)
        self.predictor = mlp([d, cfg["pred_hidden"], d])

    def forward(self, views: torch.Tensor):
        n = views.size(0)
        emb = self.backbone(views[:, :2].transpose(0, 1).flatten(0, 1))  # view-major: (2N, F)
        z = self.projector(emb)
        p = self.predictor(z)
        z1, z2, p1, p2 = z[:n], z[n:], p[:n], p[n:]
        loss = simsiam_loss(p1, p2, z1, z2)
        # Collapse check from the SimSiam paper: std of normalized z per dim ~ 1/sqrt(d) if healthy.
        z_std = F.normalize(z1.detach().float(), dim=1).std(0).mean().item()
        return loss, {"loss": loss.item(), "z_std": z_std}, emb[:n].detach()
