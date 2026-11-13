"""LeJEPA: prediction (invariance) loss + SIGReg (arXiv 2511.08544, Algorithm 2)."""

from __future__ import annotations

import torch
from torch import nn

from .base import SSLMethod, mlp
from .sigreg import SIGReg


def build_sigreg(cfg: dict) -> nn.Module:
    if cfg.get("sigreg_impl", "official") == "official":
        from lejepa.multivariate import SlicingUnivariateTest
        from lejepa.univariate import EppsPulley

        return SlicingUnivariateTest(
            univariate_test=EppsPulley(t_max=cfg["t_max"], n_points=cfg["n_points"]),
            num_slices=cfg["num_slices"],
        )
    return SIGReg(num_slices=cfg["num_slices"], t_max=cfg["t_max"], n_points=cfg["n_points"])


def lejepa_loss(proj: torch.Tensor, sigreg: nn.Module, lamb: float):
    """proj: (V, N, K) projections of V views of N images.

    With a ResNet every view is a global view, so the prediction target for each
    image is the mean projection over its views. SIGReg is applied per view
    (the leading V dim) and averaged.
    Returns (loss, prediction_loss, sigreg_loss).
    """
    centers = proj.mean(0, keepdim=True)
    pred = (centers - proj).square().mean()
    sig = sigreg(proj)
    return (1 - lamb) * pred + lamb * sig, pred, sig


class LeJEPA(SSLMethod):
    def __init__(self, backbone: nn.Module, feat_dim: int, cfg: dict):
        super().__init__(backbone, feat_dim)
        self.lamb = cfg["lamb"]
        self.projector = mlp([feat_dim, cfg["proj_hidden"], cfg["proj_hidden"], cfg["proj_dim"]])
        self.sigreg = build_sigreg(cfg)

    def forward(self, views: torch.Tensor):
        n, v = views.shape[:2]
        emb = self.backbone(views.flatten(0, 1))  # (N*V, F), image-major
        proj = self.projector(emb).reshape(n, v, -1).transpose(0, 1)  # (V, N, K)
        # SIGReg uses cos/sin of projections: compute the loss in fp32.
        with torch.autocast(device_type=views.device.type, enabled=False):
            loss, pred, sig = lejepa_loss(proj.float(), self.sigreg, self.lamb)
        logs = {"loss": loss.item(), "pred": pred.item(), "sigreg": sig.item()}
        return loss, logs, emb.reshape(n, v, -1)[:, 0].detach()
