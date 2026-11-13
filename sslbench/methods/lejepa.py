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
    # TODO(2a): see DIRECTIONS.md, Task 2. Return (loss, prediction_loss, sigreg_loss).
    raise NotImplementedError("lejepa_loss")


class LeJEPA(SSLMethod):
    def __init__(self, backbone: nn.Module, feat_dim: int, cfg: dict):
        super().__init__(backbone, feat_dim)
        self.lamb = cfg["lamb"]
        self.projector = mlp([feat_dim, cfg["proj_hidden"], cfg["proj_hidden"], cfg["proj_dim"]])
        self.sigreg = build_sigreg(cfg)

    def forward(self, views: torch.Tensor):
        # TODO(2b): run all N*V views through the backbone in one batch, project them,
        # reshape to (V, N, K), compute lejepa_loss in fp32 (autocast disabled), and return
        # (loss, logs dict with "loss", "pred", "sigreg", detached backbone features of view 0).
        raise NotImplementedError("LeJEPA.forward")
