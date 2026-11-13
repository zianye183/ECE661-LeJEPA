"""SIGReg with the Epps-Pulley test, written out step by step.

This mirrors the official implementation in the `lejepa` package
(lejepa.univariate.EppsPulley + lejepa.multivariate.SlicingUnivariateTest), and
tests/test_sigreg.py checks that both give the same numbers. LeJEPA training uses the
official one by default on `main`. On this branch the LeJEPA config uses ours
(method.sigreg_impl: ours), so training only works once Task 1 is done.
"""

from __future__ import annotations

import torch
from torch import nn


def epps_pulley(x: torch.Tensor, t_max: float = 3.0, n_points: int = 17) -> torch.Tensor:
    """Epps-Pulley statistic of each column of x against N(0, 1).

    x: (..., N, K) holding N samples of K one-dimensional projections.
    Returns (..., K): N * integral of w(t) |ecf(t) - exp(-t^2/2)|^2 dt, with w(t) = exp(-t^2/2).

    The integrand is symmetric in t, so we integrate over [0, t_max] with the
    trapezoid rule and double the weights.
    """
    # TODO(1a): implement. See DIRECTIONS.md, Task 1.
    # Steps: integration points t in [0, t_max]; phi(t) = exp(-t^2/2); trapezoid weights
    # (doubled, because we only integrate t >= 0) times phi; empirical CF of each column
    # (mean over the N axis, which is dim -2 of x); weighted squared error; times N.
    raise NotImplementedError("epps_pulley")


def random_directions(dim: int, num_slices: int, seed: int, device) -> torch.Tensor:
    """(dim, num_slices) matrix of unit-norm Gaussian directions, reproducible from `seed`."""
    # TODO(1b): Gaussian (dim, num_slices) matrix from a torch.Generator seeded with `seed`,
    # each column scaled to unit L2 norm. Must match the official package exactly: draw
    # with torch.randn((dim, num_slices), device=device, generator=g).
    raise NotImplementedError("random_directions")


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularization.

    Projects embeddings onto `num_slices` random unit directions (new ones every call)
    and averages the Epps-Pulley statistic over directions (and any leading dims).
    """

    def __init__(self, num_slices: int = 1024, t_max: float = 3.0, n_points: int = 17):
        super().__init__()
        self.num_slices = num_slices
        self.t_max = t_max
        self.n_points = n_points
        self.register_buffer("step", torch.zeros((), dtype=torch.long))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (..., N, D) -> scalar."""
        # TODO(1c): directions from random_directions(..., seed=self.step) without gradients,
        # then increment self.step; project z, apply epps_pulley, average everything.
        raise NotImplementedError("SIGReg.forward")
