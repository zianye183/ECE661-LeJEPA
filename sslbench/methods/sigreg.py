"""SIGReg with the Epps-Pulley test, written out step by step.

This mirrors the official implementation in the `lejepa` package
(lejepa.univariate.EppsPulley + lejepa.multivariate.SlicingUnivariateTest), and
tests/test_sigreg.py checks that both give the same numbers. LeJEPA training uses the
official one by default (method.sigreg_impl: official); this file exists so we
understand, and can modify, what SIGReg computes.
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
    n = x.size(-2)
    t = torch.linspace(0, t_max, n_points, device=x.device, dtype=x.dtype)
    phi = torch.exp(-0.5 * t.square())  # CF of N(0, 1), also the Gaussian window w(t)
    dt = t_max / (n_points - 1)
    weights = torch.full_like(t, 2 * dt)
    weights[0] = weights[-1] = dt
    weights = weights * phi

    xt = x.unsqueeze(-1) * t  # (..., N, K, T)
    ecf_real = torch.cos(xt).mean(-3)  # (..., K, T): real part of the empirical CF
    ecf_imag = torch.sin(xt).mean(-3)  # imaginary part
    err = (ecf_real - phi).square() + ecf_imag.square()  # |ecf - phi|^2
    return (err @ weights) * n


def random_directions(dim: int, num_slices: int, seed: int, device) -> torch.Tensor:
    """(dim, num_slices) matrix of unit-norm Gaussian directions, reproducible from `seed`."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    a = torch.randn((dim, num_slices), device=device, generator=g)
    return a / a.norm(p=2, dim=0)


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
        with torch.no_grad():
            a = random_directions(z.size(-1), self.num_slices, int(self.step.item()), z.device)
            self.step.add_(1)
        return epps_pulley(z @ a, self.t_max, self.n_points).mean()
