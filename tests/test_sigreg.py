import pytest
import torch

from sslbench.methods.sigreg import SIGReg, epps_pulley, random_directions


def test_epps_pulley_shape():
    x = torch.randn(3, 256, 8)  # (..., N, K)
    assert epps_pulley(x).shape == (3, 8)


def test_epps_pulley_small_for_gaussian_large_for_non_gaussian():
    torch.manual_seed(0)
    gaussian = epps_pulley(torch.randn(2048, 16)).mean()
    shifted = epps_pulley(torch.randn(2048, 16) + 2.0).mean()
    collapsed = epps_pulley(torch.zeros(2048, 16)).mean()
    assert gaussian < 3.0  # N * distance: stays O(1) for Gaussian data
    assert shifted > 50 * gaussian
    assert collapsed > 50 * gaussian


def test_random_directions_are_unit_and_reproducible():
    a = random_directions(32, 64, seed=3, device="cpu")
    assert a.shape == (32, 64)
    assert torch.allclose(a.norm(dim=0), torch.ones(64), atol=1e-5)
    assert torch.equal(a, random_directions(32, 64, seed=3, device="cpu"))


def test_sigreg_matches_official_lejepa_package():
    lejepa = pytest.importorskip("lejepa")
    torch.manual_seed(0)
    z = torch.randn(4, 128, 32) * 1.5 + 0.3  # (V, N, D)
    official = lejepa.multivariate.SlicingUnivariateTest(
        univariate_test=lejepa.univariate.EppsPulley(t_max=3.0, n_points=17), num_slices=64)
    ours = SIGReg(num_slices=64, t_max=3.0, n_points=17)
    for _ in range(2):  # also checks both advance their random directions the same way
        assert torch.allclose(ours(z), official(z), rtol=1e-4, atol=1e-5)


def test_sigreg_has_gradients():
    z = torch.randn(64, 16, requires_grad=True)
    SIGReg(num_slices=32)(z * 3).backward()
    assert z.grad is not None and z.grad.abs().sum() > 0
