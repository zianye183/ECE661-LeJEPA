import torch

from sslbench.methods.lejepa import lejepa_loss
from sslbench.methods.sigreg import SIGReg
from sslbench.methods.simclr import nt_xent
from sslbench.methods.simsiam import simsiam_loss


def test_lejepa_prediction_loss_is_zero_when_views_agree():
    z = torch.randn(1, 64, 8).expand(4, 64, 8)  # 4 identical views
    _, pred, _ = lejepa_loss(z, SIGReg(num_slices=16), lamb=0.05)
    assert pred.abs() < 1e-8


def test_lejepa_prediction_loss_value():
    z = torch.tensor([[[0.0, 0.0]], [[2.0, 4.0]]])  # V=2, N=1, K=2; center = (1, 2)
    _, pred, _ = lejepa_loss(z, SIGReg(num_slices=4), lamb=0.0)
    assert torch.isclose(pred, torch.tensor(2.5))  # mean of squared deviations: (1+4+1+4)/4


def test_lejepa_combines_terms_with_lambda():
    torch.manual_seed(0)
    z = torch.randn(2, 128, 8)
    loss, pred, sig = lejepa_loss(z, SIGReg(num_slices=16), lamb=0.25)
    assert torch.isclose(loss, 0.75 * pred + 0.25 * sig)


def test_nt_xent_low_for_matching_views_high_for_random():
    torch.manual_seed(0)
    z = torch.randn(64, 32)
    matched = nt_xent(z, z + 0.01 * torch.randn_like(z), temperature=0.2)
    random = nt_xent(z, torch.randn(64, 32), temperature=0.2)
    assert matched < 1.0 and random > 3.0  # random ~ log(2N - 1) = 4.8


def test_nt_xent_lower_bound_uses_2n_minus_1_candidates():
    # All embeddings identical: every logit equal, loss = log(2N - 1).
    z = torch.ones(8, 4)
    assert torch.isclose(nt_xent(z, z, temperature=1.0), torch.log(torch.tensor(15.0)), atol=1e-5)


def test_simsiam_loss_range_and_stop_gradient():
    p1, p2 = torch.randn(16, 8, requires_grad=True), torch.randn(16, 8, requires_grad=True)
    z1, z2 = torch.randn(16, 8, requires_grad=True), torch.randn(16, 8, requires_grad=True)
    loss = simsiam_loss(p1, p2, z1, z2)
    assert -1.0 <= loss.item() <= 1.0
    loss.backward()
    assert p1.grad is not None and z1.grad is None and z2.grad is None
    same = torch.randn(16, 8)
    assert torch.isclose(simsiam_loss(same, same, same, same), torch.tensor(-1.0))
