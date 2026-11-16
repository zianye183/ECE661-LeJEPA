"""Optimizers. LARS follows SimCLR: biases and BatchNorm parameters (1-D tensors) get
neither weight decay nor the layer-wise trust ratio."""

from __future__ import annotations

import torch


class LARS(torch.optim.Optimizer):
    """SGD with momentum and layer-wise adaptive rate scaling (You et al., 2017).

    A param group with `lars_exclude=True` (e.g. the online probe) is plain SGD + momentum.
    """

    def __init__(self, params, lr: float, momentum: float = 0.9, weight_decay: float = 0.0, eta: float = 0.001):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eta=eta,
                                      lars_exclude=False))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                update = p.grad
                if p.ndim > 1 and not group["lars_exclude"]:
                    update = update.add(p, alpha=group["weight_decay"])
                    p_norm, u_norm = p.norm(), update.norm()
                    trust = torch.where((p_norm > 0) & (u_norm > 0), group["eta"] * p_norm / u_norm,
                                        torch.ones_like(p_norm))
                    update = update.mul(trust)
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.clone(update).detach()
                else:
                    state["momentum_buffer"].mul_(group["momentum"]).add_(update)
                p.add_(state["momentum_buffer"], alpha=-group["lr"])
        return loss


def resolve_lr(optim_cfg: dict, batch_size: int) -> float:
    """Absolute `lr`, or `base_lr` scaled linearly by batch_size / 256."""
    if "lr" in optim_cfg:
        return optim_cfg["lr"]
    return optim_cfg["base_lr"] * batch_size / 256


def build_optimizer(model: torch.nn.Module, probe_params, cfg: dict) -> torch.optim.Optimizer:
    """Optimizer for the SSL model, plus a plain group for the online probe (lr 0.1 SGD-style,
    or 1e-3 for AdamW). The probe group never uses LARS scaling or weight decay.

    With `optim.fix_pred_lr: true` (SimSiam paper), the model's `predictor` gets its own
    group marked `fixed_lr`, which the training loop keeps at the base LR (no warmup/decay).
    """
    o = cfg["optim"]
    lr = resolve_lr(o, cfg["train"]["batch_size"])
    predictor = getattr(model, "predictor", None) if o.get("fix_pred_lr") else None
    pred_ids = {id(q) for q in predictor.parameters()} if predictor is not None else set()
    groups = [{"params": [q for q in model.parameters() if id(q) not in pred_ids]}]
    if predictor is not None:
        groups.append({"params": list(predictor.parameters()), "fixed_lr": True})
    probe_params = list(probe_params)
    if o["name"] == "adamw":
        if probe_params:
            groups.append({"params": probe_params, "lr": 1e-3, "weight_decay": 0.0})
        return torch.optim.AdamW(groups, lr=lr, weight_decay=o["weight_decay"])
    if probe_params:
        groups.append({"params": probe_params, "lr": 0.1, "weight_decay": 0.0, "lars_exclude": True})
    if o["name"] == "sgd":
        for g in groups:
            g.pop("lars_exclude", None)
        return torch.optim.SGD(groups, lr=lr, momentum=o["momentum"], weight_decay=o["weight_decay"])
    if o["name"] == "lars":
        return LARS(groups, lr=lr, momentum=o["momentum"], weight_decay=o["weight_decay"])
    raise ValueError(f"Unknown optimizer {o['name']!r}")
