"""Pretrain one SSL method, then run the shared frozen-backbone evaluation.

Examples:
  python train.py --method lejepa
  python train.py --method simclr --set data.dataset=cifar100 --seed 1
  python train.py --method lejepa --config configs/smoke.yaml          # quick check
  python train.py --method lejepa --set method.lamb=0.2 experiment=exp3_lambda
  python train.py --method simsiam --runs-dir /content/drive/MyDrive/ece661/runs

Outputs:
  <runs-dir>/<experiment>/<method>/<dataset>/s<seed>/   config.yaml, metrics.jsonl, last.pt
  <results-dir>/<experiment>/<method>/<dataset>/s<seed>.json   final metrics (commit these)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch import nn

from sslbench.data import NUM_CLASSES, eval_datasets, pretrain_loader
from sslbench.eval import evaluate
from sslbench.methods import build_method
from sslbench.utils import append_jsonl, load_config, pick_device, set_seed, warmup_cosine, write_json


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", required=True, choices=["lejepa", "simclr", "simsiam"])
    p.add_argument("--config", action="append", default=[], help="extra YAML merged on top (repeatable)")
    p.add_argument("--set", dest="overrides", action="append", default=[], help="key.sub=value override")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--runs-dir", default="runs", help="checkpoints and logs (put on Google Drive in Colab)")
    p.add_argument("--results-dir", default="results", help="final metric JSONs")
    p.add_argument("--resume", action="store_true", help="continue from last.pt in the run dir")
    p.add_argument("--skip-eval", action="store_true")
    return p.parse_args(argv)


def build_optimizer(params, cfg: dict) -> torch.optim.Optimizer:
    o = cfg["optim"]
    if o["name"] == "adamw":
        return torch.optim.AdamW(params, lr=o["lr"], weight_decay=o["weight_decay"])
    if o["name"] == "sgd":
        return torch.optim.SGD(params, lr=o["lr"], momentum=o["momentum"], weight_decay=o["weight_decay"])
    raise ValueError(f"Unknown optimizer {o['name']!r}")


def main(argv=None) -> dict:
    args = parse_args(argv)
    overrides = args.overrides + ([f"seed={args.seed}"] if args.seed is not None else [])
    cfg = load_config(args.method, args.config, overrides)
    set_seed(cfg["seed"])
    device = pick_device()
    dataset = cfg["data"]["dataset"]
    num_classes = NUM_CLASSES[dataset]
    run_name = Path(cfg["experiment"]) / args.method / dataset / f"s{cfg['seed']}"
    run_dir = Path(args.runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    loader = pretrain_loader(cfg)
    model = build_method(cfg).to(device)
    # Online probe: trained on detached features, never affects the backbone. Monitoring only.
    probe = nn.Linear(model.feat_dim, num_classes).to(device) if cfg["train"]["online_probe"] else None
    params = [{"params": model.parameters()}]
    if probe is not None:
        params.append({"params": probe.parameters(), "lr": 1e-3, "weight_decay": 0.0})
    opt = build_optimizer(params, cfg)
    if probe is not None and cfg["optim"]["name"] == "sgd":
        opt.param_groups[1]["lr"] = 0.1  # plain SGD needs a larger probe LR than AdamW
    base_lrs = [g["lr"] for g in opt.param_groups]

    epochs = cfg["train"]["epochs"]
    total_steps = epochs * len(loader)
    warmup_steps = cfg["train"]["warmup_epochs"] * len(loader)
    use_amp = cfg["train"]["amp"] and device.type == "cuda"
    start_epoch, step = 0, 0
    ckpt_path = run_dir / "last.pt"
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        if probe is not None:
            probe.load_state_dict(ckpt["probe"])
        start_epoch, step = ckpt["epoch"] + 1, ckpt["step"]
        print(f"Resumed from epoch {start_epoch}")

    use_wandb = cfg["logging"]["wandb"]
    if use_wandb:
        import wandb

        wandb.init(project=cfg["logging"]["wandb_project"], name=str(run_name), config=cfg)

    t0 = time.time()
    record: dict = {}
    for epoch in range(start_epoch, epochs):
        model.train()
        sums: dict[str, float] = {}
        for views, y in loader:
            mult = warmup_cosine(step, warmup_steps, total_steps)
            for group, lr in zip(opt.param_groups, base_lrs):
                group["lr"] = lr * mult
            views, y = views.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss, logs, feats = model(views)
                if probe is not None:
                    logits = probe(feats)
                    probe_loss = F.cross_entropy(logits.float(), y)
                    logs["probe_loss"] = probe_loss.item()
                    logs["probe_acc"] = (logits.argmax(1) == y).float().mean().item()
                    loss = loss + probe_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            for key, val in logs.items():
                sums[key] = sums.get(key, 0.0) + val
        record = {"epoch": epoch, "step": step, "lr": opt.param_groups[0]["lr"],
                  "time_s": round(time.time() - t0, 1),
                  **{k: v / len(loader) for k, v in sums.items()}}
        if device.type == "cuda":
            record["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        append_jsonl(run_dir / "metrics.jsonl", record)
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()))
        if use_wandb:
            wandb.log({f"train/{k}": v for k, v in record.items()})
        if (epoch + 1) % cfg["train"]["save_every"] == 0 or epoch + 1 == epochs:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "probe": probe.state_dict() if probe is not None else None,
                        "epoch": epoch, "step": step, "config": cfg}, ckpt_path)

    train_time = time.time() - t0
    if args.skip_eval:
        return {}
    train_ds, test_ds = eval_datasets(cfg)
    metrics = evaluate(model, train_ds, test_ds, num_classes, cfg["eval"], device,
                       num_workers=cfg["data"]["num_workers"])
    final_loss = {f"final_train_{k}": v for k, v in record.items() if k not in ("epoch", "step", "lr", "time_s")}
    result = {"method": args.method, "dataset": dataset, "seed": cfg["seed"], "experiment": cfg["experiment"],
              "epochs": epochs, "train_time_s": round(train_time, 1), "device": str(device),
              **final_loss, **metrics, "config": cfg}
    write_json(Path(args.results_dir) / run_name.with_suffix(".json"), result)
    write_json(run_dir / "results.json", result)
    print({k: v for k, v in result.items() if k != "config"})
    if use_wandb:
        wandb.log({f"eval/{k}": v for k, v in metrics.items()})
        wandb.finish()
    return result


if __name__ == "__main__":
    main()
