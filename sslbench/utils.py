"""Config loading, seeding, devices, schedules and result files."""

from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict with `override` merged into `base` recursively."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def parse_override(text: str) -> dict:
    """Turn 'train.epochs=5' into {'train': {'epochs': 5}} (value parsed as YAML)."""
    if "=" not in text:
        raise ValueError(f"Override must look like key.sub=value, got {text!r}")
    dotted, raw = text.split("=", 1)
    value = yaml.safe_load(raw)
    result: dict = {}
    node = result
    keys = dotted.split(".")
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value
    return result


def load_config(method: str, extra_files: list[str] = (), overrides: list[str] = ()) -> dict:
    """shared.yaml <- methods/<method>.yaml <- extra files <- key=value overrides."""
    files = [REPO_ROOT / "configs" / "shared.yaml", REPO_ROOT / "configs" / "methods" / f"{method}.yaml"]
    files += [Path(f) for f in extra_files]
    cfg: dict = {}
    for path in files:
        with open(path) as fh:
            cfg = deep_merge(cfg, yaml.safe_load(fh) or {})
    for text in overrides:
        cfg = deep_merge(cfg, parse_override(text))
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def warmup_cosine(step: int, warmup_steps: int, total_steps: int) -> float:
    """LR multiplier: linear warmup to 1, then cosine decay to 0."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(record) + "\n")
