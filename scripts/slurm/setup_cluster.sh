#!/bin/bash
# One-time setup on the Duke CS cluster login node.
# Usage: PYTHON=/path/to/python3.11 bash scripts/slurm/setup_cluster.sh
set -euo pipefail
ROOT=${ECE661_ROOT:-/usr/xtmp/$USER/ece661-lejepa}
PYTHON=${PYTHON:-python3.11}
BRANCH=${BRANCH:-main}
mkdir -p "$ROOT"/{data,runs,results,logs}
[ -d "$ROOT/repo" ] || git clone -b "$BRANCH" https://github.com/zianye183/ECE661-LeJEPA.git "$ROOT/repo"
[ -d "$ROOT/venv" ] || "$PYTHON" -m venv "$ROOT/venv"
source "$ROOT/venv/bin/activate"
pip install -q --upgrade pip
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -q -e "$ROOT/repo[dev]"
python -c "from torchvision import datasets as d; r='$ROOT/data'; d.CIFAR10(r, download=True); d.CIFAR10(r, train=False, download=True); d.CIFAR100(r, download=True); d.CIFAR100(r, train=False, download=True)"
echo "setup done in $ROOT"
