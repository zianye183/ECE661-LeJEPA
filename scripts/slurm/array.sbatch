#!/bin/bash
# One training run per array task; task N runs line N of the manifest.
# Submit through scripts/slurm/submit.sh, which sets --array and the log path.
#SBATCH --job-name=lejepa
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=12:00:00
set -euo pipefail
MANIFEST=$1
ROOT=${ECE661_ROOT:-/usr/xtmp/$USER/ece661-lejepa}
ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")
source "$ROOT/venv/bin/activate"
cd "$ROOT/repo"
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) commit=$(git rev-parse --short HEAD)"
echo "args: $ARGS"
# shellcheck disable=SC2086
python train.py $ARGS --runs-dir "$ROOT/runs" --results-dir "$ROOT/results" \
  --set data.root="$ROOT/data" --set data.num_workers=5 --resume
