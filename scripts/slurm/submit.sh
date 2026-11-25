#!/bin/bash
# Usage: scripts/slurm/submit.sh <manifest.txt> [max_parallel]
# Pulls nothing: run `git -C $ROOT/repo pull` first so jobs use the code you expect.
set -euo pipefail
MANIFEST=$(readlink -f "$1")
MAX=${2:-12}
ROOT=${ECE661_ROOT:-/usr/xtmp/$USER/ece661-lejepa}
N=$(grep -c . "$MANIFEST")
mkdir -p "$ROOT/logs"
cd "$ROOT/repo"
echo "commit $(git rev-parse --short HEAD), $N runs from $MANIFEST"
sbatch --array="1-${N}%${MAX}" --output="$ROOT/logs/%x_%A_%a.out" \
  --job-name="$(basename "$MANIFEST" .txt)" scripts/slurm/array.sbatch "$MANIFEST"
