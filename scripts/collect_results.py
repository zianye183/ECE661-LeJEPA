"""Gather result JSONs into one CSV and print mean ± std over seeds.

  python scripts/collect_results.py results/            # or a Google Drive folder
  python scripts/collect_results.py results/ --out results/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

KEYS = ["experiment", "method", "dataset", "seed", "epochs", "train_time_s", "knn_top1", "linear_top1_1pct",
        "linear_top1_10pct", "linear_top1_100pct", "effective_rank", "final_train_loss"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    rows = []
    for path in sorted(args.root.rglob("*.json")):
        data = json.loads(path.read_text())
        if "method" in data and "knn_top1" in data:
            row = {k: data.get(k) for k in KEYS}
            row["lamb"] = data.get("config", {}).get("method", {}).get("lamb")
            rows.append(row)
    if not rows:
        raise SystemExit(f"No result JSONs found under {args.root}")

    if args.out:
        with open(args.out, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {args.out}")

    groups = defaultdict(list)
    for r in rows:
        groups[(r["experiment"], r["method"], r["dataset"], r["lamb"])].append(r)
    metrics = ["knn_top1", "linear_top1_100pct", "linear_top1_10pct", "linear_top1_1pct", "effective_rank"]
    print("experiment | method | dataset | lambda | n | " + " | ".join(metrics))
    for key, rs in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        cells = []
        for m in metrics:
            vals = [r[m] for r in rs if r[m] is not None]
            if not vals:
                cells.append("-")
            elif len(vals) == 1:
                cells.append(f"{vals[0]:.4g}")
            else:
                cells.append(f"{statistics.mean(vals):.4g} ± {statistics.stdev(vals):.2g}")
        print(" | ".join(str(k) for k in key) + f" | {len(rs)} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
