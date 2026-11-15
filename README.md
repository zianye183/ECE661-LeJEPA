# ECE661: LeJEPA as a Self-Supervised Backbone

Reproduce LeJEPA on CIFAR-10, compare it with SimCLR and SimSiam under matched conditions,
and sweep its SIGReg weight λ. Proposal: "Proposal v2" tab of the team Google Doc.

## Layout

```
configs/
  shared.yaml            # data, augmentations, backbone, epochs, eval: shared by ALL methods
  methods/{lejepa,simclr,simsiam}.yaml
  smoke.yaml             # tiny fake-data run for checks
sslbench/
  data.py                # CIFAR loaders, shared augmentations, class-balanced nested subsets
  backbones.py           # ResNet-18 with CIFAR stem (timm:<name> for the stretch goal)
  methods/
    base.py              # SSLMethod interface: forward(views) -> (loss, logs, feats), embed(x)
    lejepa.py            # LeJEPA loss; SIGReg from the official `lejepa` package
    sigreg.py            # SIGReg / Epps-Pulley written out; tested to match the official one
    simclr.py, simsiam.py
  eval.py                # k-NN, linear probes (1/10/100% labels), effective rank
train.py                 # pretrain + evaluate one run
scripts/collect_results.py
notebooks/colab_runner.ipynb
tests/
```

**What comes from upstream:** SIGReg (Epps-Pulley test + random slicing) is imported from
the official repo, [rbalestr-lab/lejepa](https://github.com/rbalestr-lab/lejepa), pinned to
commit `c293d29` in `pyproject.toml`. That repo ships the loss library only. The training
loop follows its `MINIMAL.md` (projector MLP, `(1-λ)·pred + λ·SIGReg`, online probe) adapted
to ResNet-18 + CIFAR, using the paper's Algorithm 2 rule for ResNets: every view is a global
view. Upstream license is CC BY-NC 4.0 (fine for coursework).

## Setup

```bash
pip install -e ".[dev]"
python -m pytest -q          # ~2 min on CPU; includes 1-epoch runs of all 3 methods on fake data
```

## Running

```bash
python train.py --method lejepa                                   # defaults: CIFAR-10, seed 0
python train.py --method simclr --seed 1 --set data.dataset=cifar100 --set experiment=exp2
python train.py --method lejepa --set method.lamb=0.2 --set experiment=exp3_lambda
python scripts/collect_results.py results/
```

Every run writes `runs/<experiment>/<method>/<dataset>/s<seed>/` (config, per-epoch
`metrics.jsonl`, `last.pt`) and `results/<experiment>/<method>/<dataset>/s<seed>.json`
(final metrics). `--resume` continues from `last.pt` after a Colab disconnect.

## Colab

`notebooks/colab_runner.ipynb` does everything in one go: mounts Google Drive, pulls this
repo, runs the tests, measures cost per epoch, trains all three methods on CIFAR-10 and
writes results, a summary CSV and training curves to `MyDrive/ECE661/lejepa`. Open it in
Colab (or VS Code with the Colab extension), pick a GPU runtime and *Run all*. After a
disconnect, *Run all* again: finished runs are skipped and unfinished ones resume.

The notebook runs code from GitHub, not your local checkout: push before re-running.

## Team workflow

| Person | Owns | Deliverable |
|---|---|---|
| A: infra & eval | `data.py`, `eval.py`, `train.py`, `configs/shared.yaml`, analysis | Shared pipeline, cost probe, figures, poster lead |
| B: LeJEPA | `methods/lejepa.py`, `methods/sigreg.py`, `configs/methods/lejepa.yaml` | Exp 1 reproduction, Exp 3 λ sweep |
| C: baselines | `methods/simclr.py`, `methods/simsiam.py`, their configs | Exp 2 matched comparison |

Fill in `.github/CODEOWNERS` with your GitHub usernames.

**Gates:** G1 skeleton (this commit) → G2 all methods pass a short CIFAR run and cost per run
is measured, then epochs are locked → G3 all runs done → G4 poster (Dec 9).

**Rules**
- `main` is protected: PRs only, one approval from a teammate, tests green.
- Branch names: `infra/<topic>`, `lejepa/<topic>`, `baselines/<topic>`, `exp/<name>`.
  Rebase on `main` before opening a PR; keep PRs small and merge every 1–2 days.
- After G2, changing `configs/shared.yaml` needs all three approvals.
- Commit configs and result JSONs, never checkpoints or data. Tag the commit used for final
  runs (`git tag exp2-final`).

## Branches

- `main`: working implementation, use this for experiments.
- `to-be-implemented`: same repo with the core algorithms removed, plus `DIRECTIONS.md`, for
  learning to implement them yourself. The tests are the grader.
