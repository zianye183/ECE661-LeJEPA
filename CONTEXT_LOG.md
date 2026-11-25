# Context Log

Newest first. Decisions, status and open questions only.

## 2026-11-25

**Decided**
- Run experiments on the Duke CS Slurm cluster (`compsci-gpu`, A5000), not Colab. Setup lives in
  `/usr/xtmp/zy223/ece661-lejepa`; job arrays via `scripts/slurm/submit.sh <manifest>`.
- G2: epochs locked at **200**, batch 256, for all methods. A5000 cost: LeJEPA ~36 s/epoch,
  SimCLR/SimSiam ~26 s/epoch (first epoch incl. startup).
- Exp 3 λ grid: {0, 0.01, 0.02, 0.1, 0.2} on CIFAR-100 × 3 seeds; λ=0.05 comes from Exp 2.

**Status**
- First Colab baselines (CIFAR-10, seed 0, 100 epochs): LeJEPA 88.5%, SimCLR 83.6%,
  SimSiam 74.7% linear probe. SimSiam re-run with paper-aligned fixes running on the cluster.
- Colab SimSiam "v2" run had silently used old code; jobs now log the git commit they run.

**Open**
- After Exp 3: best λ re-run on CIFAR-10 × 3 seeds (transfer check).

## 2026-11-12

**Decided**
- Plan follows Proposal v2 (Google Doc): LeJEPA vs SimCLR vs SimSiam on ResNet-18/CIFAR,
  plus a λ sweep. Supersedes the MAE/data-scaling design below.
- Roles: A infra & eval, B LeJEPA (Exp 1, Exp 3), C baselines (Exp 2). Gates G1–G4 in README.
- Use the official `lejepa` package for SIGReg rather than re-implementing it; keep a
  `to-be-implemented` branch for learning.

**Status**
- Repo scaffolded (G1). Tests pass on CPU with fake data. Not yet run on real CIFAR or a GPU.

**Open**
- GitHub remote and CODEOWNERS usernames. Number of LeJEPA views vs 2 for baselines (fairness).

## 2026-10-22

**Decided**
- Topic: data-scaling comparison of LeJEPA vs SimCLR vs MAE, built on course Project 4.
- The comparison is joint-embedding vs reconstruction. SimCLR is itself joint-embedding, so MAE
  is the needed contrast.
- Proposal drafted in the Claude doc (link in CLAUDE.md).
