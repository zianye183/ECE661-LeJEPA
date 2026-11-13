# Context Log

Newest first. Decisions, status and open questions only.

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
