# ECE661 Final Project — LeJEPA Section

Duke ECE661 (Fall 2026, NN training). Team of 3. This folder holds the LeJEPA part of the project.
Running decisions and status: [CONTEXT_LOG.md](CONTEXT_LOG.md).

## Project in one line

Is LeJEPA a good self-supervised backbone at student compute? Reproduce LeJEPA on CIFAR-10,
compare it with **SimCLR** and **SimSiam** under matched conditions, and sweep its SIGReg
weight λ. Scaling (Tiny-ImageNet, bigger backbone) is a stretch goal only.

Long-term motivation: JEPA-style latent world models (out of scope here).

## Proposal doc

Google Doc, tab "Proposal v2" (current):
https://docs.google.com/document/d/16BGKl3-JgQDaV_wYaT2do4YntHRFeka13ck8duEPvgc/edit?tab=t.j6cv5uvriwfo
Structured on the Heilmeier Catechism (required by the course). The older Claude doc
(https://claude.ai/artifact/Ne335CHVUdGhZpPPgfAK7A) describes a superseded MAE/data-scaling plan.

## Course constraints

- Grade: proposal 10%, poster 15%, presentation 15%, Q&A 20% (graded individually), report 40%.
- **Poster session, in person: Dec 9, 2026, 1–4pm.** About 6 min presenting, 3.5 min Q&A.
- Report: ≤ 6 pages excluding references, LaTeX template, code repo strongly recommended.
- Proposal and midterm TA check-in dates: not in the PDFs. Check Canvas.

## Experimental design (Proposal v2)

- **Exp 1, reproduce:** LeJEPA, ResNet-18, CIFAR-10, default settings.
- **Exp 2, matched comparison:** LeJEPA vs SimCLR vs SimSiam on CIFAR-10 and CIFAR-100; same
  backbone, augmentations, epochs, batch size; equal small sweep per method; 3 seeds (18 runs).
- **Exp 3, λ sweep:** CIFAR-100, ≥5 values incl. λ=0, 3 seeds; check best λ transfers to CIFAR-10.
- **Evaluation:** frozen linear probe (1/10/100% labels), k-NN, effective rank, GPU-hours,
  seed std, and correlation of LeJEPA training loss with probe accuracy.
- **Budget:** $250 compute credit; size epochs to ~80% of it after measuring cost per run.

## Code

- `main`: working code. SIGReg imported from the official `lejepa` package (pinned commit);
  `sslbench/methods/sigreg.py` is our own version, tested to match it.
- `to-be-implemented`: core functions stubbed, with `DIRECTIONS.md`, for learning.
- Runs on Colab via the VS Code extension: push, then pull inside `notebooks/colab_runner.ipynb`.
- Team roles, gates and branch rules: README.md.

## LeJEPA essentials

- Loss = `(1−λ)·prediction + λ·SIGReg`, with λ ≈ 0.05. The prediction term pulls each view's
  embedding toward the mean of the global-view embeddings. SIGReg pushes embeddings toward an
  isotropic Gaussian, using random 1-D projections and the Epps–Pulley test.
- No EMA teacher, stop-gradient, predictor asymmetry or schedulers.
- Uses multi-crop: 2 global + local views. Minimum batch size is about 128.
- Training loss tracks downstream accuracy, so it can be used for label-free model selection.
- Paper: arXiv 2511.08544. Code: https://github.com/rbalestr-lab/lejepa
