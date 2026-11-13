# Directions: implement the core yourself

This branch (`to-be-implemented`) is `main` with the core algorithms removed. Everything
else is already done: data, augmentations, backbone, training loop, linear probe and configs.
You write the parts that make each method what it is, and the tests tell you when you're done.

**Ground rules**
- Don't open `main`'s versions of these files (`git diff main -- sslbench` shows the answers).
  Look only after your tests pass, to compare.
- Work on your own branch off this one, e.g. `git switch -c learn/<your-name>`. Never merge
  this branch into `main`: experiments run on `main`.
- Shapes are the main thing to get right. Write them as comments next to each line.

## Setup

```bash
pip install -e ".[dev]"                   # also installs the official `lejepa` package
python -m pytest -q tests/test_sigreg.py  # fails until Task 1 is done
```

## Task order

```
Task 1  SIGReg (Epps-Pulley)   ──>  Task 2  LeJEPA loss + forward  ──┐
Task 3  SimCLR + SimSiam losses (independent)                         ├──> Task 5  train it
Task 4  k-NN + effective rank  (independent)                          ┘
```

Suggested split: B (LeJEPA) does Tasks 1–2, C (baselines) does Task 3, A (infra) does Task 4.
Then each person does one task they didn't own; Q&A is graded individually.

---

## Task 1: SIGReg with the Epps-Pulley test (`sslbench/methods/sigreg.py`)

**Read:** paper (arXiv 2511.08544) Section 4: the Epps-Pulley equation and Algorithm 1.
The proposal's "Background" section has the same formula.

SIGReg asks "do these embeddings look like samples from an isotropic Gaussian?" Testing that
directly in D dimensions is hard, so it checks many random 1-D projections instead: if every
1-D projection is N(0, 1), the whole distribution is N(0, I) (Cramér–Wold).

**1a. `epps_pulley(x, t_max, n_points)`**. `x` is `(..., N, K)`: N samples for each of K
projections. For each column, compare the *empirical characteristic function*
`ecf(t) = mean_n exp(i t x_n)` with the standard normal's `phi(t) = exp(-t²/2)`:

```
T = N · ∫ w(t) |ecf(t) − phi(t)|² dt,   w(t) = exp(−t²/2)
```

- No complex numbers needed: `exp(i t x) = cos(t x) + i sin(t x)`, and phi is real, so
  `|ecf − phi|² = (mean cos − phi)² + (mean sin)²`.
- The integrand is symmetric in t. Integrate over `[0, t_max]` with `n_points` evenly spaced
  points and the trapezoid rule, then double the weights. Trapezoid weights are `dt` for
  interior points and `dt/2` at both ends, so after doubling: `2·dt` inside and `dt` at the ends.
  Fold `w(t)` into the weights.
- Broadcasting: `x.unsqueeze(-1) * t` gives `(..., N, K, T)`. Average over the N axis (which
  axis number is it now?), and finish with a matrix product against the weights.

**1b. `random_directions(dim, num_slices, seed, device)`**. Unit-norm Gaussian columns. Use
exactly the recipe in the TODO so your numbers match the official package.

**1c. `SIGReg.forward(z)`**. Fresh directions each call (seed = `self.step`, then increment),
project `z @ A`, apply `epps_pulley`, average to a scalar. The directions need no gradient.

**Done when:** `python -m pytest -q tests/test_sigreg.py` passes, including
`test_sigreg_matches_official_lejepa_package`.

**Check you understand:**
- Why multiply by N? What happens to T for Gaussian data as N grows, and for non-Gaussian data?
- Why do we draw new random directions every step instead of fixing them?
- What does SIGReg return for fully collapsed embeddings (all identical)? Why is that what
  prevents collapse?

## Task 2: the LeJEPA objective (`sslbench/methods/lejepa.py`)

**Read:** paper Section 5, Eq. 5 (prediction loss) and Algorithm 2. Note the line "for
non-ViT architectures (e.g., ResNet), set global_views = all_views".

**2a. `lejepa_loss(proj, sigreg, lamb)`**. `proj` is `(V, N, K)`.
- Prediction loss: each view should predict the *center*, the mean over that image's views.
  Mean squared distance from each view's projection to its center, averaged over everything.
- SIGReg loss: `sigreg(proj)`. It treats V as a batch dim and averages the per-view statistics.
- `loss = (1 − λ)·pred + λ·sigreg`. Return all three.

**2b. `LeJEPA.forward(views)`**. `views` is `(N, V, C, H, W)`.
- Run all `N·V` images through the backbone in **one** batch (`flatten(0, 1)`), then the
  projector. The result is image-major: rows `0..V-1` are image 0's views.
- Reshape to `(N, V, K)`, then transpose to `(V, N, K)`.
- Compute the loss in fp32 with autocast disabled (`torch.autocast(..., enabled=False)` and
  `.float()`): cos/sin of bf16 values is too coarse.
- Return `(loss, {"loss", "pred", "sigreg"} as floats, backbone features of view 0 detached)`.

**Done when:** `python -m pytest -q tests/test_losses.py -k lejepa` passes and
`python -m pytest -q tests/test_train_smoke.py -k lejepa` runs one epoch end to end.

**Check you understand:**
- With λ = 0, what is the trivial minimizer of the loss? (Exp 3 tests exactly this.)
- No stop-gradient, no EMA teacher, no predictor: what replaces them?
- Why do SimCLR and SimSiam not need this view-0 bookkeeping for multiple views?

## Task 3: baseline losses (`simclr.py`, `simsiam.py`)

**3a. `nt_xent(z1, z2, temperature)`** (SimCLR paper, Section 2.1).
- L2-normalize, stack into `2N` rows, cosine similarity matrix `/ temperature`.
- A sample must not be its own negative: mask the diagonal with `-inf` (use `masked_fill`).
- Row `i`'s positive is `i + N` for the first half and `i − N` for the second. Then the loss
  is just `F.cross_entropy(logits, targets)`.

**3b. `simsiam_loss(p1, p2, z1, z2)`** (SimSiam paper, Algorithm 1).
- Negative cosine similarity between predictor output `p` and the *other* view's projection
  `z`, symmetrized: `−½ [cos(p1, z2) + cos(p2, z1)]`, batch-averaged.
- The stop-gradient goes on `z` (`.detach()`). The test checks that no gradient reaches `z`.

**Done when:** `python -m pytest -q tests/test_losses.py` passes.

**Check you understand:**
- What's the minimum possible NT-Xent value with N pairs, and the value when everything has
  collapsed? (One test checks the second.)
- Remove the `.detach()` in SimSiam and train the smoke run: what happens to `z_std` in the
  logs? Why?

## Task 4: evaluation metrics (`sslbench/eval.py`)

**4a. `knn_accuracy(...)`**: weighted k-NN (Wu et al., 2018, the usual SSL k-NN monitor).
- Cosine similarity: L2-normalize both sets, then `test @ train.T`. Do it in chunks of test
  points: the full CIFAR matrix is 10k × 50k.
- Take the top-k neighbours. Each votes for its label with weight `exp(sim / temperature)`
  (`scatter_add_` into a `(B, num_classes)` tensor), and the prediction is the argmax.

**4b. `effective_rank(feats)`**: RankMe (Garrido et al., 2023).
- Center the features, compute singular values `s` (`torch.linalg.svdvals`, float64),
  normalize `p = s / sum(s)` (add a tiny eps), and return `exp(−Σ p log p)`.
- 1 means complete collapse; D means all directions are equally used.

**Done when:** `python -m pytest -q tests/test_eval.py` passes.

**Check you understand:** why is effective rank a collapse check that needs no labels, and
which of our three methods would you expect to score highest on it?

## Task 5: train it

```bash
python -m pytest -q                                            # everything green
python train.py --method lejepa --config configs/smoke.yaml    # fake data, ~30 s on CPU
```

Then on Colab (`notebooks/colab_runner.ipynb`, with `BRANCH` set to your learning branch),
do a short real run and compare against the same run on `main`:

```bash
python train.py --method lejepa --set train.epochs=5 --set experiment=learn
```

With the same seed and config, your SIGReg matches the official one, so both runs should
give nearly identical losses.
