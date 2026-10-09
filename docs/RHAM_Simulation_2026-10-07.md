---
project: RHAM
repo: https://github.com/6nhym5j2nk-design/rham-project
date: 2026-10-07
tags: [rham, simulation, hopfield, results]
mode: VERIFY
---
# RHAM – Minimal Simulation, Results (2026-10-07)

**Code:** `sim/rham_sim.py` (experiments E1–E4), `sim/diag_routing.py`, `sim/diag_v3.py`, `sim/plot_results.py`. Raw data: `sim/results_*.json`, `sim/results_v3.txt`. Figure: `sim/rham_sim_ergebnisse.png`.

**Setup:** Synthetic tree data: each leaf is the sum of random unit vectors along its path (branching $K$, depth $h$, $d=64$), normalized onto the unit sphere. Siblings have cosine $\approx (h-1)/h$, i.e. strongly correlated patterns (the realistic case). Queries are noisy leaves (noise norm $\eta$). 5 seeds × 300 queries per condition. Flat memory = one Hopfield update step with softmax attention (Ramsauer et al. 2020). RHAM = learned prototypes (k-means) per level, top-down retrieval with beam search.

**Two metrics** (first treated separately, see Section 5):
- *Top-1 hit*: the correct pattern has the highest weight.
- *Sharp retrieval*: top-1 **and** weight > 0.9 (the state has converged onto a single pattern, no mixed state).

---

## 1. Core results

| # | Prediction (feasibility analysis §10) | Result | Verdict |
|---|---|---|---|
| P1 | RHAM has higher retrieval capacity at the same budget | **Only at a fixed, bounded β.** With freely chosen β, the flat memory reaches the nearest-neighbor ceiling; RHAM cannot exceed it | **partially refuted, refined** |
| P2 | Number of levels saturates at the data depth | Levels form **until interference is resolved** – that is ≤ data depth (h=4: 4/4 exact; h=5: 4 instead of 5, because the topmost level no longer has interference) | **confirmed, refined** |
| P3 | No new level on noise data | 20/20 runs: no level. Interference alone would have wrongly created levels in 15/20; the structure check (gap statistic) prevents that | **confirmed** |
| P4 | Marginal storage → entropy rate | not tested | open |
| – | Selective forgetting by reconstructibility beats random forgetting | 50% forgotten: mean fidelity 0.949 vs. 0.897; worst case 0.83 vs. 0.32 | **confirmed** |

## 2. E1 – Retrieval vs. memory size (noise η = 0.35)

| N | NN ceiling | flat β=16 sharp | RHAM β=16 sharp | flat top-1 | RHAM top-1 | cost flat | cost RHAM |
|---|---|---|---|---|---|---|---|
| 64 | 1.00 | 0.995 | 0.995 | 1.00 | 0.997 | 64 | 20 |
| 512 | 1.00 | 0.845 | 0.967 | 1.00 | 0.997 | 512 | 40 |
| 1000 | 1.00 | 0.679 | 0.979 | 1.00 | 0.997 | 1000 | 50 |
| 4096 | 1.00 | **0.259** | **0.928** | 1.00 | 0.999 | 4096 | **81** |

**Reading:** The flat memory picks the correct pattern almost every time but **does not converge sharply** – at fixed β it gets stuck in a metastable mixed state (exactly the regime Ramsauer et al. describe for correlated patterns). RHAM retrieves sharply and, at N = 4096, needs roughly **50 times fewer dot products** to do so; cost grows ≈ logarithmically. With freely chosen β (up to 256), the flat memory also reaches 100% sharp – at the price of a very steep energy potential and linear cost.

## 3. E1b/E1c – Noise and beam width (N = 1000)

| η | NN ceiling | RHAM v2 top-1 (best β) | RHAM v3, beam 1 | v3, beam 2 | v3, beam 4 |
|---|---|---|---|---|---|
| 0.35 | 1.000 | 0.998 | 0.997 | 0.997 | 0.997 |
| 1.0 | 0.995 | 0.973 | 0.954 | 0.973 | 0.978 |
| 1.6 | 0.807 | 0.674 | 0.603 | 0.665 | 0.698 |
| 2.0 | 0.550 | 0.417 | 0.368 | 0.439 | 0.453 |

**Reading:** Under strong noise, RHAM loses ground relative to the exact nearest-neighbor comparison (up to ≈ 10 percentage points). This is **not an implementation detail but fundamental**: a hierarchy makes early, coarse decisions on noisy information and can only approximate a full comparison, never exceed it. A wider beam narrows the gap (cost grows linearly in the beam).

## 4. E3 – Automatic level growth

Rule: new level when interference $I(M_n) > 0.2$ (fraction of patterns with Ramsauer error bound $>0.05$) **and** gap statistic > 0.15 (Tibshirani, Walther & Hastie 2001, *JRSS B*; null model: uniform on the sphere).

| Data | N | target | found (5 seeds) |
|---|---|---|---|
| Tree K=8, h=2 | 64 | 2 | 2, 3, 3, 2, 2 |
| Tree K=6, h=3 | 216 | 3 | 3, 4, 3, 3, 4 |
| Tree K=4, h=4 | 256 | 4 | 4, 4, 4, 4, 4 |
| Tree K=3, h=5 | 243 | 5 | 4, 4, 4, 4, 4 |
| i.i.d., d=64 | 256 / 1024 | 1 | all 1 |
| i.i.d., d=8 | 256 / 1024 | 1 | all 1 |

**Reading:** (a) The rule builds levels **for as long as they are needed**, not as many as the data "would warrant": at h = 5, the 9 prototypes of the third level are already well separated, so a further level would be useless. (b) Occasional over-segmentation (+1 level) arises when k-means picks a slightly wrong number of clusters (candidate grid; e.g. 12 instead of 8). (c) **Interference alone is an insufficient trigger**: i.i.d. data in low dimension or large N has interference ≈ 1 but no structure. The structure check is necessary (confirms red-team point 4 of the feasibility analysis).

## 5. Methodological findings during the simulation

1. **Bug: NaN keys** for singleton clusters (residual = 0 → division by zero). Discovered via RuntimeWarnings in the log and conspicuous outliers at K = 4 (88%). Fixed through safe normalization; outliers disappeared (98–100%).
2. **Design flaw v1: shared softmax over residuals from different parents.** Discovered because a wider beam *lowered* accuracy (0.59 → 0.42 at η = 1.6). Diagnosis (`diag_routing.py`): routing alone was not the bottleneck. Cause: residuals normalized relative to *different* prototypes are not comparable. Fix v2: **hierarchical softmax** (normalization per sibling group, path probabilities multiplied).
3. **Measurement artifact: sharpness ≠ hit.** v2 appeared worse because the multiplied path probabilities honestly fall below 0.9 under uncertainty. Hence the separation of top-1 and sharp retrieval. v2 is **better calibrated**, not worse.
4. **Noise amplification in residual space.** The per-level own space ($q-p$ normalized) amplifies noise by $\|x\|/\|x-p\|$. Fix v3: routing in residual space, final step with exact decomposition $q\cdot x = q\cdot p + q\cdot(x-p)$. v3 is the recommended design: beam helps monotonically, top-1 = sharp retrieval.

## 6. Consequences for the RHAM hypothesis

- **Correction of the core thesis:** The hierarchy's advantage is **not** higher capacity in the sense of "more correctly retrievable patterns" – with unbounded β and unbounded compute, the flat memory is equally good or better. The advantage is (a) **sharp, stable retrieval at bounded inverse temperature** (realistic for trainable systems and hardware with limited dynamic range) and (b) **logarithmic instead of linear retrieval cost**. This matches feasibility analysis §7: the main argument is energy per retrieval, not storage amount.
- **Price:** Under very noisy cues, accuracy is lost. Biologically plausible (weak cue → wrong "folder"), technically controllable via beam width.
- **Growth rule:** The two-criterion trigger (interference + structure) works and produces the *needed*, not the maximum, depth.
- **Forgetting:** The reconstruction rule is clearly better than random. Combined with the three-layer model (the archive retains raw data), the loss of fidelity is only a loss of direct associative access.

## 7. Limitations of this simulation

Synthetic, perfectly hierarchical data with isotropic noise; only one Hopfield step; prototypes via k-means instead of learned; fixed cluster count in E1 (oracle), automatic only in E3; no real data (text/image embeddings); P4 not tested; hyperparameters (θ = 0.2, gap threshold 0.15) not systematically varied. Results are feasibility evidence, not a performance proof.

## 8. Next steps

1. Adopt v3 as the default in `rham_sim.py` and fully recompute E1/E1b with it.
2. Real embeddings (e.g. sentence or image embeddings from a public dataset) instead of synthetic trees.
3. Online operation: feed episodes one at a time, growth and forgetting in a running stream; measure P4.
4. Sensitivity analysis for θ, gap threshold, beam.
5. Independent code review (skill "code-review-and-sandbox") before any publication.
