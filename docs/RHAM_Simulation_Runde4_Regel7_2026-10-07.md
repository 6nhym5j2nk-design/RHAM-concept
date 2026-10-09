---
project: RHAM
repo: https://github.com/6nhym5j2nk-design/rham-project
date: 2026-10-07
tags: [rham, simulation, drift, rule-7, anisotropy, g-means]
mode: VERIFY
---
# RHAM – Simulation Round 4: Rule 7 and Anisotropic Noise (2026-10-07)

**Code:** `sim/rham_rule7.py` (class `OnlineRHAM7`; experiments `drift_*`, `aniso_*`, `gmeans_*`), `sim/diag_cap.py`, `sim/eval_e8.py`, `sim/eval_e8a.py`, `sim/plot_rule7.py`. Raw data: `sim/results_E8_*.json`. Figure: `sim/rham_runde4_ergebnisse.png`. Predecessor: [[Projects/RHAM/RHAM_Simulation_Runde3_Haertetest_2026-10-07]].

## Summary

1. **Rule 7 complete** (capped learning rate + age limit for episodes + retirement for prototypes + merging) holds accuracy for both frequent **and** rare concepts at 100% under strong drift and **halves memory** (600 vs. 1214 vectors). Growth comes to a halt, but at a level 2.2 times above the drift-free case.
2. **The sub-rules only work together.** The capped learning rate alone is *worse* than doing nothing under strong drift (frequent concepts 87.7% vs. 97.8%). Cause (diagnosed): outdated episodes in the buffer win the comparison against the current prototypes; only the age limit clears them out.
3. **Retirement does not hurt rare concepts:** same accuracy as without retirement; falling back on cold prototypes costs < 1 extra comparison per retrieval under weak drift.
4. **Split test under anisotropic noise:** there is no outright winner, but a trade-off between sensitivity and specificity. The iso null model wrongly splits elongated noise clouds (α = 8: 395 prototypes for 207 concepts) but detects overlap best (98.4%). G-means (with cross-fitting) rarely over-splits (226 prototypes) but detects overlap only moderately (94.2%).
5. **Remaining gap:** Under strong drift, the buffer stays at ≈ 290 episodes – 92% never consolidated, 75% from the rare half of the concepts.

---

## 1. Rule 7 – components

| Part | Mechanism | Biological counterpart (hypothesis) |
|---|---|---|
| 7a | capped learning rate (n ≤ 30): prototype = exponential moving average | plasticity is preserved |
| 7b | age limit: episodes folded in go to the archive after 3 sleep phases, even with poor reconstruction | decay of hippocampal traces after consolidation |
| 7c | retirement: a prototype with no assignment for 5 sleep phases → cold (not deleted); reactivated on a new assignment; retrieval falls back to cold prototypes when the best hot hit is < 0.80 | hard-to-access but preserved memory content |
| 7d | merging of mutually nearest prototypes whose combined episodes **fail** the split test (threshold lower than for splitting → no oscillation) | schema integration |

## 2. E8 Drift (20,000 episodes, 216 concepts, Zipf, 3 seeds)

Queries evaluated separately for frequent concepts (top 20%) and rare ones (bottom 50%).

| Drift | Variant | frequent, end | rare, end | frequent, mean | rare, mean | buffer | hot | cold | associative total | cost per retrieval |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | baseline | 1.000 | 1.000 | 0.998 | 0.955 | 0 | 216 | 0 | 266 | 30 |
| 0 | rule 7 | 1.000 | 1.000 | 0.999 | 0.955 | 0 | 212 | 4 | 279 | 28 |
| 0.2 | baseline | 0.999 | 0.978 | 0.997 | 0.950 | 424 | 320 | 0 | 895 | 455 |
| 0.2 | capped | 0.963 | 0.982 | 0.990 | 0.946 | 230 | 286 | 0 | 651 | 259 |
| 0.2 | **rule 7** | 0.984 | 0.999 | 0.993 | 0.949 | 148 | 233 | 53 | **481** | **186** |
| 0.3 | baseline | 0.978 | 0.981 | 0.987 | 0.939 | 419 | 543 | 0 | 1214 | 456 |
| 0.3 | capped | **0.877** | 0.967 | 0.970 | 0.940 | 374 | 508 | 0 | 1115 | 409 |
| 0.3 | **rule 7** | **1.000** | **1.000** | 0.996 | 0.953 | 279 | 241 | 267 | **600** | 366 |

(Drift 0.1: all variants ≈ equal, see table in `eval_e8.py`.) With rule 7 and drift 0.3, the number of hot prototypes (241) matches almost exactly the 216 true concepts; the 267 cold ones are outdated positions. Merges occurred in no run (concepts remain separate in this generator).

### Diagnosis "capped worse than baseline"
- Hypothesis 1 (label carry-over in prototypes) **refuted**: 40 of 52 errors come from the buffer, not from prototypes; the label of the most recent assignment is also wrong for the faulty prototypes.
- Hypothesis 2 **confirmed**: buffer episodes that lead to errors are on average 7.7–11.8 sleep phases old; those leading to hits are 1.1–2.3. With a capped learning rate, prototypes track the drift, so *young*, matching episodes get forgotten while *old*, no-longer-assignable ones remain and misdirect queries.

### Remaining buffer under strong drift (rule 7, seed 0)
293 episodes; 92% never folded into a prototype; median frequency rank 151 of 216; 75% from the rare half. Rare concepts keep drifting before three similar episodes accumulate for a prototype. An age limit also for never-folded episodes would bound the buffer but would remove exactly these rare concepts from hot memory – an open trade-off.

## 3. E8 Anisotropic noise and split test (8000 episodes, 3 seeds)

Noise per concept stretched by α along its own tangential direction, total variance held constant (mean cosine episode–center 0.93 as in the isotropic case; at α = 4, 18% of the noise variance lies along one of 64 directions). Four variants of the split test:
- **iso-null** (round 3): one concept + isotropic noise.
- **cov-null**: Gaussian with the data's estimated covariance.
- **G-means** (Hamerly & Elkan 2003, NeurIPS): project onto the axis of the 2-means centers, Anderson-Darling normality test (α = 0.0001). **Own addition: cross-fitting** (axis on half A, test on half B). Without it, the test also splits isotropic clouds in 64 dimensions (statistic +0.74 > 0): 2-means picks exactly the direction that looks randomly bimodal. With cross-fitting, over 20 repetitions each: isotropic 0%, elongated 0%, bimodal 100% splits.

**Separated concepts** (ideal: ≈ 207 prototypes):

| α | no split | iso-null | cov-null | G-means |
|---|---|---|---|---|
| 1 | 208 | 208 | 208 | 208 |
| 4 | 208 | **270** | 208 | 208 |
| 8 | 221 | **395** | 254 | 226 |

**Overlapping concepts** (sibling cosine 0.85), concept top-1 (oracle 1.000):

| α | no split | iso-null | cov-null | G-means |
|---|---|---|---|---|
| 1 | 0.924 | **0.993** | 0.939 | 0.957 |
| 4 | 0.936 | **0.990** | 0.943 | 0.950 |
| 8 | 0.927 | **0.984** | 0.928 | 0.942 |

**Reading:** No null model is simultaneously sensitive and specific. The iso null model confuses elongated noise clouds with two concepts. The cov-null model estimates covariance from data that is already merged and so absorbs the separating direction into the null model itself. G-means is the most specific but loses discriminative power by halving the sample via cross-fitting. For accuracy, over-splitting is almost harmless (majority label), but it costs memory and distorts the "concept count" – which matters for a memory meant to form abstractions.

## 4. Methodological notes

- **Scaling bug in the first draft** of the anisotropic generator (σ·√d instead of σ, eightfold too much noise); found while re-reading before the first run. Additionally pre-checked: mean cosine episode–center matched the isotropic case.
- **Selection bias in the G-means test** found via its own plausibility check (an isotropic cloud produced a positive statistic); fixed through cross-fitting, error rates then measured.
- **Own hypothesis refuted** (label carry-over) and replaced with a measured alternative hypothesis (age of buffer episodes).

## 5. Rule set v3

1. Distributed storage · 2. Associative retrieval (routing in residual space, exact final step) · 3. Dynamic hierarchy (interference + structure test) · 4. Offline consolidation · 5. Forgetting = demotion to the archive (reconstruction or age), tagged episodes excepted · 6. Splitting (null model open: iso = sensitive, G-means = specific) · 7. Prototype aging: capped learning rate + age limit + retirement + merging – **effective only together**.

## 6. Open points and next steps

1. Improve the split test: discriminative power of G-means without halving the sample (e.g. repeated cross-fitting with combined p-values) or two-stage (iso proposes, G-means confirms).
2. Rare concepts under drift: form prototypes starting from 2 episodes with a time window, or a separate "residual" buffer with its own age limit; measure the cost for rare concepts.
3. Real embeddings (text/image) before further fine-tuning on synthetic data – the risk grows of fitting the rule set to the generator rather than to reality.
4. Independent code review.
