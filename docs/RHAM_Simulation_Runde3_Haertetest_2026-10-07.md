---
project: RHAM
repo: https://github.com/6nhym5j2nk-design/rham-project
date: 2026-10-07
tags: [rham, simulation, stress-test, overlap, drift, rare-episodes]
mode: VERIFY
---
# RHAM – Simulation Round 3: Stress Test (2026-10-07)

**Code:** `sim/rham_hard.py` (class `OnlineRHAM` with archive, split rule, important-tagging; scenarios S1, S2, S3), `sim/plot_hard.py`. Raw data: `sim/results_E7_*.json`. Figure: `sim/rham_haertetest_ergebnisse.png`. Predecessor: [[Projects/RHAM/RHAM_Simulation_Runde2_2026-10-07]].

Base setup as in E5: 216 concepts (K = 6, h = 3, d = 64), Zipf frequencies, episodes = concept + noise (norm 0.4), sleep phase every 1000 episodes, 3 seeds.

## Summary

| Scenario | Finding | Consequence for RHAM |
|---|---|---|
| **S1 Overlap** | Without a countermeasure, concepts merge once they are more similar to each other than the fixed assignment threshold allows: at sibling cosine 0.85, only 104 of 216 concepts, 92.8% accuracy. The **new split rule** raises this to 203 concepts and 99.3%. At extreme overlap (0.91), 75% → 92%; the flat memory remains better there (99%). | **Rule 6 (new): splitting.** Consolidation needs, besides merging, also splitting, driven by the same structure test. |
| **S2 Drift** | Concept accuracy stays at ≈ 99% up to strong drift. But: under strong drift, consolidation stalls; episodes can no longer be forgotten, and memory grows (270 → 790 vectors). A capped learning rate dampens this (→ 490) but does not stop it. | Prototypes need an **aging/forgetting rate of their own** (capped learning rate) – this alone is not enough; merging/retiring outdated prototypes is also needed. Open. |
| **S3 Rare episodes** | Atypical single episodes are automatically preserved (100%, never consolidated). **Typical-looking single episodes are associatively lost (0%)** – but 100% retrievable via the archive. With important-tagging, 100% direct. | Confirms the **three-layer model**: forgetting = demotion, not deletion. For important content, a **tag** is needed (biologically: emotional/salience tagging). |

---

## S1 – Overlapping concepts

The last tree level is shrunk until sibling concepts become more similar to each other than two episodes of the same concept (≈ 0.86).

**Split rule (new):** In every sleep phase, for each prototype with ≥ 12 episodes (archive + buffer), check whether 2-means reduces the spread more than a null model of "one concept + isotropic noise" with the same spread (gap logic, Tibshirani et al. 2001). If the difference is > 0.1, the prototype is split, the archive and assignments are carried along, and the test is repeated for the parts.

| Sibling cosine | Oracle | flat | RHAM without split | RHAM + split | concepts recognized without / with | cost per retrieval without / with | splits |
|---|---|---|---|---|---|---|---|
| 0.66 | 1.000 | 1.000 | 1.000 | 0.996 | 213 / 213 | 33 / 32 | **0** |
| 0.75 | 1.000 | 1.000 | 0.999 | 0.998 | 209 / 213 | 46 / 35 | 5 |
| 0.85 | 1.000 | 0.999 | 0.928 | **0.993** | 104 / 203 | 350 / 53 | 99 |
| 0.91 | 1.000 | 0.991 | 0.748 | **0.918** | 38 / 132 | 302 / 106 | 94 |

**Reading:** (a) **No false splitting** for well-separated concepts (0 splits at 0.66). (b) Merged prototypes are doubly expensive: they reconstruct their episodes poorly, the episodes remain in the buffer and raise retrieval cost (350 instead of 53). (c) At extreme overlap, splitting does not quite suffice – here the concept signal per episode is weak, and a mean-based representation loses out to storing all individual cases. (d) The null model assumes isotropic noise; with real, anisotropic data there is a risk of over-splitting. Must be checked against real data.

## S2 – Concept drift

After every sleep phase, each concept center takes a random step of length δ. Comparison: running mean (learning rate 1/n) vs. capped learning rate (n ≤ 30, equivalent to an exponential moving average). 15,000 episodes.

| δ | Learning rate | top-1 at end | top-1 mean over time | flat at end | prototypes | associative memory at end |
|---|---|---|---|---|---|---|
| 0 | 1/n | 1.000 | 0.992 | 1.000 | 216 | 269 |
| 0.05 | 1/n | 0.999 | 0.993 | 1.000 | 216 | 269 |
| 0.1 | 1/n | 1.000 | 0.992 | 1.000 | 216 | 285 |
| 0.1 | capped | 1.000 | 0.990 | 1.000 | 216 | 277 |
| 0.2 | 1/n | 1.000 | 0.989 | 1.000 | 240 | **790** |
| 0.2 | capped | 0.961 | 0.987 | 1.000 | 228 | **488** |

**Reading:** Accuracy holds, but for the wrong reason. Under strong drift, prototypes lag behind, current episodes no longer reach the forgetting threshold and remain in the episodic buffer – the buffer takes over retrieval ("the hippocampus compensates"), and consolidation stalls. The capped learning rate lets prototypes keep up and saves 38% of storage, but it does not stop the growth, and in the end one seed drops to 0.90 (single measurement). **Open gap:** outdated and duplicate prototypes are never retired. A merging/aging rule for prototypes would be needed.

## S3 – Rare single episodes

Per sleep phase, 20 tagged single episodes: 10 atypical (random direction, not similar to any concept) and 10 typical-looking (statistically indistinguishable from ordinary episodes). At the end: exact retrieval of exactly this episode from a slightly noisy cue (noise 0.1). 10,200 episodes total.

| Episode type | associative | associative + important-tag | with archive | flat |
|---|---|---|---|---|
| rare, atypical | 1.00 | 1.00 | 1.00 | 1.00 |
| rare, typical-looking | **0.00** | 1.00 | 1.00 | 1.00 |
| ordinary | 0.01 | 0.02 | 1.00 | 1.00 |

Cost per retrieval with archive: ≈ 320 "hot" comparisons (buffer + prototypes) plus ≈ 85 (atypical) resp. ≈ 740 (typical/ordinary) "cold" archive accesses – versus 10,200 for the flat memory. Tagging enlarges the buffer by exactly the tagged episodes (107 → 207).

**Reading:** Forgetting by reconstructibility is **selective in the right direction**: what doesn't fit the schema is preserved automatically. What fits the schema becomes gist – even if it was individually important. This is exactly the case of the "50-year-old file that looks typical": without the archive it would be lost; with the archive, retrieving it costs a two-step access (prototype → archive list). Important-tagging is the cheap protection for known important cases.

## Methodological notes

- An abort due to the 10-minute tool limit also killed a run started in the same command. Fix: decouple runs with `setsid`/`nohup`, poll status at intervals < 10 min.
- Own misreading corrected: memory under drift with capped learning rate was initially reported as "≈ 370" (an intermediate reading at 13,000 episodes); the final value is 488.

## Consequences for the architecture (rule set v2)

1. Distributed storage · 2. Associative retrieval (v3: routing in residual space, exact final step) · 3. Dynamic hierarchy (interference **and** structure test) · 4. Offline consolidation · 5. Selective forgetting = **demotion to the archive**, never deletion; tagged episodes excepted · **6. Splitting** of overly broad prototypes (structure test against a one-concept null model) · **7. Prototype aging** (capped learning rate; merging/retirement still open).

## Next steps

1. Complete Rule 7: merge duplicate and retire outdated prototypes; repeat S2.
2. Anisotropic noise (test over-splitting from Rule 6).
3. Real embeddings.
4. Independent code review before any publication.
