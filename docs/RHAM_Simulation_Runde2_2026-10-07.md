---
project: RHAM
repo: https://github.com/6nhym5j2nk-design/rham-project
date: 2026-10-07
tags: [rham, simulation, online, p4, sensitivity]
mode: VERIFY
---
# RHAM – Simulation Round 2: Online Stream, P4, Sensitivity (2026-10-07)

**Code:** `sim/rham_online.py` (E5), `sim/rham_sim.py` (E1 with v3, E6), `sim/plot_online.py`. Raw data: `sim/results_E5.json`, `sim/results_E5_noforget.json`, `sim/results_E6.json`, `sim/results_E1*.json`. Figure: `sim/rham_online_ergebnisse.png`. Predecessor: [[Projects/RHAM/RHAM_Simulation_2026-10-07]].

## Summary

1. **P4 confirmed (idealized):** In the online stream, the marginal cost per new episode falls to the source's true entropy rate – 73.3 bit vs. 73.5 bit (raw: 151 bit). After new topic branches are introduced, it briefly rises and converges again (75.6 vs. 74.4 bit after 10,000 further episodes).
2. **Memory grows with structure, not with experience:** After 20,000 episodes, RHAM holds ≈ 270 vectors associatively (flat: 20,000), at **100% concept retrieval** for both old and new concepts (flat with exact NN: likewise 100%). Compute per retrieval ≈ 31 instead of 20,000 dot products.
3. **Unsupervised concept discovery:** At the end, 213–216 prototypes for 216 true concepts (5 seeds).
4. **No catastrophic forgetting:** Old concepts remain at 99.5–100% after new branches are added; new concepts reach ≥ 99.7% after 4–5 sleep phases.
5. **Sensitivity:** False growth on noise data is fully prevented by any gap threshold ≥ 0.05, regardless of θ. Depth accuracy is limited by the choice of cluster count, not by the thresholds.
6. **v2 vs. v3:** a genuine trade-off between sharpness (v2) and accuracy under noise (v3). The sharpness advantage of the hierarchy from round 1 comes mostly from per-level normalization (acting like a locally adapted β).

---

## 1. E1 addendum: v2 vs. v3 at the same β = 16 (beam 2)

| N | flat sharp | v2 sharp | v3 sharp | flat top-1 | v2 top-1 | v3 top-1 |
|---|---|---|---|---|---|---|
| 512 | 0.845 | 0.967 | 0.861 | 1.00 | 0.997 | 1.00 |
| 1000 | 0.679 | 0.979 | 0.729 | 1.00 | 0.997 | 0.999 |
| 4096 | 0.259 | **0.928** | 0.337 | 1.00 | 0.999 | 0.999 |

Noise sweep (N = 1000, best β in each case), top-1:

| η | NN ceiling | flat | v2 | v3 |
|---|---|---|---|---|
| 1.0 | 0.995 | 0.995 | 0.973 | 0.985 |
| 1.6 | 0.807 | 0.809 | 0.674 | **0.710** |
| 2.0 | 0.550 | 0.555 | 0.417 | **0.458** |

**Reading:** The exact final step (v3) removes the noise amplification but loses sharpness. v2's sharpness comes from the residual normalization – equivalent to a locally raised inverse temperature. Honest conclusion: the claim "the hierarchy retrieves more sharply at bounded β" holds only if per-level normalization is understood as part of the architecture (Rule 2: each level its own representation space). A hybrid (decide via the exact score, sharpen in residual space) is a natural next step, not yet tested.

## 2. E5 – Online stream

**Setup:** 216 concepts (tree K = 6, h = 3, d = 64), Zipf frequencies (a = 1.1). Episode = concept + noise (norm 0.4). Episodes 1–10,000 drawn only from 3 of 6 main branches (108 concepts), afterward all of them. Sleep phase every 1000 episodes:
1. assignment to a prototype at cosine ≥ 0.80 → running mean update;
2. unassigned episodes → new prototype if ≥ 3 similar episodes (DP-means-like; Kulis & Jordan 2012, ICML);
3. forgetting: an episode leaves M0 when its cosine to the prototype ≥ 0.85 and the prototype carries ≥ 3 episodes (the episode stays in the archive);
4. upper levels rebuilt from M1 with the growth rule.
Retrieval: v3 (routing β = 16, final step β = 64, beam 2) over the hierarchy plus an exact comparison with the remaining M0 buffer.

**Bit accounting (P4):** Two-part code per window: model part = new prototypes × 64 × 32 bit / 1000; data part = entropy of the prototype assignment + Gaussian rate-distortion of the residual at distortion D = 0.0005 per dimension. True rate = concept entropy + the generator's noise rate (measured from the generator).

| Episodes | M0 | M1 | associative total | bits RHAM | bits raw | entropy rate | concept top-1 old / new | cost per retrieval |
|---|---|---|---|---|---|---|---|---|
| 1,000 | 58 | 56 | 135 | 190 | 151 | 73.4 | 0.977 / – | 80 |
| 5,000 | 3 | 106 | 132 | 77.3 | 151 | 73.4 | 1.000 / – | 30 |
| 10,000 | 0 | 108 | 138 | **73.3** | 151 | **73.5** | 1.000 / – | 24 |
| 11,000 | 65 | 140 | 268 | 143 | 154 | 74.4 | 0.998 / 0.929 | 93 |
| 15,000 | 18 | 203 | 287 | 92.2 | 154 | 74.4 | 0.999 / 0.997 | 46 |
| 20,000 | 2 | 215 | 269 | **75.6** | 154 | **74.4** | 1.000 / 1.000 | 31 |

(Averages over 5 seeds.) **Control without forgetting** (3 seeds): same accuracy (1.00 / 1.00), but ≈ 20,260 associative vectors and ≈ 20,030 dot products per retrieval. Forgetting costs **no** concept accuracy in this scenario and saves 98.7% of storage and compute.

**Reading:**
- Marginal cost follows exactly the predicted curve: high while the model (the prototypes) is still being learned, then convergence onto the entropy rate. New types of experience produce a "learning bump," then convergence again. This is the quantitative form of "more experience → better representation instead of more bytes."
- The initial value above "raw" (190 bit) is the model part: prototypes must be paid for first.

## 3. E6 – Sensitivity of the growth rule

7 datasets (4 trees with h = 2–5, 3 noise datasets) × 5 seeds; θ ∈ {0 … 0.7}, gap ∈ {0 … 0.5}. Sanity check: θ = 0.2 / gap = 0.15 reproduces E3 seed for seed.

**False growth on noise data** (fraction of runs with > 1 level): at gap = 0, depending on θ, 67–100%; **at any gap threshold ≥ 0.05: 0%**. Gap values along the paths: noise −0.03 to 0.01; real structure 0.33–1.27.

**Depth on tree data** (fraction exactly = data depth h): best field θ = 0.5, gap 0.3–0.5 with 75% exact (mean error 0.25 levels); plateau for θ 0.2–0.5. At θ = 0.7, needed levels are suppressed (50%).

**Reading:** The choice of thresholds is uncritical as long as some structure test is active at all. The remaining depth errors arise from (a) the wrong cluster count from the geometric candidate grid (+1 level) and (b) the intended property of never building a level without interference (h = 5 → 4).

## 4. Methodological findings of this round

1. **Bookkeeping bug in the first draft of `rham_online.py`** (episodes could have been folded into prototype means more than once). Found while reading through the code before the first run; replaced with an explicit "already counted" flag per episode.
2. **Wait loop hung:** `pgrep -f "rham_..."` matched its own shell command, which contains the search pattern, and therefore never terminated (aborted after 10 min). Fix: store process IDs at start and check them with `wait` or `kill -0 <pid>`.

## 5. Limitations

- The data are exactly "prototype + isotropic noise." That the data part of the code hits the entropy rate is in part a structural artifact of this generator; the non-trivial achievement is that the system finds the concepts **unsupervised and online** (213–216 of 216).
- Bit accounting uses a Gaussian rate-distortion approximation and 32-bit prototypes.
- Concept retrieval was measured, not exact episode retrieval. Forgotten episodes are only reachable via the prototype (gist) or an archive pointer – intended, but useful for exact content only together with the archive (three-layer model).
- Thresholds τ_assign = 0.80 / τ_forget = 0.85 are tuned to this noise level; for real data they would need to be learned or adaptive.

## 6. Next steps

1. Test a hybrid retrieval (decision exact, sharpening in residual space).
2. Harder synthetic data: anisotropic noise, overlapping concepts, concept drift (wandering prototypes), very rare concepts (single episodes that are never consolidated).
3. Real embeddings (text/image).
4. Independent code review before any publication.
