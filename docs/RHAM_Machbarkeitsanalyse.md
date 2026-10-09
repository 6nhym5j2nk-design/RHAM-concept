---
project: RHAM
repo: https://github.com/6nhym5j2nk-design/rham-project
date: 2026-10-07
tags: [rham, associative-memory, hopfield, capacity, thermodynamics, feasibility]
mode: VERIFY
---
# RHAM – Mathematical-Physical Feasibility Analysis

**Question:** Is a *Recursive Hierarchical Associative Memory* (RHAM) – an associative Q/K/V memory whose level depth grows when triggered by interference, with offline consolidation and selective forgetting – mathematically and physically possible?

**Short answer:** Yes, with three precise caveats.

1. **Mathematically possible.** Every building block has a proven foundation today: exponential capacity of dense associative memories (Ramsauer 2020, Hu/Wu/Liu 2024), Lyapunov-stable multi-layer Hopfield networks (Krotov 2021), prototype emergence *from interference* (Cowsik & Sriram 2026), demand-driven growth of associative memories (Self-Sizing Hopfield 2025). The combination is consistent; there is no theorem that forbids it.
2. **The capacity does not become exponential – the addressability does.** Information-theoretically: stored bits ≤ parameters × bits/parameter (Shannon). The hierarchy, with $L$ levels of $K$ prototypes each, produces up to $K^L$ addressable combinations while storing only $L\cdot K$ prototypes. That is the formal content of "exponential": *combinatorial addressability*, not information content. The real gain per experience is exactly the **compressibility of the experience stream** (entropy rate $h$ vs. raw size $b$) – for incompressible experience, RHAM buys nothing.
3. **Physically possible, and forgetting is the only thermodynamically mandatory cost.** Landauer: erasing one bit costs at least $k_BT\ln 2 \approx 2.9\cdot10^{-21}$ J at 300 K (experimentally confirmed, Bérut et al. 2012). Still et al. 2012 show: a system that retains non-predictive information dissipates exactly that information as heat. **Selective forgetting is therefore not merely permitted but thermodynamically optimal** – a strong physical argument for Rule 5 of RHAM.

The actual novelty remains narrow: *dynamically growing abstraction depth* plus *hierarchical top-down addressing*. Both exist so far only in parts (Section 9). The depth does not grow without bound; on current evidence it grows at most **logarithmically** with the number of episodes (Section 5).

---

## 1. Formalization

Let $\mathcal{E}=(e_1,e_2,\dots)$ be a stream of experiences, $e_t\in\mathbb{R}^{b}$ (raw size $b$ bits).

**Level $n$:** $M_n=\{(k_i^{(n)},v_i^{(n)})\}_{i=1}^{N_n}$, keys $k\in\mathbb{R}^{d_n}$.

**Retrieval (associative):**
$$r_n(q)=\sum_i \operatorname{softmax}_i\!\big(\beta\,q^\top k_i^{(n)}\big)\,v_i^{(n)}.$$

**Interference of a level:** With separation $\Delta_i = k_i^\top k_i-\max_{j\neq i}k_i^\top k_j$ (Ramsauer et al. 2020) we define
$$I(M_n)=\frac{1}{N_n}\big|\{i:\ 2(N_n-1)\,e^{-\beta\Delta_i}\,>\,\varepsilon\}\big|,$$
the fraction of patterns whose guaranteed retrieval error bound exceeds $\varepsilon$. $I$ is computable online (only dot products).

**Growth rule:** $I(M_n)>\theta_n \Rightarrow$ create $M_{n+1}$.

**Consolidation operator:** $M_{n+1}\leftarrow C(M_n)$ with $C$ = clustering/prototype formation, such that $k_i^{(n)}\approx p_{c(i)}^{(n+1)}+\rho_i^{(n)}$ (prototype plus residual).

**Selective forgetting:** remove $(k_i,v_i)$ from $M_n$ when reconstruction from $M_{n+1}$ is sufficiently good: $\|v_i - \hat v_i(M_{n+1})\|<\delta$.

**Cycle:** Experience → $M_0$ → interference → consolidation → new level → compression → forgetting.

---

## 2. Capacity of a single level – what is proven

| Model | Capacity | Condition | Source (status) |
|---|---|---|---|
| Classical Hopfield | $\approx 0.138\,N$ | random patterns | Amit/Gutfreund/Sompolinsky 1985 (peer-reviewed, standard) |
| Dense AM, exponential coupling | $\propto e^{\alpha N}$ | random patterns | Demircigil et al. 2017, J. Stat. Phys. (peer-reviewed) |
| Modern Hopfield = attention | $N\ge\sqrt p\,c^{(d-1)/4}$, exponential in $d$ | patterns on sphere, large separation $\Delta_i$ | Ramsauer et al. 2020/ICLR 2021 (peer-reviewed) |
| Kernelized Hopfield | $M^\star\asymp c^{D_\Phi}$ optimal, when memories form an optimal spherical code | $\Delta\ge\frac1\beta\ln\frac{2(M-1)}{R}$ | Hu, Wu, Liu 2024 (arXiv; preprint) |
| Biologically plausible dense AM | exponential in number of hidden units via distributed (compositional) representation | threshold nonlinearity | Shafiei Kafraj, Krotov, Latham 2026 (arXiv; preprint) |

**Decisive insight:** Exponential capacity holds only for *well-separated* patterns. Real experiences are correlated; $\Delta_i$ becomes small, and the error bound $2(N-1)e^{-\beta\Delta_i}$ explodes. Interference is therefore not a marginal phenomenon but the **normal case** for structured data. That is precisely what makes $I(M_n)$ a sensible growth trigger.

In addition: softmax attention approximates Kanerva's *Sparse Distributed Memory* (Bricken & Pehlevan, NeurIPS 2021), i.e. the Q/K/V formulation is not merely a transformer convention but an established, biologically plausible associative memory model.

---

## 3. Why a hierarchy "rescues" capacity (proof sketch)

Let the keys in $M_0$ be correlated within $K$ clusters: $k_i=p_{c(i)}+\rho_i$, with prototypes $p_c$ and residuals $\rho_i$, $\|\rho_i\|\ll\|p_c\|$.

Flat separation: $\Delta_i^{\text{flat}} \approx \|\rho_i\|^2-\max_{j\ne i,\,c(j)=c(i)}\rho_i^\top\rho_j + \text{(cluster term, small)}$ – dominated by intra-cluster overlap, worsening as $N_0$ grows.

After consolidation, $M_1$ stores the $K$ prototypes (well separated, error bound $\propto K$ instead of $N_0$) and $M_0$ stores only residuals *relative to the prototype*. Within a cluster $c$, only $N_c\approx N_0/K$ patterns compete, and after centering the residuals are nearly isotropic → separation approaches the random case → the Ramsauer regime holds again.

**Capacity calculation:** With $L$ levels and $K$ prototypes per node,
$$\#\text{addressable combinations}=K^L,\qquad \#\text{stored prototypes}=L\cdot K .$$
This is identical to *residual/product quantization* (Chen et al. 2010; Jégou et al. 2011, IEEE TPAMI) and *HNSW* (Malkov & Yashunin 2018, IEEE TPAMI) for logarithmic search. The RHAM hierarchy is mathematically their **learned, distributed** variant.

**Convergence:** Multi-layer Hopfield networks with symmetric weights possess a Lyapunov energy $E$ with $dE/dt\le0$ (Krotov, *Hierarchical Associative Memory*, 2021). Each RHAM level can be realized as such a layer; the stability of the retrieval dynamics is thereby guaranteed, as long as forward and backward weights are kept symmetric (a restriction – see Section 8).

**Prototypes emerge spontaneously from interference:** Cowsik & Sriram (Stanford, Sept. 2026) show that dense Hopfield networks, in which only leaf patterns are stored, form the *ancestor prototypes* as stable minima of the energy landscape – triggered by the interference of the leaf patterns. With $N^{\Theta(\log N)}$ examples, prototypes can be reconstructed down to depth $\log N$ (Theorem 6: the instability probability of the ancestors vanishes under suitable scaling). This is the strongest evidence to date that "interference → abstraction" is not merely a metaphor but a physical mechanism in energy landscapes. **But:** there the depth $h$ is fixed in advance; nothing grows.

---

## 4. The information-theoretic limit – what "exponential" does and does not mean

**Hard limit:** A system with $P$ parameters of $q$ bits each stores at most $P\cdot q$ bits. No hierarchy circumvents this.

**Source coding:** If the experience stream has entropy rate $h$ bits/experience, any lossless storage of $T$ experiences needs at least $T\cdot h$ bits (Shannon). Raw storage needs $T\cdot b$. RHAM's maximum gain is the factor
$$G=\frac{b}{h}.$$
For strongly structured experience ($h\ll b$), $G$ is large; for noise ($h\approx b$), $G\approx1$ – the hierarchy then only costs.

**What actually changes substantially is the marginal storage per experience:**
$$\frac{dS}{dT}\ \xrightarrow{T\to\infty}\ h \ll b .$$
This is the precise formulation of the thesis "more experience → less additional storage per experience": it is true exactly when experiences carry little new entropy *conditional on* the already learned world model. RHAM is thus an **online MDL system**: levels $M_1,\dots,M_L$ are the model part of a two-part code, $M_0$ the data part.

**Combinatorial addressability:** $K^L$ combinations are *representable*, not *stored*. That is exactly the distinction the brain also exploits (distributed representation). It is real and valuable (generalization, compositionality), but it is not storage capacity in the Shannon sense.

---

## 5. Dynamic growth – mathematically unproblematic, but with a natural ceiling

Growth of neural structures under capacity demand is well established: Cascade-Correlation (Fahlman & Lebiere 1990), Growing Neural Gas (Fritzke 1995), Progressive Networks (Rusu et al. 2016), Dynamically Expandable Networks (Yoon et al., ICLR 2018). Specifically for associative memories: *Self-Sizing Hopfield* (arXiv 2507.10443, 2025) grows "only on genuine novelty" up to the environment's intrinsic storage requirement (estimated via Urysohn width) – without a preset target and without validation search. **But:** there, the *width* of a level grows, not an *abstraction level*.

**When is a new level worthwhile?** Only when it reduces the description length:
$$H(M_n\mid M_{n+1})+\operatorname{cost}(M_{n+1})\ <\ H(M_n).$$
This is the MDL stopping criterion. Since each level reduces the number of units by a factor of $K$, the achievable depth is
$$L_{\max}\approx\log_K N_0 .$$
This matches the result of Cowsik & Sriram (generalization down to depth $\log N$). **Depth grows logarithmically, not without bound.** An "infinite" recursion $M_0\to M_1\to M_2\to\dots$ is not mathematically sensible, because eventually $K$ prototypes remain that carry no further structure. For realistic $N_0\sim10^6$–$10^9$ and $K\sim10$–$100$, this means $L\approx3$–$9$ levels – a practicable, not an explosive, number.

---

## 6. Consolidation and forgetting

**Biological model** (peer-reviewed): Complementary Learning Systems – fast hippocampal episodic memory, slow cortical structure extraction, replay during sleep (McClelland, McNaughton & O'Reilly 1995, *Psychol. Rev.*; update Kumaran, Hassabis & McClelland 2016, *Trends Cogn. Sci.*). RHAM is its recursive generalization: not one pair (hippocampus, cortex), but $L$ levels with the same mechanism.

**Machine learning:** Nested Learning/HOPE (Behrouz et al., NeurIPS 2025) with multiple update frequencies; "Language Models Need Sleep" (Behrouz, Hashemi, Mirrokni, arXiv 2606.03979, 2026) with memory consolidation and dreaming. Both: fixed number of levels.

**Mathematically**, $C$ is a clustering/distillation step (EM, k-means, prototype learning), forgetting is pruning with a reconstruction condition. Both are standard; the open question is not *whether* but *which* $C$ actually reaches the $\log N$ depth range without destroying the leaves (memorization–generalization trade-off, Cowsik & Sriram: sharper activation $n$ stabilizes leaves and destabilizes ancestors).

---

## 7. Physics

**Landauer bound.** Erasing one bit costs $\ge k_BT\ln2$ ($2.87\cdot10^{-21}$ J at 300 K). Experimentally confirmed (Bérut et al., *Nature* 2012). Storing and reversible computing have *no* fundamental lower bound – only **erasure** does. In RHAM, selective forgetting is thus the only thermodynamically unavoidable cost item. Per forgotten episode of $b$ bits: $\ge b\,k_BT\ln2$. For $b=10^4$, $10^9$ episodes: $\approx 3\cdot10^{-8}$ J – negligible. Practical hardware runs $10^{3}$–$10^{6}$ times above that; the bound is therefore no obstacle.

**Thermodynamics of prediction** (Still, Sivak, Bell & Crooks, *Phys. Rev. Lett.* 2012): For a system interacting with a stochastic environment, dissipation is lower-bounded by the *non-predictive* information it retains about the past. Translated: **a memory that stores details which predict nothing about the future pays for it in heat.** This is the physical justification for why consolidation + forgetting (retaining the regularity, erasing the residual) is not only storage-economical but thermodynamically optimal. To my knowledge, this connection has not previously been drawn in the RHAM literature.

**Hardware realizability.** Associative memories with $O(1)$ in-memory retrieval exist: memristive Hopfield networks with superlinear capacity $K\approx0.3\,N^{1.2}$ on 25×25 arrays (arXiv 2605.07223, 2026; preprint, conference status unclear) and in-memory hyperdimensional computing (Karunaratne et al., *Nature Electronics* 2020). Retrieval at level $n$ costs $O(N_n d_n)$ operations; the hierarchy reduces this to $O(L\cdot K\cdot d)=O(d\log N_0)$ – the main physical argument *for* the hierarchy is retrieval energy, not storage amount.

**Biological plausibility check.** Synapses carry ≈ 4.7 bits (26 distinguishable strengths; Bartol et al., *eLife* 2015). With $\sim10^{14}$ synapses, that gives an upper bound of about $5\cdot10^{14}$ bits ≈ 60 TB. A lifetime of sensory data exceeds that by orders of magnitude. The brain therefore **must** compress and forget – consistent with Section 4's factor $G=b/h$.

---

## 8. Where it can fail (red-team points)

1. **Misrouting.** If $M_1$ picks the wrong cluster, $M_0$ cannot correct it. Hard top-down addressing is a greedy search tree; soft routes are required (top-$k$/beam across levels), which raises retrieval cost by a factor of $k$. HNSW solves exactly this problem with multiple candidates per level.
2. **Symmetry constraint.** Lyapunov guarantees (Krotov 2021) require symmetric weights. Asymmetric Q/K/V projections lose the convergence guarantee; single-step attention (Ramsauer) sidesteps this but offers no multi-step stability.
3. **Drift.** Prototypes age under a non-stationary environment. Self-Sizing Hopfield addresses this with *re-binding* instead of deletion; RHAM needs a rule for when a prototype itself gets consolidated or abandoned.
4. **Diminishing returns.** Level $n+1$ is worthwhile only if $M_n$ itself is compressible. The MDL criterion from Section 5 must supplement the trigger $I(M_n)>\theta_n$, otherwise empty levels arise.
5. **Leaf/prototype trade-off.** Per Cowsik & Sriram, there are parameter regimes in which either leaves or ancestors are stable, not both. RHAM must decouple the levels (own $\beta_n$, own $d_n$) – this argues for Rule 2 ("each level its own representation space") and against shared weights.
6. **No free lunch.** At high entropy rate ($h\approx b$), RHAM is an expensive detour around a flat memory.

---

## 9. Novelty update (addendum to the prior-art matrix)

| Work | Covers | Does **not** cover |
|---|---|---|
| Krotov 2021, Hierarchical AM (arXiv) | multi-layer AM with energy, bottom-up abstraction | no growth, no consolidation/forgetting |
| Cowsik & Sriram 2026 (arXiv) | prototypes emerge from interference; depth $\log N$ | fixed depth, no top-down addressing, no forgetting |
| Self-Sizing Hopfield 2025 (arXiv 2507.10443) | demand-triggered growth with no preset target | grows in width, not in abstraction levels; explicitly "no forgetting" |
| Hu/Wu/Liu 2024 (arXiv) | optimal capacity via spherical codes | no hierarchy |
| Shafiei Kafraj/Krotov/Latham 2026 (arXiv) | exponential capacity via distributed composition | no hierarchy, no growth |
| Behrouz et al. 2025/2026 (NeurIPS; arXiv) | multiple timescales, sleep consolidation | fixed number of levels |
| Jégou 2011 / Malkov 2018 (IEEE TPAMI) | residual quantization, log-time search | not learned, not distributed, no consolidation |

**Remaining novelty core:** (a) interference- *and* MDL-triggered growth of the *abstraction depth*, (b) each level stores structure of the representations below it, with its own Q/K/V space, addressed top-down, (c) forgetting coupled to reconstructibility from above, (d) the thermodynamic justification (Still 2012 → selective forgetting). None of the cited works contains (a)+(b)+(c) together. The novelty verdict remains "plausibly novel, not established" – an adversarial search against DNC, predictive coding, hyperdimensional computing is still outstanding.

---

## 10. Testable predictions and next step

**Predictions**
- P1: On hierarchically generated synthetic data (a tree with branching $K$, depth $h$), at the same parameter budget RHAM achieves error-free retrieval capacity higher by roughly a factor $K$ per level than a flat modern Hopfield memory.
- P2: The automatically reached number of levels saturates at $\approx\log_K N_0$ and matches the data depth $h$.
- P3: On i.i.d. noise data, no level is created above $M_0$ (the MDL criterion fires).
- P4: The marginal storage per episode $dS/dT$ converges to the data's empirical entropy rate $h$.

**Update 2026-10-07 (simulation):** P1 partially refuted in this form: the hierarchy's advantage lies in sharp retrieval at bounded β and logarithmic retrieval cost, not in higher capacity at unconstrained β. P2 refined (depth = depth needed, not maximum depth), P3 confirmed. Details: [[Projects/RHAM/RHAM_Simulation_2026-10-07]] resp. `docs/RHAM_Simulation_2026-10-07.md`.

**Next step (concrete):** minimal simulation in NumPy/PyTorch: modern Hopfield memory ($\beta$, $d$) + online interference measure $I$ + k-means consolidation + reconstruction-based pruning; compare flat vs. RHAM on synthetic tree data (P1–P3). Effort: 1–2 days. Afterward, draft a theorem: capacity of $L$-level residual attention under the Ramsauer separation condition.

---

## 11. Sources and assessment

Peer-reviewed (high reliability):
- Ramsauer et al., *Hopfield Networks is All You Need*, ICLR 2021 – https://arxiv.org/abs/2008.02217 (Theorem 3, error bound, attention equivalence)
- Bricken & Pehlevan, *Attention Approximates Sparse Distributed Memory*, NeurIPS 2021 – https://proceedings.neurips.cc/paper/2021/hash/8171ac2c5544a5cb54ac0f38bf477af4-Abstract.html
- Demircigil et al., *On a model of associative memory with huge storage capacity*, J. Stat. Phys. 2017 – https://arxiv.org/abs/1702.01929
- Amit, Gutfreund, Sompolinsky, Phys. Rev. Lett. 55 (1985) 1530 – capacity 0.138 N
- Bérut et al., *Experimental verification of Landauer's principle*, Nature 483 (2012) 187
- Still, Sivak, Bell, Crooks, *Thermodynamics of prediction*, Phys. Rev. Lett. 109 (2012) 120604 – https://arxiv.org/abs/1203.3271
- Bartol et al., *Nanoconnectomic upper bound on the variability of synaptic plasticity*, eLife 2015 – https://elifesciences.org/articles/10778
- McClelland, McNaughton, O'Reilly, Psychol. Rev. 102 (1995) 419; Kumaran, Hassabis, McClelland, Trends Cogn. Sci. 20 (2016) 512
- Yoon et al., *Lifelong Learning with Dynamically Expandable Networks*, ICLR 2018 – https://arxiv.org/abs/1708.01547
- Jégou, Douze, Schmid, *Product Quantization for Nearest Neighbor Search*, IEEE TPAMI 2011; Malkov & Yashunin, HNSW, IEEE TPAMI 2018
- Behrouz et al., *Nested Learning*, NeurIPS 2025 – https://arxiv.org/abs/2512.24695
- Karunaratne et al., *In-memory hyperdimensional computing*, Nature Electronics 2020 – https://arxiv.org/abs/1906.01548

Preprints (reviewed for content, not peer-reviewed – treat with caution):
- Krotov, *Hierarchical Associative Memory*, arXiv 2107.06446 (2021)
- Hu, Wu, Liu, *Provably Optimal Memory Capacity for Modern Hopfield Models*, arXiv 2410.23126 (2024)
- Cowsik & Sriram, *Hierarchical Prototype Emergence in Modern Hopfield Models*, arXiv 2609.12079 (Sept. 2026)
- *Associative Memory for Non-Stationary Environments: A Self-Sizing Generalization of Hopfield Networks*, arXiv 2507.10443 (2025)
- Shafiei Kafraj, Krotov, Latham, *A Biologically Plausible Dense Associative Memory with Exponential Capacity*, arXiv 2601.00984 (2026)
- Behrouz, Hashemi, Mirrokni, *Language Models Need Sleep*, arXiv 2606.03979 (2026)
- *Hardware-aware Hopfield Network with a Nonlinear Memristor Array*, arXiv 2605.07223 (2026)

Not verifiable in this session (full text unavailable): arXiv 2609.02195 "Memory as an Energy Landscape" – not used.
