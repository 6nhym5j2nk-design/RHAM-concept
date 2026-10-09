# RHAM — Recursive Hierarchical Associative Memory

**Status: architecture hypothesis + simulation. Not a finished system, not a
product, not a published paper.** This repo documents an idea, released for
discussion and further development, because the author has limited time to
work it out alone.

## The core idea

Classical memory: `address → value`.

RHAM replaces the address with a learned, content-based access
(content-addressable, Q/K/V-style as in attention):

```
q → softmax(q·Kᵀ/√d) → weighted sum over V
```

But the real core is not "attention as memory" (that already exists in many
forms, see below), but a **dynamically growing hierarchy of memory levels**,
where each level compresses/abstracts the one below it:

```
M₀ (episodes) → M₁ = C(M₀) (patterns over episodes) → M₂ = C(M₁) (patterns over patterns) → ...
```

A new level is not fixed in the architecture from the start, but
**triggered** when an existing level exceeds an interference/capacity
threshold. Added to this are offline consolidation (patterns are abstracted
from raw data "during sleep") and selective forgetting (redundant details
are weakened once they are reliably represented at a higher level).

Five rules — see [`docs/RHAM_Machbarkeitsanalyse.md`](docs/RHAM_Machbarkeitsanalyse.md)
for the full derivation:

1. Distributed storage (high-dimensional activation patterns instead of addresses)
2. Associative retrieval (Q/K/V-style attention)
3. Dynamic hierarchy (new level on interference + structure test)
4. Offline consolidation (`M_{n+1} ← C(M_n)`)
5. Selective forgetting (demotion to the archive, never deletion of the raw data)

**Important correction to the first intuition:** physical information
capacity does *not* grow exponentially this way — Shannon cannot be
circumvented. What grows is the combinatorial representation capacity
(relations between relations), and the hypothesis is that as experience
accumulates, the system increasingly stores the **structure** of the
experience space instead of raw data — similar to the distinction between
episodic and semantic memory.

## What already exists in the literature (as checked: October 2026)

A targeted prior-art search found that large parts of the individual idea
already exist, some of them very close to the overall combination:

- **Nested Learning / HOPE** (Behrouz et al.) — nested memory levels with
  different update frequencies
- **Titans** (NeurIPS 2025) — a neural long-term memory that keeps changing
  at inference time
- **Memory Layers at Scale** (Meta) — trainable key-value lookups as a
  memory layer, up to 128B parameters
- **HMT — Hierarchical Memory Transformer** (NAACL 2025) — explicitly
  hierarchical memory with recall and segment-wise recurrence
- **"Language Models Need Sleep"** (Behrouz, Hashemi, Mirrokni, 2026) — an
  explicit sleep/consolidation mechanism
- **DeltaStack** (ICML 2026) — a differentiable stack addressing the
  limitation of fixed-size associative memory for recursive structures

**Not found** (as of this check, no exhaustive adversarial novelty search):
an architecture in which the **number/depth of memory levels itself grows
dynamically as a function of the stored information** (rather than being
architecturally fixed in advance, as in HOPE), combined with
capacity-/interference-triggered level creation, offline consolidation, and
subsequent selective forgetting in one closed cycle. That is the concrete
point that would need further checking — among others against Neural Turing
Machines/DNC, adaptive computation, growing neural networks, hierarchical
predictive coding, hippocampal–cortical models, and vector-symbolic/
hyperdimensional computing.

## What's in this repo — and what isn't

This repo contains the **conceptual derivation and the synthetic
simulations** (four stress-test rounds with artificially generated concept
clusters, plus a first functional check with real text embeddings on a
small, non-sensitive example corpus).

It does **not** contain the empirical tests on real domain-text corpora
(including a domain analysis on a medical reference dictionary) — those
continue in a private repo, among other reasons because questions of text
provenance and due diligence (no personal/patient data, source criticism)
need to be checked there on an ongoing basis before any citable claims could
arise from them.

## Results of the synthetic simulations (summary)

| Round | Question | Result |
|---|---|---|
| 1 | Does associative access help at all? | P1 partially refuted: the advantage is sharp retrieval at bounded β + logarithmic retrieval cost, not more capacity |
| 2 | Online stream, realistic concept distribution | Bits/episode track the source's entropy rate; 270 instead of 20,000 vectors at 100% concept retrieval |
| 3 | Stress test: overlapping concepts | A split rule resolves merging; rare episodes without tagging are lost in the index but remain retrievable in the archive |
| 4 | Drift + aging | An aging rule halves memory under drift at 100% accuracy |

Details in `docs/RHAM_Simulation_*.md`. Code in `sim/`:
`rham_sim.py` (round 1), `rham_online.py` (round 2), `rham_hard.py`
(round 3), `rham_rule7.py` (round 4, the most current core implementation
of the memory class).

## Get involved

This is an open idea, not a finished paper. Anyone who wants to develop it
further, refute it, reproduce the simulations, or check it against more
prior art — issues and PRs welcome. Especially helpful:

- A real adversarial novelty search against the literature cited above
- Independent reproduction of the simulation results
- Theoretical grounding of the capacity argument (Shannon limits,
  combinatorial vs. physical capacity)

## License

Not finalized yet — until then: code and text are free for discussion,
please credit this repo if you use it. Open an issue for a formal license.
