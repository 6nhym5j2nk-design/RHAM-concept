---
project: RHAM
repo: https://github.com/6nhym5j2nk-design/rham-project
date: 2026-10-08
tags: [rham, embeddings, ollama, smoke-test]
mode: FAST
---
# RHAM – Smoke Test with Real Embeddings (2026-10-08)

**Code:** `sim/rham_embed.py smoke`. Model: `bge-m3` (Ollama, local on the Mac mini), 1024 dimensions. Corpus: the project's own markdown files `docs/*.md` (5 files, 57 chunks of ~600 characters each) – real, non-synthetic text, but used only to test the pipeline.

## Result

| | RHAM | flat (exact NN) |
|---|---|---|
| document assignment (held-out, 14 queries) | 0.643 | 0.643 |
| stored vectors | 33 (M0=28, hot=5) | 43 |
| cost per query | 33 | 43 |

## Reading

First end-to-end confirmation that the pipeline (chunking → real Ollama embeddings → `OnlineRHAM7` consolidation → held-out evaluation) works with real, non-synthetic vectors, without crashes or nonsensical values. RHAM reaches the same accuracy as the flat baseline at 23% less storage – consistent with the pattern from the synthetic experiments (round 2, E5).

**Caveat:** The sample is far too small for robust conclusions (57 chunks, 5 documents, 14 queries; 9/14 hits). This is only a functional proof of the pipeline, not a statement about the actual research question (meaningful sub-structure within a topic). Next step: a run on a real domain corpus (`sim/corpus_real/`, local, not in the repo) – a dictionary, an instrument catalog, own publications.

## Next steps

1. Place a real domain corpus in `sim/corpus_real/` and run `python3 sim/rham_embed.py real`.
2. On a larger corpus: test the split rule (Rule 6) and the G-means split test from round 4 against real embedding clusters, not just the isotropic null-model assumption.
3. Qualitative spot-check: which chunks end up in the same prototype – does that make substantive sense?
