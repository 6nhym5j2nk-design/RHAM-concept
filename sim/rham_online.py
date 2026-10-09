"""
RHAM - online experiment E5 (as of 2026-10-07)

Episodes arrive one at a time as a stream. Each episode is a noisy instance
of a "concept" (a leaf of a tree K^h); concepts occur with Zipf-distributed
frequency. Halfway through the stream, new main branches become active
(non-stationarity).

Memory
  M0  episodic buffer (raw episodes)
  M1  concept prototypes (running means), created online (DP-means-like,
      Kulis & Jordan 2012): new prototype candidates emerge from M0 once
      enough similar episodes are present
  M2+ rebuilt from M1 in every sleep phase using the growth rule
      (interference + gap statistic)

Sleep phase every S episodes
  1. assign episodes in M0 to a prototype (cosine >= tau_assign) -> running mean update
  2. cluster unassigned episodes; a cluster with >= m_min episodes -> new prototype
  3. forgetting: episodes with cosine to the prototype >= tau_forget and
     prototype support >= m_min leave M0 (go to the archive, only a pointer remains)
  4. rebuild the upper levels

Metrics per sleep phase
  - associative memory (vectors in M0 + all levels) vs. flat memory (all episodes)
  - depth of the hierarchy
  - concept retrieval for old and new concepts (top-1 concept correct)
  - P4: bits per episode in the two-part code vs. true entropy rate of the source
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

import rham_sim as rs

normalize = rs.normalize


def make_concepts(K, h, d, rng):
    X, labels = rs.make_tree(K, h, d, rng)
    return X, labels[:, 0]          # leaf prototypes, main branch per leaf


def gauss_bits(var_per_dim, D, d):
    """Rate-distortion of a Gaussian source: d/2 * log2(var/D), >= 0 (approximation)."""
    return d / 2 * max(0.0, np.log2(var_per_dim / D))


def run_stream(seed, K=6, h=3, d=64, T=20000, S=1000, sigma=0.4, zipf_a=1.1,
               tau_assign=0.80, tau_forget=0.85, m_min=3, beta=16, D=0.0005,
               new_branch_frac=0.5, probe_n=300, grow=True, forget=True):
    rng = np.random.default_rng(seed)
    C, branch = make_concepts(K, h, d, rng)
    nC = len(C)
    # Zipf frequencies in random order
    w = 1.0 / np.arange(1, nC + 1) ** zipf_a
    w = w[rng.permutation(nC)]
    old_branches = np.arange(int(np.ceil(K * (1 - new_branch_frac))))
    is_old = np.isin(branch, old_branches)

    def sample(n, active):
        p = w * active; p = p / p.sum()
        c = rng.choice(nC, size=n, p=p)
        noise = normalize(rng.standard_normal((n, d))) * sigma
        return normalize(C[c] + noise), c

    # True entropy rate (empirical, from the generator): concept entropy + noise
    def true_rate(active):
        p = w * active; p = p / p.sum()
        Hc = -(p[p > 0] * np.log2(p[p > 0])).sum()
        E, c = sample(4000, active)
        var_noise = ((E - C[c]) ** 2).mean()
        return Hc + gauss_bits(var_noise, D, d), Hc

    # State
    M0 = np.zeros((0, d)); M0_lab = np.zeros(0, int); M0_counted = np.zeros(0, bool)
    P = np.zeros((0, d)); P_n = np.zeros(0); P_labcounts = []     # M1
    archive_n = 0
    R = None
    log = []
    flat_store = 0

    t = 0
    while t < T:
        active = np.ones(nC, bool) if t >= T * (1 - new_branch_frac) else is_old
        E, c = sample(S, active)
        t += S
        flat_store += S
        # ---- wake phase: episodes land in M0 -------------------------------
        M0 = np.vstack([M0, E]); M0_lab = np.concatenate([M0_lab, c])
        M0_counted = np.concatenate([M0_counted, np.zeros(len(E), bool)])

        # ---- bits for this window (two-part code, before consolidation) ----
        raw_var = ((E - E.mean(0)) ** 2).mean()
        raw_bits = gauss_bits(raw_var, D, d)
        n_proto_before = len(P)

        # ---- sleep: consolidation -------------------------------------------
        # M0_counted[i] = episode i has already been folded into exactly one prototype mean
        if len(P):
            sim = M0 @ P.T
            best = sim.argmax(1)
            assigned = sim.max(1) >= tau_assign
        else:
            best = np.zeros(len(M0), int); assigned = np.zeros(len(M0), bool)
        newly = assigned & ~M0_counted
        for j in np.unique(best[newly]):
            m = newly & (best == j)
            tot = P_n[j] + m.sum()
            P[j] = normalize(P[j] * P_n[j] / tot + M0[m].sum(0) / tot)   # running mean
            P_n[j] = tot
            for lab_ in M0_lab[m]:
                P_labcounts[j][lab_] = P_labcounts[j].get(lab_, 0) + 1
        M0_counted = M0_counted | assigned

        # new prototypes from unassigned episodes (DP-means-like, greedy)
        un = np.where(~M0_counted)[0]
        if len(un) >= m_min:
            U = M0[un]
            used = np.zeros(len(un), bool)
            for i in rng.permutation(len(un)):
                if used[i]:
                    continue
                grp = np.where((U @ U[i] >= tau_assign) & ~used)[0]
                if len(grp) < m_min:
                    continue
                proto = normalize(U[grp].mean(0, keepdims=True))[0]
                grp = np.where((U @ proto >= tau_assign) & ~used)[0]     # refine against the mean
                if len(grp) < m_min:
                    continue
                proto = normalize(U[grp].mean(0, keepdims=True))[0]
                P = np.vstack([P, proto]); P_n = np.append(P_n, len(grp))
                lc = {}
                for lab_ in M0_lab[un[grp]]:
                    lc[lab_] = lc.get(lab_, 0) + 1
                P_labcounts.append(lc)
                used[grp] = True
            M0_counted[un[used]] = True

        # ---- forgetting ------------------------------------------------------
        forgot = 0
        if forget and len(P):
            sim = M0 @ P.T
            b2 = sim.argmax(1); s2 = sim.max(1)
            drop = (s2 >= tau_forget) & (P_n[b2] >= m_min)
            forgot = int(drop.sum())
            archive_n += forgot
            M0 = M0[~drop]; M0_lab = M0_lab[~drop]
            M0_counted = M0_counted[~drop]

        # ---- rebuild upper levels --------------------------------------------
        if len(P) >= 2:
            if grow:
                R, glog = rs.grow_hierarchy(P, beta, seed=seed)
            else:
                R = rs.RHAM([rs.Level(vecs=P, parent=None)]); rs._attach_residuals(R.levels)
        upper = sum(L.vecs.shape[0] for L in R.levels[1:]) if R is not None else 0
        assoc_store = len(M0) + len(P) + upper

        # ---- bits: two-part code for the window --------------------------
        if len(P):
            simE = E @ P.T
            bj = simE.argmax(1); ok = simE.max(1) >= tau_assign
            res_var = ((E[ok] - P[bj[ok]]) ** 2).mean() if ok.any() else raw_var
            frac_ok = ok.mean()
            # concept index: empirical entropy of the assignments
            pj = np.bincount(bj[ok], minlength=len(P)) / max(1, ok.sum())
            Hj = -(pj[pj > 0] * np.log2(pj[pj > 0])).sum()
            data_bits = frac_ok * (Hj + gauss_bits(res_var, D, d)) + (1 - frac_ok) * raw_bits
        else:
            data_bits = raw_bits
        model_bits = (len(P) - n_proto_before) * d * 32 / S
        hier_bits = data_bits + model_bits
        h_true, Hc = true_rate(active)

        # ---- retrieval probe: concept top-1 for old and new concepts -----------
        def probe(mask):
            if not mask.any():
                return None, None
            p = w * mask; p = p / p.sum()
            cc = rng.choice(nC, size=probe_n, p=p)
            Q = normalize(C[cc] + normalize(rng.standard_normal((probe_n, d))) * sigma)
            proto_lab = np.array([max(lc, key=lc.get) if lc else -1 for lc in P_labcounts]) if len(P) else np.zeros(0, int)
            score_P = np.full(probe_n, -np.inf); lab_P = np.full(probe_n, -1)
            cost = np.zeros(probe_n)
            if R is not None and len(P):
                idx, _, cst = rs.rham_retrieve(R, Q, beta, beam=2, mode="exact", beta_final=64)
                score_P = np.einsum("ij,ij->i", Q, P[idx]); lab_P = proto_lab[idx]; cost += cst
            if len(M0):
                sM = Q @ M0.T
                jm = sM.argmax(1); score_M = sM.max(1); lab_M = M0_lab[jm]; cost += len(M0)
            else:
                score_M = np.full(probe_n, -np.inf); lab_M = np.full(probe_n, -1)
            pred = np.where(score_P >= score_M, lab_P, lab_M)
            return float((pred == cc).mean()), float(cost.mean())

        acc_old, cost_old = probe(is_old)
        acc_new, _ = probe(~is_old & active) if (~is_old & active).any() else (None, None)

        entry = {"t": t, "M0": int(len(M0)), "M1": int(len(P)), "upper": int(upper),
                 "depth": (R.depth + 1) if R is not None else 1,   # +1 for M0
                 "assoc_store": int(assoc_store), "flat_store": int(flat_store),
                 "archive": int(archive_n), "forgot_now": forgot, "new_protos": int(len(P) - n_proto_before),
                 "bits_raw": round(raw_bits, 2), "bits_rham": round(hier_bits, 2),
                 "bits_model": round(model_bits, 2), "h_true": round(h_true, 2), "H_concepts": round(Hc, 2),
                 "acc_old": acc_old, "acc_new": acc_new, "probe_cost": cost_old}
        log.append(entry)
        print(f"E5 s={seed} t={t:5d} M0={len(M0):4d} M1={len(P):3d} depth={entry['depth']} "
              f"assoc={assoc_store:5d} flat={flat_store:5d} bits rham/raw/h={hier_bits:6.1f}/{raw_bits:6.1f}/{h_true:6.1f} "
              f"acc old/new={acc_old}/{acc_new}", flush=True)
    return log


def flat_baseline_acc(seed, K=6, h=3, d=64, T=20000, sigma=0.4, zipf_a=1.1, probe_n=300):
    """Flat memory holding all T episodes: concept top-1 via exact NN."""
    rng = np.random.default_rng(seed)
    C, branch = make_concepts(K, h, d, rng)
    nC = len(C)
    w = 1.0 / np.arange(1, nC + 1) ** zipf_a; w = w[rng.permutation(nC)]
    p = w / w.sum()
    c = rng.choice(nC, size=T, p=p)
    E = normalize(C[c] + normalize(rng.standard_normal((T, d))) * sigma)
    cc = rng.choice(nC, size=probe_n, p=p)
    Q = normalize(C[cc] + normalize(rng.standard_normal((probe_n, d))) * sigma)
    return float((c[(Q @ E.T).argmax(1)] == cc).mean())


if __name__ == "__main__":
    seeds = [0, 1, 2, 3, 4]
    which = sys.argv[1] if len(sys.argv) > 1 else "E5"
    t0 = time.time()
    out = {}
    if which == "E5":
        out["E5"] = {str(s): run_stream(s) for s in seeds}
        out["flat_acc"] = [flat_baseline_acc(s) for s in seeds]
    elif which == "E5_noforget":
        out["E5_noforget"] = {str(s): run_stream(s, forget=False) for s in seeds[:3]}
    out["runtime_s"] = round(time.time() - t0, 1)
    json.dump(out, open(f"results_{which}.json", "w"), indent=1)
    print("done", out["runtime_s"])
