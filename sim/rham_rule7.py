"""
RHAM - Rule 7 (prototype aging) and anisotropic noise, experiment E8 (as of 2026-10-07)

OnlineRHAM7 extends OnlineRHAM (rham_hard.py) with
  7a capped learning rate (n_cap, already present)
  7b age limit: episodes folded into a prototype go to the archive after
     age_limit sleep phases, even if reconstruction is below tau_forget
  7c retirement: prototypes with no assignment for retire_after sleep phases
     become "cold" (not deleted). They are reactivated on a new assignment;
     retrieval falls back to cold prototypes when the best hot match is
     < tau_assign (cost counted separately)
  7d merging: pairs of mutually nearest active prototypes with cosine
     >= tau_assign are merged when their combined episodes FAIL the split
     test (statistic <= merge_margin < split_margin)
and adds a second null model for the split test:
  split_null="iso": one concept + isotropic noise (round 3)
  split_null="cov": Gaussian with the data's estimated covariance (+ ridge)

Prototype states: 0 active, 1 cold (retired), 2 dead (merged away; row = 0)
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
from sklearn.cluster import KMeans

import rham_sim as rs
from rham_hard import OnlineRHAM, concepts, zipf_weights

normalize = rs.normalize


class OnlineRHAM7(OnlineRHAM):
    def __init__(self, d, seed, age_limit=None, retire_after=None, merge=False,
                 merge_margin=0.05, split_null="iso", **kw):
        super().__init__(d, seed, **kw)
        self.age_limit, self.retire_after, self.merge = age_limit, retire_after, merge
        self.merge_margin, self.split_null = merge_margin, split_null
        self.sleep_idx = 0
        self.M0_birth = np.zeros(0, int)
        self.state = np.zeros(0, int); self.last = np.zeros(0, int)
        self.hot_idx = np.zeros(0, int)
        self.n_merges = 0; self.n_splits = 0; self.n_revived = 0

    # ------------------------------------------------------------ helpers
    def observe(self, E, lab, ids, tag=None):
        super().observe(E, lab, ids, tag)
        self.M0_birth = np.concatenate([self.M0_birth, np.full(len(E), self.sleep_idx)])

    def _pad(self):
        k = len(self.P) - len(self.state)
        if k > 0:
            self.state = np.r_[self.state, np.zeros(k, int)]
            self.last = np.r_[self.last, np.full(k, self.sleep_idx)]

    def _filter_M0(self, keep):
        for a in ("M0", "M0_lab", "M0_id", "M0_tag", "M0_counted", "M0_birth"):
            setattr(self, a, getattr(self, a)[keep])

    def _split_stat(self, V):
        def red(X):
            w1 = ((X - X.mean(0)) ** 2).sum()
            km = KMeans(2, n_init=3, random_state=self.seed, max_iter=60).fit(X)
            return np.log(w1) - np.log(km.inertia_), km
        if self.split_null == "gmeans":
            # G-means (Hamerly & Elkan 2003), with cross-fitting: project onto the
            # axis of the two 2-means centers, Anderson-Darling normality test.
            # Returns A*2 - critical value (alpha = 0.0001: 1.8692) -> > 0 means
            # "not normal" = split. Cross-fitting guards against selection bias:
            # axis fit on half A, test evaluated on half B.
            perm = self.rng.permutation(len(V)); A, B = V[perm[: len(V) // 2]], V[perm[len(V) // 2:]]
            kmA = KMeans(2, n_init=3, random_state=self.seed, max_iter=60).fit(A)
            v = kmA.cluster_centers_[0] - kmA.cluster_centers_[1]
            x = B @ v / (v @ v)
            km = KMeans(2, n_init=3, random_state=self.seed, max_iter=60).fit(V)
            x = np.sort((x - x.mean()) / x.std(ddof=1))
            n = len(x)
            from scipy.stats import norm
            F = np.clip(norm.cdf(x), 1e-12, 1 - 1e-12)
            i = np.arange(1, n + 1)
            A2 = -n - np.mean((2 * i - 1) * (np.log(F) + np.log(1 - F[::-1])))
            A2s = A2 * (1 + 4 / n - 25 / n ** 2)
            return A2s - 1.8692, km
        r_data, km = red(V)
        mu = V.mean(0)
        if self.split_null == "cov":
            Cv = np.cov(V.T) + 1e-4 * np.eye(V.shape[1])
            Lc = np.linalg.cholesky(Cv)
            nulls = [mu + self.rng.standard_normal(V.shape) @ Lc.T for _ in range(2)]
        else:
            sd = np.sqrt(((V - mu) ** 2).mean())
            nulls = [mu + sd * self.rng.standard_normal(V.shape) for _ in range(2)]
        return r_data - np.mean([red(Z)[0] for Z in nulls]), km

    # ------------------------------------------------------------ sleep
    def sleep(self):
        self.sleep_idx += 1
        self._pad()
        M0 = self.M0
        if len(self.P):
            sim = M0 @ self.P.T; best = sim.argmax(1); assigned = sim.max(1) >= self.tau_a
        else:
            best = np.zeros(len(M0), int); assigned = np.zeros(len(M0), bool)
        newly = assigned & ~self.M0_counted
        for j in np.unique(best[newly]):
            m = newly & (best == j)
            self._update(j, M0[m], self.M0_lab[m])
            self.last[j] = self.sleep_idx
            if self.state[j] == 1:
                self.state[j] = 0; self.n_revived += 1
        self.M0_counted |= assigned
        # new prototypes (as in the base class)
        un = np.where(~self.M0_counted)[0]
        if len(un) >= self.m_min:
            U = M0[un]; used = np.zeros(len(un), bool)
            for i in self.rng.permutation(len(un)):
                if used[i]:
                    continue
                grp = np.where((U @ U[i] >= self.tau_a) & ~used)[0]
                if len(grp) < self.m_min:
                    continue
                proto = normalize(U[grp].mean(0, keepdims=True))[0]
                grp = np.where((U @ proto >= self.tau_a) & ~used)[0]
                if len(grp) < self.m_min:
                    continue
                proto = normalize(U[grp].mean(0, keepdims=True))[0]
                self.P = np.vstack([self.P, proto]); self.P_n = np.append(self.P_n, len(grp))
                lc = {}
                for l in self.M0_lab[un[grp]]:
                    lc[l] = lc.get(l, 0) + 1
                self.P_lab.append(lc); self.archive.append([])
                used[grp] = True
            self.M0_counted[un[used]] = True
        self._pad()
        # rule 6: splitting (active prototypes only)
        if self.split and len(self.P):
            self._split_pass_active()
        # rule 7d: merging
        if self.merge:
            self._merge_pass()
        # rule 5 + 7b: forgetting -> archive
        if self.do_forget and len(self.P):
            sim = self.M0 @ self.P.T; b = sim.argmax(1); s = sim.max(1)
            drop = (s >= self.tau_f) & (self.P_n[b] >= self.m_min)
            if self.age_limit is not None:
                drop |= self.M0_counted & (self.sleep_idx - self.M0_birth >= self.age_limit)
            if self.protect:
                drop &= ~self.M0_tag
            for i in np.where(drop)[0]:
                self.archive[b[i]].append((int(self.M0_id[i]), self.M0[i].copy(), int(self.M0_lab[i])))
            self._filter_M0(~drop)
        # rule 7c: retirement
        if self.retire_after is not None:
            old = (self.state == 0) & (self.sleep_idx - self.last >= self.retire_after)
            self.state[old] = 1
        # upper levels over active prototypes
        self.hot_idx = np.where(self.state == 0)[0]
        self.R = rs.grow_hierarchy(self.P[self.hot_idx], self.beta, seed=self.seed)[0] if len(self.hot_idx) >= 2 else None

    def _split_pass_active(self):
        j = 0
        while j < len(self.P):
            if self.state[j] != 0:
                j += 1; continue
            V, L, src = self._members(j)
            if len(V) >= self.split_min:
                stat, km = self._split_stat(V[-300:] if len(V) > 300 else V)
                if stat > self.split_margin:
                    lab2 = km.predict(V)
                    if min((lab2 == 0).sum(), (lab2 == 1).sum()) >= self.m_min:
                        self.P[j] = normalize(V[lab2 == 0].mean(0, keepdims=True))[0]
                        self.P = np.vstack([self.P, normalize(V[lab2 == 1].mean(0, keepdims=True))[0]])
                        self.P_n[j] = (lab2 == 0).sum(); self.P_n = np.append(self.P_n, (lab2 == 1).sum())
                        lc0, lc1 = {}, {}
                        for l, g in zip(L, lab2):
                            dd = lc0 if g == 0 else lc1
                            dd[l] = dd.get(l, 0) + 1
                        self.P_lab[j] = lc0; self.P_lab.append(lc1)
                        self.archive[j] = [x[1] for x, g in zip(src, lab2) if x[0] == "A" and g == 0]
                        self.archive.append([x[1] for x, g in zip(src, lab2) if x[0] == "A" and g == 1])
                        self._pad(); self.last[-1] = self.last[j]
                        self.n_splits += 1
                        continue
            j += 1

    def _merge_pass(self):
        act = np.where(self.state == 0)[0]
        if len(act) < 2:
            return
        S = self.P[act] @ self.P[act].T; np.fill_diagonal(S, -1)
        nn = S.argmax(1)
        done = set()
        for a, b in enumerate(nn):
            if nn[b] != a or a in done or b in done or S[a, b] < self.tau_a:
                continue
            i, j = act[a], act[b]
            Vi, Li, _ = self._members(i); Vj, Lj, _ = self._members(j)
            if len(Vi) < self.m_min or len(Vj) < self.m_min:
                continue
            V = np.vstack([Vi[-150:], Vj[-150:]])
            stat, _ = self._split_stat(V)
            if stat <= self.merge_margin:
                w_i, w_j = self.P_n[i], self.P_n[j]
                self.P[i] = normalize((self.P[i] * w_i + self.P[j] * w_j)[None])[0]
                self.P_n[i] = w_i + w_j
                for l, c in self.P_lab[j].items():
                    self.P_lab[i][l] = self.P_lab[i].get(l, 0) + c
                self.archive[i].extend(self.archive[j]); self.archive[j] = []
                self.P[j] = 0.0; self.P_n[j] = 0; self.P_lab[j] = {}; self.state[j] = 2
                self.last[i] = max(self.last[i], self.last[j])
                self.n_merges += 1
                done.update((a, b))

    # ------------------------------------------------------------ retrieval
    def retrieve_concept(self, Q):
        n = len(Q); cost = np.zeros(n); cold = np.zeros(n)
        labs = self.proto_labels()
        sP = np.full(n, -np.inf); lP = np.full(n, -1)
        if self.R is not None:
            idx, _, c = rs.rham_retrieve(self.R, Q, self.beta, beam=2, mode="exact", beta_final=64)
            gi = self.hot_idx[idx]
            sP = np.einsum("ij,ij->i", Q, self.P[gi]); lP = labs[gi]; cost += c
        if len(self.M0):
            S = Q @ self.M0.T; sM = S.max(1); lM = self.M0_lab[S.argmax(1)]; cost += len(self.M0)
        else:
            sM = np.full(n, -np.inf); lM = np.full(n, -1)
        score = np.maximum(sP, sM); pred = np.where(sP >= sM, lP, lM)
        cold_idx = np.where(self.state == 1)[0]
        if len(cold_idx):
            need = score < self.tau_a
            if need.any():
                SC = Q[need] @ self.P[cold_idx].T
                better = SC.max(1) > score[need]
                pr = pred[need]; pr[better] = labs[cold_idx[SC.argmax(1)[better]]]
                pred[need] = pr; cold[need] += len(cold_idx)
        return pred, cost, cold

    def sizes(self):
        up = sum(L.vecs.shape[0] for L in self.R.levels[1:]) if self.R is not None else 0
        return {"M0": int(len(self.M0)), "hot": int((self.state == 0).sum()), "cold": int((self.state == 1).sum()),
                "upper": int(up), "assoc": int(len(self.M0) + (self.state == 0).sum() + up)}


# ============================================================================
VARIANTS = {
    "baseline (1/n)":     dict(),
    "capped":             dict(n_cap=30),
    "rule 7 complete":    dict(n_cap=30, age_limit=3, retire_after=5, merge=True),
}


def E8_drift(seeds, deltas=(0.0, 0.1, 0.2, 0.3), K=6, h=3, d=64, T=20000, S=1000, sigma=0.4, variants=None):
    variants = variants or list(VARIANTS)
    rows = []
    for dl in deltas:
        for vname in variants:
            for s in seeds:
                rng = np.random.default_rng(s)
                C = concepts(K, h, d, rng); p = zipf_weights(len(C), 1.1, rng)
                rank = np.argsort(np.argsort(-p))            # 0 = most frequent concept
                head = rank < int(0.2 * len(C)); tail = rank >= int(0.5 * len(C))
                M = OnlineRHAM7(d, s, **VARIANTS[vname]); nid = 0; traj = []
                for t in range(0, T, S):
                    c = rng.choice(len(C), size=S, p=p)
                    E = normalize(C[c] + normalize(rng.standard_normal((S, d))) * sigma)
                    M.observe(E, c, np.arange(nid, nid + S)); nid += S
                    M.sleep()
                    # probe: 300 queries each from head and tail concepts (uniform within)
                    res = {}
                    for name, mask in (("head", head), ("tail", tail)):
                        cq = rng.choice(np.where(mask)[0], 300)
                        Q = normalize(C[cq] + normalize(rng.standard_normal((300, d))) * sigma)
                        pred, cost, cold = M.retrieve_concept(Q)
                        res[f"acc_{name}"] = float((pred == cq).mean())
                        res[f"cost_{name}"] = float(cost.mean()); res[f"cold_{name}"] = float(cold.mean())
                    traj.append({"t": t + S, **res, **M.sizes(), "merges": M.n_merges, "revived": M.n_revived})
                    if dl > 0:
                        C = normalize(C + normalize(rng.standard_normal(C.shape)) * dl)
                last = traj[-1]
                rows.append({"delta": dl, "variant": vname, "seed": s, "traj": traj})
                print(f"E8drift d={dl} {vname:18s} s={s} head={last['acc_head']:.3f} tail={last['acc_tail']:.3f} "
                      f"M0={last['M0']} hot={last['hot']} cold={last['cold']} assoc={last['assoc']} "
                      f"cost={last['cost_head']:.0f}/{last['cost_tail']:.0f}+cold {last['cold_tail']:.0f} "
                      f"merges={last['merges']} revived={last['revived']}", flush=True)
    return rows


def anisotropic_episodes(C, U, p, n, sigma, alpha, rng):
    """Noise per concept stretched by factor alpha along its own direction
    u_c, total variance matched to the isotropic case."""
    d = C.shape[1]
    c = rng.choice(len(C), size=n, p=p)
    g = rng.standard_normal((n, d))
    proj = np.einsum("ij,ij->i", g, U[c])
    g = g + (alpha - 1) * proj[:, None] * U[c]
    g = g * sigma / np.sqrt(d + alpha ** 2 - 1)
    return normalize(C[c] + g), c


def E8_aniso(seeds, alphas=(1.0, 4.0, 8.0), scales=(1.0, 0.6), K=6, h=3, d=64, T=8000, S=1000, sigma_norm=0.4, only=None):
    sigma = sigma_norm  # expected noise norm, as in the isotropic case (0.4)
    rows = []
    variants = [("no split", dict(split=False)),
                ("split, iso-null", dict(split=True, split_null="iso")),
                ("split, cov-null", dict(split=True, split_null="cov")),
                ("split, G-means", dict(split=True, split_null="gmeans", split_margin=0.0, merge_margin=-1.0))]
    for a in alphas:
        for sc in scales:
            for vname, kw in variants:
                if only and vname != only:
                    continue
                for s in seeds:
                    rng = np.random.default_rng(s)
                    C = concepts(K, h, d, rng, sc); p = zipf_weights(len(C), 1.1, rng)
                    Ud = normalize(rng.standard_normal(C.shape))
                    Ud = normalize(Ud - np.einsum("ij,ij->i", Ud, C)[:, None] * C)   # tangential
                    M = OnlineRHAM7(d, s, **kw); nid = 0
                    for t in range(0, T, S):
                        E, c = anisotropic_episodes(C, Ud, p, S, sigma, a, rng)
                        M.observe(E, c, np.arange(nid, nid + S)); nid += S
                        M.sleep()
                    Q, cq = anisotropic_episodes(C, Ud, p, 1000, sigma, a, rng)
                    pred, cost, _ = M.retrieve_concept(Q)
                    covered = len(set(M.proto_labels()[M.state == 0].tolist()) - {-1})
                    row = {"alpha": a, "scale": sc, "variant": vname, "seed": s,
                           "acc": float((pred == cq).mean()), "oracle": float(((Q @ C.T).argmax(1) == cq).mean()),
                           "n_proto": int((M.state == 0).sum()), "covered": covered, "splits": M.n_splits,
                           "cost": float(cost.mean()), "assoc": M.sizes()["assoc"]}
                    rows.append(row); print("E8aniso", row, flush=True)
    return rows


if __name__ == "__main__":
    which = sys.argv[1]
    seeds = [0, 1, 2]
    t0 = time.time()
    if which == "drift":
        out = {"drift": E8_drift(seeds)}
    elif which.startswith("drift_"):          # drift_0.2 etc. for parallelization
        out = {which: E8_drift(seeds, deltas=(float(which.split("_")[1]),))}
    elif which == "aniso":
        out = {"aniso": E8_aniso(seeds)}
    elif which.startswith("gmeans_"):        # recompute only the G-means variant
        out = {which: E8_aniso(seeds, alphas=(float(which.split("_")[1]),), only="split, G-means")}
    elif which.startswith("aniso_"):
        out = {which: E8_aniso(seeds, alphas=(float(which.split("_")[1]),))}
    out["runtime_s"] = round(time.time() - t0, 1)
    json.dump(out, open(f"results_E8_{which}.json", "w"), indent=1)
    print("done", out["runtime_s"])
