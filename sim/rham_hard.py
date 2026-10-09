"""
RHAM - stress test E7 (as of 2026-10-07)

S1  Overlapping concepts (siblings more similar than two episodes of the same concept)
S2  Concept drift (centers wander) - running mean vs. capped learning rate
S3  Rare single episodes - exact episode retrieval: associative, via the archive
    (three-layer model), with "important" tagging

Memory model as in E5 (rham_online.py), here as the class OnlineRHAM with an
archive: forgotten episodes move to the prototype that reconstructs them and
remain retrievable there as a raw vector + ID ("cold" access, counted separately).
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

import rham_sim as rs

normalize = rs.normalize


class OnlineRHAM:
    def __init__(self, d, seed, tau_assign=0.80, tau_forget=0.85, m_min=3, beta=16,
                 n_cap=None, protect_tagged=False, forget=True, split=False, split_margin=0.1, split_min=12):
        self.d, self.rng = d, np.random.default_rng(seed + 7)
        self.seed = seed
        self.tau_a, self.tau_f, self.m_min, self.beta = tau_assign, tau_forget, m_min, beta
        self.n_cap, self.protect, self.do_forget = n_cap, protect_tagged, forget
        self.split, self.split_margin, self.split_min = split, split_margin, split_min
        self.M0 = np.zeros((0, d)); self.M0_lab = np.zeros(0, int); self.M0_id = np.zeros(0, int)
        self.M0_tag = np.zeros(0, bool); self.M0_counted = np.zeros(0, bool)
        self.P = np.zeros((0, d)); self.P_n = np.zeros(0); self.P_lab = []
        self.archive = []                     # per prototype: list of (id, vec)
        self.R = None

    # ---------------------------------------------------------------- wake phase
    def observe(self, E, lab, ids, tag=None):
        tag = np.zeros(len(E), bool) if tag is None else tag
        self.M0 = np.vstack([self.M0, E]); self.M0_lab = np.concatenate([self.M0_lab, lab])
        self.M0_id = np.concatenate([self.M0_id, ids]); self.M0_tag = np.concatenate([self.M0_tag, tag])
        self.M0_counted = np.concatenate([self.M0_counted, np.zeros(len(E), bool)])

    # ---------------------------------------------------------------- sleep
    def _update(self, j, X, labs):
        n_old = self.P_n[j] if self.n_cap is None else min(self.P_n[j], self.n_cap)
        tot = n_old + len(X)
        self.P[j] = normalize(self.P[j] * n_old / tot + X.sum(0) / tot)
        self.P_n[j] += len(X)
        for l in labs:
            self.P_lab[j][l] = self.P_lab[j].get(l, 0) + 1

    def sleep(self):
        M0 = self.M0
        if len(self.P):
            sim = M0 @ self.P.T; best = sim.argmax(1); assigned = sim.max(1) >= self.tau_a
        else:
            best = np.zeros(len(M0), int); assigned = np.zeros(len(M0), bool)
        newly = assigned & ~self.M0_counted
        for j in np.unique(best[newly]):
            m = newly & (best == j)
            self._update(j, M0[m], self.M0_lab[m])
        self.M0_counted |= assigned
        # new prototypes (DP-means-like)
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
        # split rule: split a prototype if its episodes form two groups
        if self.split and len(self.P):
            self._split_pass()
        # forgetting -> archive
        if self.do_forget and len(self.P):
            sim = self.M0 @ self.P.T; b = sim.argmax(1); s = sim.max(1)
            drop = (s >= self.tau_f) & (self.P_n[b] >= self.m_min)
            if self.protect:
                drop &= ~self.M0_tag
            for i in np.where(drop)[0]:
                self.archive[b[i]].append((int(self.M0_id[i]), self.M0[i].copy(), int(self.M0_lab[i])))
            keep = ~drop
            for a in ("M0", "M0_lab", "M0_id", "M0_tag", "M0_counted"):
                setattr(self, a, getattr(self, a)[keep])
        # upper levels
        if len(self.P) >= 2:
            self.R, _ = rs.grow_hierarchy(self.P, self.beta, seed=self.seed)


    def _members(self, j):
        """Episodes of a prototype: archive + assigned M0 episodes."""
        V, L, src = [], [], []
        for a in self.archive[j]:
            V.append(a[1]); L.append(a[2]); src.append(("A", a))
        if len(self.M0) and len(self.P):
            b = (self.M0 @ self.P.T).argmax(1)
            for i in np.where((b == j) & self.M0_counted)[0]:
                V.append(self.M0[i]); L.append(self.M0_lab[i]); src.append(("M", i))
        return (np.array(V) if V else np.zeros((0, self.d))), np.array(L, int), src

    def _split_stat(self, V):
        """log W(1) - log W(2) of the data minus the same value for a null
        model of 'one concept + isotropic noise' with equal spread (gap logic)."""
        from sklearn.cluster import KMeans
        def red(X):
            w1 = ((X - X.mean(0)) ** 2).sum()
            km = KMeans(2, n_init=3, random_state=self.seed, max_iter=60).fit(X)
            return np.log(w1) - np.log(km.inertia_), km
        r_data, km = red(V)
        mu = V.mean(0); sd = np.sqrt(((V - mu) ** 2).mean())
        r_null = np.mean([red(mu + sd * self.rng.standard_normal(V.shape))[0] for _ in range(2)])
        return r_data - r_null, km

    def _split_pass(self):
        j = 0
        while j < len(self.P):
            V, L, src = self._members(j)
            if len(V) >= self.split_min:
                stat, km = self._split_stat(V[-300:] if len(V) > 300 else V)
                if stat > self.split_margin:
                    lab2 = km.predict(V)
                    if min((lab2 == 0).sum(), (lab2 == 1).sum()) >= self.m_min:
                        c0 = normalize(V[lab2 == 0].mean(0, keepdims=True))[0]
                        c1 = normalize(V[lab2 == 1].mean(0, keepdims=True))[0]
                        new = len(self.P)
                        self.P[j] = c0
                        self.P = np.vstack([self.P, c1])
                        self.P_n[j] = (lab2 == 0).sum(); self.P_n = np.append(self.P_n, (lab2 == 1).sum())
                        lc0, lc1 = {}, {}
                        for l, g in zip(L, lab2):
                            d_ = lc0 if g == 0 else lc1
                            d_[l] = d_.get(l, 0) + 1
                        self.P_lab[j] = lc0; self.P_lab.append(lc1)
                        a0 = [x[1] for x, g in zip(src, lab2) if x[0] == "A" and g == 0]
                        a1 = [x[1] for x, g in zip(src, lab2) if x[0] == "A" and g == 1]
                        self.archive[j] = a0; self.archive.append(a1)
                        self.n_splits = getattr(self, "n_splits", 0) + 1
                        continue          # re-check the same prototype
            j += 1

    # ---------------------------------------------------------------- retrieval
    def proto_labels(self):
        return np.array([max(lc, key=lc.get) if lc else -1 for lc in self.P_lab])

    def retrieve_concept(self, Q):
        n = len(Q); cost = np.zeros(n)
        sP = np.full(n, -np.inf); lP = np.full(n, -1)
        if self.R is not None:
            idx, _, c = rs.rham_retrieve(self.R, Q, self.beta, beam=2, mode="exact", beta_final=64)
            sP = np.einsum("ij,ij->i", Q, self.P[idx]); lP = self.proto_labels()[idx]; cost += c
        if len(self.M0):
            S = Q @ self.M0.T; sM = S.max(1); lM = self.M0_lab[S.argmax(1)]; cost += len(self.M0)
        else:
            sM = np.full(n, -np.inf); lM = np.full(n, -1)
        return np.where(sP >= sM, lP, lM), cost

    def recall_episode(self, Q, deep=False, n_protos=2):
        """Exact episode retrieval: returns the ID of the most similar stored
        episode. deep=True additionally searches the archive lists of the
        n_protos most similar prototypes (cold access, counted separately)."""
        n = len(Q); out = np.full(n, -1); hot = np.zeros(n); cold = np.zeros(n)
        best = np.full(n, -np.inf)
        if len(self.M0):
            S = Q @ self.M0.T; best = S.max(1); out = self.M0_id[S.argmax(1)]; hot += len(self.M0)
        if deep and len(self.P):
            SP = Q @ self.P.T; hot += len(self.P)
            top = np.argsort(-SP, 1)[:, :n_protos]
            for i in range(n):
                for j in top[i]:
                    if not self.archive[j]:
                        continue
                    ids = np.array([a[0] for a in self.archive[j]]); V = np.stack([a[1] for a in self.archive[j]])
                    s = V @ Q[i]; cold[i] += len(V)
                    if s.max() > best[i]:
                        best[i] = s.max(); out[i] = ids[s.argmax()]
        return out, hot, cold

    def assoc_size(self):
        up = sum(L.vecs.shape[0] for L in self.R.levels[1:]) if self.R is not None else 0
        return len(self.M0) + len(self.P) + up


# ============================================================================
def concepts(K, h, d, rng, last_scale=1.0):
    C, _ = rs.make_tree(K, h, d, rng, level_scale=np.r_[np.ones(h - 1), last_scale])
    return C


def zipf_weights(n, a, rng):
    w = 1.0 / np.arange(1, n + 1) ** a
    return w[rng.permutation(n)] / w.sum()


def episodes(C, p, n, sigma, rng):
    c = rng.choice(len(C), size=n, p=p)
    return normalize(C[c] + normalize(rng.standard_normal((n, C.shape[1]))) * sigma), c


def flat_concept_acc(Ef, Lf, Q, cq):
    return float((Lf[(Q @ Ef.T).argmax(1)] == cq).mean())


# ---------------------------------------------------------------- S1 overlap
def S1(seeds, scales=(1.0, 0.8, 0.6, 0.45), K=6, h=3, d=64, T=10000, S=1000, sigma=0.4, split=False):
    rows = []
    for sc in scales:
        for s in seeds:
            rng = np.random.default_rng(s)
            C = concepts(K, h, d, rng, sc); p = zipf_weights(len(C), 1.1, rng)
            sib = float(np.mean([C[i] @ C[i + 1] for i in range(0, len(C), K)]))
            M = OnlineRHAM(d, s, split=split); Ef, Lf = [], []
            nid = 0
            for t in range(0, T, S):
                E, c = episodes(C, p, S, sigma, rng)
                M.observe(E, c, np.arange(nid, nid + S)); nid += S; Ef.append(E); Lf.append(c)
                M.sleep()
            Ef = np.vstack(Ef); Lf = np.concatenate(Lf)
            Q, cq = episodes(C, p, 1000, sigma, rng)
            pred, cost = M.retrieve_concept(Q)
            oracle = float(((Q @ C.T).argmax(1) == cq).mean())          # nearest true center
            # how many true concepts share one prototype? (merging)
            pl = M.proto_labels(); covered = len(set(pl.tolist()))
            row = {"split": split, "n_splits": getattr(M, "n_splits", 0), "scale": sc, "seed": s, "sibling_cos": round(sib, 3), "oracle": oracle,
                   "rham": float((pred == cq).mean()), "flat": flat_concept_acc(Ef, Lf, Q, cq),
                   "n_proto": int(len(M.P)), "concepts_covered": covered, "n_concepts": len(C),
                   "assoc": M.assoc_size(), "cost": float(cost.mean())}
            rows.append(row); print("S1", row, flush=True)
    return rows


# ---------------------------------------------------------------- S2 drift
def S2(seeds, deltas=(0.0, 0.05, 0.1, 0.2), caps=(None, 30), K=6, h=3, d=64, T=15000, S=1000, sigma=0.4):
    rows = []
    for dl in deltas:
        for cap in caps:
            for s in seeds:
                rng = np.random.default_rng(s)
                C = concepts(K, h, d, rng); p = zipf_weights(len(C), 1.1, rng)
                M = OnlineRHAM(d, s, n_cap=cap); Ef, Lf = [], []
                nid = 0; traj = []
                for t in range(0, T, S):
                    E, c = episodes(C, p, S, sigma, rng)
                    M.observe(E, c, np.arange(nid, nid + S)); nid += S; Ef.append(E); Lf.append(c)
                    M.sleep()
                    Q, cq = episodes(C, p, 300, sigma, rng)
                    pred, _ = M.retrieve_concept(Q)
                    traj.append({"t": t + S, "rham": float((pred == cq).mean()),
                                 "flat": flat_concept_acc(np.vstack(Ef), np.concatenate(Lf), Q, cq),
                                 "n_proto": int(len(M.P)), "assoc": M.assoc_size()})
                    # drift: each center takes a step of length dl
                    if dl > 0:
                        C = normalize(C + normalize(rng.standard_normal(C.shape)) * dl)
                row = {"delta": dl, "cap": cap, "seed": s, "traj": traj,
                       "rham_end": traj[-1]["rham"], "flat_end": traj[-1]["flat"],
                       "n_proto_end": traj[-1]["n_proto"], "assoc_end": traj[-1]["assoc"]}
                rows.append(row)
                print(f"S2 delta={dl} cap={cap} seed={s} rham={row['rham_end']:.3f} flat={row['flat_end']:.3f} "
                      f"prototypes={row['n_proto_end']} assoc={row['assoc_end']}", flush=True)
    return rows


# ---------------------------------------------------------------- S3 rare episodes
def S3(seeds, K=6, h=3, d=64, T=10000, S=1000, sigma=0.4, n_rare=20, q_noise=0.1):
    rows = []
    for protect in (False, True):
        for s in seeds:
            rng = np.random.default_rng(s)
            C = concepts(K, h, d, rng); p = zipf_weights(len(C), 1.1, rng)
            M = OnlineRHAM(d, s, protect_tagged=protect)
            nid = 0; rare = []           # (id, vec, kind, timestamp)
            allE, allID = [], []
            for t in range(0, T, S):
                E, c = episodes(C, p, S, sigma, rng)
                ids = np.arange(nid, nid + S); nid += S
                # rare episodes: half "atypical" (random direction), half "typical-looking"
                far = normalize(rng.standard_normal((n_rare // 2, d)))
                near, cn = episodes(C, p, n_rare // 2, sigma, rng)
                R_ = np.vstack([far, near]); Rl = np.r_[np.full(n_rare // 2, -1), cn]
                Rid = np.arange(nid, nid + n_rare); nid += n_rare
                for k in range(n_rare):
                    rare.append((int(Rid[k]), R_[k], "atypical" if k < n_rare // 2 else "typical_looking", t))
                X = np.vstack([E, R_]); L = np.r_[c, Rl]; I = np.r_[ids, Rid]
                tag = np.r_[np.zeros(S, bool), np.ones(n_rare, bool)]
                M.observe(X, L, I, tag); allE.append(X); allID.append(I)
                M.sleep()
            allE = np.vstack(allE); allID = np.concatenate(allID)
            rid = np.array([r[0] for r in rare]); rv = np.stack([r[1] for r in rare])
            kind = np.array([r[2] for r in rare]); age = np.array([T - r[3] for r in rare])
            Q = normalize(rv + normalize(rng.standard_normal(rv.shape)) * q_noise)
            hot_id, hot_c, _ = M.recall_episode(Q, deep=False)
            deep_id, deep_hot, deep_cold = M.recall_episode(Q, deep=True)
            flat_id = allID[(Q @ allE.T).argmax(1)]
            for kd in ("atypical", "typical_looking"):
                m = kind == kd
                row = {"protect": protect, "seed": s, "kind": kd,
                       "assoc_recall": float((hot_id[m] == rid[m]).mean()),
                       "deep_recall": float((deep_id[m] == rid[m]).mean()),
                       "flat_recall": float((flat_id[m] == rid[m]).mean()),
                       "hot_cost": float(deep_hot[m].mean()), "cold_cost": float(deep_cold[m].mean()),
                       "flat_cost": int(len(allE)), "M0": int(len(M.M0)), "assoc": M.assoc_size()}
                rows.append(row); print("S3", row, flush=True)
            # additionally: retrieve ordinary episodes exactly (control)
            pick = rng.choice(len(allE), 300, replace=False)
            Qo = normalize(allE[pick] + normalize(rng.standard_normal((300, d))) * q_noise)
            h_, _, _ = M.recall_episode(Qo, deep=False); dp_, dh_, dc_ = M.recall_episode(Qo, deep=True)
            rows.append({"protect": protect, "seed": s, "kind": "ordinary",
                         "assoc_recall": float((h_ == allID[pick]).mean()),
                         "deep_recall": float((dp_ == allID[pick]).mean()),
                         "flat_recall": float((allID[(Qo @ allE.T).argmax(1)] == allID[pick]).mean()),
                         "hot_cost": float(dh_.mean()), "cold_cost": float(dc_.mean()),
                         "flat_cost": int(len(allE)), "M0": int(len(M.M0)), "assoc": M.assoc_size()})
            print("S3", rows[-1], flush=True)
    return rows


if __name__ == "__main__":
    which = sys.argv[1]
    seeds = [0, 1, 2]
    t0 = time.time()
    fn = {"S1": S1, "S2": S2, "S3": S3, "S1split": lambda sd: S1(sd, split=True)}[which]
    out = {which: fn(seeds)}
    out["runtime_s"] = round(time.time() - t0, 1)
    json.dump(out, open(f"results_E7_{which}.json", "w"), indent=1)
    print("done", out["runtime_s"])
