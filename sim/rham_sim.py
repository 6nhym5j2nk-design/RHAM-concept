"""
RHAM – Minimalsimulation (Stand 2026-10-07)

Vergleicht einen flachen modernen Hopfield-Speicher (= Softmax-Attention,
Ramsauer et al. 2020) mit einer RHAM-Hierarchie auf synthetischen Baumdaten.

Experimente
  E1  Abrufgenauigkeit vs. Speichergröße N (P1), bei festem beta und bei
      für den flachen Speicher optimal gewähltem beta
  E2  Rechenaufwand pro Abruf (Skalarprodukte)
  E3  Automatisches Ebenenwachstum: Interferenz-Trigger + Gap-Statistik (P2, P3)
  E4  Selektives Vergessen: Rekonstruktionsregel vs. Zufall

Alle Vektoren liegen auf der Einheitssphäre. Ein Abruf gilt als erfolgreich,
wenn nach einem Hopfield-Update-Schritt das maximale Attention-Gewicht auf
dem richtigen Muster liegt UND > 0.9 ist (scharfer Abruf, keine Mischung).
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import KMeans

# ----------------------------------------------------------------------------
# Daten
# ----------------------------------------------------------------------------

def normalize(x, axis=-1):
    # Nullvektoren (z. B. Residuum eines Ein-Element-Clusters) bleiben null,
    # statt NaN zu erzeugen (Bugfix 2026-10-07, siehe Fehlerlog).
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return np.divide(x, n, out=np.zeros_like(x, dtype=float), where=n > 1e-12)


def make_tree(K: int, h: int, d: int, rng, level_scale=None, leaf_jitter=False):
    """Baum mit Verzweigung K und Tiefe h. Jedes Blatt = Summe zufälliger
    Einheitsvektoren entlang seines Pfades (eine Komponente pro Ebene).
    Rückgabe: Blätter (K^h x d, normiert) und Pfad-Labels (K^h x h)."""
    if level_scale is None:
        level_scale = np.ones(h)
    n = K ** h
    labels = np.array(np.unravel_index(np.arange(n), [K] * h)).T  # n x h
    X = np.zeros((n, d))
    # Komponenten je Ebene: für jeden Präfix ein eigener Zufallsvektor
    for lvl in range(h):
        n_nodes = K ** (lvl + 1)
        comps = normalize(rng.standard_normal((n_nodes, d)))
        prefix_id = np.ravel_multi_index(labels[:, : lvl + 1].T, [K] * (lvl + 1))
        scale = level_scale[lvl]
        if leaf_jitter and lvl == h - 1:
            # variable Residuengröße der Blätter (für E4)
            scale = scale * rng.lognormal(0.0, 0.6, size=(n, 1))
        X += scale * comps[prefix_id]
    return normalize(X), labels


def corrupt(X, eta, rng):
    noise = normalize(rng.standard_normal(X.shape)) * eta
    return normalize(X + noise)


# ----------------------------------------------------------------------------
# Flacher moderner Hopfield-Speicher
# ----------------------------------------------------------------------------

def softmax_rows(S):
    S = S - S.max(axis=1, keepdims=True)
    E = np.exp(S)
    return E / E.sum(axis=1, keepdims=True)


def flat_retrieve(X, Q, beta):
    """Ein Hopfield-Schritt xi_new = X^T softmax(beta X q); danach Attention
    des neuen Zustands als Erfolgsmaß. Rückgabe: (idx, maxweight)."""
    A = softmax_rows(beta * Q @ X.T)          # nq x N
    Xi = normalize(A @ X)                      # neuer Zustand
    A2 = softmax_rows(beta * Xi @ X.T)
    return A2.argmax(1), A2.max(1)


# ----------------------------------------------------------------------------
# RHAM-Hierarchie
# ----------------------------------------------------------------------------

@dataclass
class Level:
    vecs: np.ndarray            # Rohvektoren dieser Ebene (normiert)
    parent: np.ndarray | None   # Index des Eltern-Prototyps in der Ebene darüber
    res_keys: np.ndarray | None = None  # normierte Residuen relativ zum Eltern-Prototyp


@dataclass
class RHAM:
    levels: list = field(default_factory=list)   # levels[0] = M0 (Blätter)

    @property
    def depth(self):
        return len(self.levels)


def kmeans(X, k, seed):
    km = KMeans(n_clusters=k, n_init=3, random_state=seed, max_iter=100)
    lab = km.fit_predict(X)
    return lab, km.inertia_


def build_hierarchy_fixed(X, cluster_counts, seed):
    """Hierarchie mit vorgegebener Clusterzahl je Ebene (Prototypen gelernt)."""
    levels = [Level(vecs=X, parent=None)]
    cur = X
    for k in cluster_counts:
        lab, _ = kmeans(cur, k, seed)
        protos = normalize(np.stack([cur[lab == c].mean(0) for c in range(k)]))
        levels[-1].parent = lab
        levels.append(Level(vecs=protos, parent=None))
        cur = protos
    _attach_residuals(levels)
    return RHAM(levels)


def _attach_residuals(levels):
    for i, L in enumerate(levels):
        if L.parent is None:
            L.res_keys = L.vecs.copy()   # oberste Ebene: Rohschlüssel
        else:
            P = levels[i + 1].vecs[L.parent]
            L.res_keys = normalize(L.vecs - P)


def _group_log_softmax(s, groups):
    out = np.empty_like(s)
    for g in np.unique(groups):
        m = groups == g
        z = s[m] - s[m].max()
        out[m] = z - np.log(np.exp(z).sum())
    return out


def rham_retrieve(R: RHAM, Q, beta, beam=2, mode="hsoftmax", beta_final=None):
    """mode='exact' (v3): Routing wie v2, Endschritt auf Rohvektoren der
    Kandidaten (exakte Zerlegung, keine Rauschverstärkung).
    mode='hsoftmax' (v2): Softmax je Geschwistergruppe, Pfad-
    wahrscheinlichkeiten multipliziert (hierarchische Softmax). Kandidaten
    verschiedener Eltern sind so vergleichbar.
    mode='shared' (v1): gemeinsame Softmax über normierte Residuen
    verschiedener Eltern – mathematisch inkonsistent, nur zum Vergleich."""
    if mode == "shared":
        return _rham_retrieve_shared(R, Q, beta, beam)
    nq = Q.shape[0]
    out_idx = np.empty(nq, dtype=int); out_w = np.empty(nq); cost = np.zeros(nq)
    child_lists = []
    for i in range(1, R.depth):
        par = R.levels[i - 1].parent
        child_lists.append([np.where(par == j)[0] for j in range(R.levels[i].vecs.shape[0])])
    for qi in range(nq):
        q = Q[qi]
        cand = np.arange(R.levels[-1].vecs.shape[0])
        par = np.full(len(cand), -1)
        logp_par = np.zeros(len(cand))
        for lvl in range(R.depth - 1, -1, -1):
            L = R.levels[lvl]
            qr = np.tile(q, (len(cand), 1)) if lvl == R.depth - 1 else normalize(q[None, :] - R.levels[lvl + 1].vecs[par])
            keys = L.res_keys[cand]
            s = beta * np.einsum("ij,ij->i", qr, keys)
            cost[qi] += len(cand)
            if lvl > 0:
                logp = logp_par + _group_log_softmax(s, par)
                order = np.argsort(-logp)[:beam]
                chosen, chosen_lp = cand[order], logp[order]
                cand = np.concatenate([child_lists[lvl - 1][c] for c in chosen])
                logp_par = np.concatenate([np.full(len(child_lists[lvl - 1][c]), lp) for c, lp in zip(chosen, chosen_lp)])
                par = np.concatenate([np.full(len(child_lists[lvl - 1][c]), c) for c in chosen])
            elif mode == "exact":
                # v3: Endschritt mit exakter Zerlegung q.x = q.p + q.(x-p), d. h.
                # Hopfield-Schritt auf den Rohvektoren der Kandidaten
                bf = beta if beta_final is None else beta_final
                X = L.vecs[cand]
                A = softmax_rows(bf * (q @ X.T)[None])
                xi = normalize(A @ X)
                A2 = softmax_rows(bf * xi @ X.T)[0]
                j = A2.argmax()
                out_idx[qi] = cand[j]; out_w[qi] = A2[j]
            else:
                # Hopfield-Schritt je Gruppe im eigenen Residuenraum, dann Re-Attention
                prob = np.empty(len(cand))
                for g in np.unique(par):
                    m = par == g
                    w = np.exp(s[m] - s[m].max()); w /= w.sum()
                    xi = normalize((w[:, None] * keys[m]).sum(0, keepdims=True))[0]
                    s2 = beta * keys[m] @ xi
                    w2 = np.exp(s2 - s2.max()); w2 /= w2.sum()
                    prob[m] = np.exp(logp_par[m]) * w2
                prob /= prob.sum()
                j = prob.argmax()
                out_idx[qi] = cand[j]; out_w[qi] = prob[j]
    return out_idx, out_w, cost


def _rham_retrieve_shared(R: RHAM, Q, beta, beam=2):
    """Top-down: auf jeder Ebene Hopfield-Schritt im eigenen Residuenraum,
    Beam der besten Kandidaten absteigen. Rückgabe: idx, maxweight, Kosten."""
    nq = Q.shape[0]
    out_idx = np.empty(nq, dtype=int)
    out_w = np.empty(nq)
    cost = np.zeros(nq)
    top = R.levels[-1]
    # Kinderlisten vorberechnen
    child_lists = []
    for i in range(1, R.depth):
        par = R.levels[i - 1].parent
        child_lists.append([np.where(par == j)[0] for j in range(R.levels[i].vecs.shape[0])])
    for qi in range(nq):
        q = Q[qi]
        # oberste Ebene: alle Kandidaten, kein Eltern-Prototyp
        cand = np.arange(top.vecs.shape[0])
        parents_of_cand = np.full(len(cand), -1)
        for lvl in range(R.depth - 1, -1, -1):
            L = R.levels[lvl]
            if lvl == R.depth - 1:
                qr = np.tile(q, (len(cand), 1))
            else:
                P = R.levels[lvl + 1].vecs[parents_of_cand]
                qr = normalize(q[None, :] - P)
            keys = L.res_keys[cand]
            s = beta * np.einsum("ij,ij->i", qr, keys)
            cost[qi] += len(cand)
            # Softmax getrennt je Eltern-Gruppe wäre exakter; hier gemeinsam,
            # da Residuen je Gruppe vergleichbar normiert sind.
            w = np.exp(s - s.max()); w /= w.sum()
            if lvl == 0:
                # finaler Hopfield-Schritt im Residuenraum und Re-Attention
                xi = normalize((w[:, None] * keys).sum(0, keepdims=True))[0]
                s2 = beta * keys @ xi
                w2 = np.exp(s2 - s2.max()); w2 /= w2.sum()
                j = w2.argmax()
                out_idx[qi] = cand[j]
                out_w[qi] = w2[j]
            else:
                order = np.argsort(-w)[:beam]
                chosen = cand[order]
                new_cand, new_par = [], []
                for c in chosen:
                    ch = child_lists[lvl - 1][c]
                    new_cand.append(ch)
                    new_par.append(np.full(len(ch), c))
                cand = np.concatenate(new_cand)
                parents_of_cand = np.concatenate(new_par)
    return out_idx, out_w, cost


# ----------------------------------------------------------------------------
# Interferenz und automatisches Wachstum
# ----------------------------------------------------------------------------

def interference(X, beta, eps=0.05):
    """Anteil Muster mit Ramsauer-Fehlerschranke 2(N-1)exp(-beta*Delta) > eps."""
    N = X.shape[0]
    if N < 2:
        return 0.0
    G = X @ X.T
    np.fill_diagonal(G, -np.inf)
    delta = 1.0 - G.max(1)
    bound = 2 * (N - 1) * np.exp(-beta * delta)
    return float((bound > eps).mean())


def gap_select_k(X, seed, k_grid, n_null=2):
    """Gap-Statistik (Tibshirani, Walther & Hastie 2001), Nullmodell:
    gleichverteilte Punkte auf der Sphäre gleicher Dimension."""
    rng = np.random.default_rng(seed + 991)
    gaps = []
    for k in k_grid:
        _, W = kmeans(X, k, seed)
        Wn = []
        for b in range(n_null):
            Z = normalize(rng.standard_normal(X.shape))
            _, wb = kmeans(Z, k, seed + b)
            Wn.append(np.log(wb))
        gaps.append(np.mean(Wn) - np.log(W))
    gaps = np.array(gaps)
    return k_grid[int(gaps.argmax())], float(gaps.max()), gaps


def grow_hierarchy(X, beta, seed, theta=0.2, gap_margin=0.15, max_levels=8):
    """Wachstumsregel: neue Ebene, wenn I(M_n) > theta UND die Gap-Statistik
    echte Clusterstruktur zeigt (MDL-Ersatz). Sonst Stopp."""
    levels = [Level(vecs=X, parent=None)]
    log = []
    cur = X
    while len(levels) < max_levels:
        n = cur.shape[0]
        I = interference(cur, beta)
        entry = {"level": len(levels) - 1, "n": int(n), "interference": round(I, 3)}
        if I <= theta or n < 6:
            entry["decision"] = "stop: Interferenz gering" if I <= theta else "stop: zu wenige Einheiten"
            log.append(entry); break
        k_grid = sorted({int(round(v)) for v in np.geomspace(2, max(3, n // 3), 14)})
        k, g, _ = gap_select_k(cur, seed, k_grid)
        entry.update({"k_best": int(k), "gap": round(g, 3)})
        if g < gap_margin:
            entry["decision"] = "stop: keine Struktur (Gap zu klein)"
            log.append(entry); break
        lab, _ = kmeans(cur, k, seed)
        protos = normalize(np.stack([cur[lab == c].mean(0) for c in range(k)]))
        levels[-1].parent = lab
        levels.append(Level(vecs=protos, parent=None))
        entry["decision"] = f"neue Ebene M{len(levels)-1} mit {k} Prototypen"
        log.append(entry)
        cur = protos
    _attach_residuals(levels)
    return RHAM(levels), log


# ----------------------------------------------------------------------------
# Experimente
# ----------------------------------------------------------------------------

def nn_accuracy(X, Q, target):
    return float(((Q @ X.T).argmax(1) == target).mean())


def success(idx, w, target, wmin=0.9):
    return float(((idx == target) & (w > wmin)).mean())


def exp_E1(seeds, d=64, h=3, Ks=(4, 6, 8, 10, 13, 16), eta=0.35, betas=(8, 16, 32, 64, 128, 256), nq=300):
    rows = []
    for K in Ks:
        for s in seeds:
            rng = np.random.default_rng(s)
            X, lab = make_tree(K, h, d, rng)
            N = X.shape[0]
            tgt = rng.integers(0, N, nq)
            Q = corrupt(X[tgt], eta, rng)
            R = build_hierarchy_fixed(X, [K ** (h - 1), K ** (h - 2)][: h - 1], seed=s)
            row = {"K": K, "N": N, "seed": s, "nn_ceiling": nn_accuracy(X, Q, tgt)}
            for b in betas:
                fi, fw = flat_retrieve(X, Q, b)
                ri, rw, cost = rham_retrieve(R, Q, b, beam=2)
                v3i, v3w, _ = rham_retrieve(R, Q, b, beam=2, mode="exact")
                row[f"rham3_b{b}"] = success(v3i, v3w, tgt); row[f"rham3_b{b}_top1"] = float((v3i == tgt).mean())
                row[f"flat_b{b}"] = success(fi, fw, tgt); row[f"flat_b{b}_top1"] = float((fi == tgt).mean())
                row[f"rham_b{b}"] = success(ri, rw, tgt); row[f"rham_b{b}_top1"] = float((ri == tgt).mean())
                row[f"rham_cost_b{b}"] = float(cost.mean())
            row["flat_cost"] = N
            rows.append(row)
            print(f"E1 K={K:2d} N={N:5d} seed={s} nn={row['nn_ceiling']:.2f} "
                  f"flat16={row['flat_b16']:.2f} rham16={row['rham_b16']:.2f} "
                  f"flatbest={max(row[f'flat_b{b}'] for b in betas):.2f} "
                  f"rhambest={max(row[f'rham_b{b}'] for b in betas):.2f}", flush=True)
    return rows


def exp_E1b(seeds, d=64, h=3, K=10, etas=(0.35, 0.7, 1.0, 1.3, 1.6, 2.0), betas=(8, 16, 32, 64, 128, 256), nq=300):
    """Rausch-Sweep bei N=1000: wo liegt die Grenze, und hält RHAM die
    Nächster-Nachbar-Obergrenze?"""
    rows = []
    for eta in etas:
        for s in seeds:
            rng = np.random.default_rng(s)
            X, _ = make_tree(K, h, d, rng)
            tgt = rng.integers(0, X.shape[0], nq)
            Q = corrupt(X[tgt], eta, rng)
            R = build_hierarchy_fixed(X, [K ** (h - 1), K ** (h - 2)], seed=s)
            row = {"eta": eta, "seed": s, "nn_ceiling": nn_accuracy(X, Q, tgt)}
            for b in betas:
                fi, fw = flat_retrieve(X, Q, b)
                ri, rw, _ = rham_retrieve(R, Q, b, beam=2)
                v3i, v3w, _ = rham_retrieve(R, Q, b, beam=2, mode="exact")
                row[f"rham3_b{b}"] = success(v3i, v3w, tgt); row[f"rham3_b{b}_top1"] = float((v3i == tgt).mean())
                row[f"flat_b{b}"] = success(fi, fw, tgt); row[f"flat_b{b}_top1"] = float((fi == tgt).mean())
                row[f"rham_b{b}"] = success(ri, rw, tgt); row[f"rham_b{b}_top1"] = float((ri == tgt).mean())
            rows.append(row)
            print(f"E1b eta={eta} seed={s} nn={row['nn_ceiling']:.2f} flat16={row['flat_b16']:.2f} "
                  f"rham16={row['rham_b16']:.2f} flatbest={max(row[f'flat_b{b}'] for b in betas):.2f} "
                  f"rhambest={max(row[f'rham_b{b}'] for b in betas):.2f}", flush=True)
    return rows


def exp_E1c(seeds, d=64, h=3, K=10, etas=(1.0, 1.6, 2.0), beams=(1, 2, 4, 8), beta=64, nq=300):
    """Beam-Breite vs. Fehlrouting: Genauigkeit und Kosten."""
    rows = []
    for eta in etas:
        for s in seeds:
            rng = np.random.default_rng(s)
            X, _ = make_tree(K, h, d, rng)
            tgt = rng.integers(0, X.shape[0], nq)
            Q = corrupt(X[tgt], eta, rng)
            R = build_hierarchy_fixed(X, [K ** (h - 1), K ** (h - 2)], seed=s)
            row = {"eta": eta, "seed": s, "nn_ceiling": nn_accuracy(X, Q, tgt)}
            for bm in beams:
                ri, rw, cost = rham_retrieve(R, Q, beta, beam=bm)
                row[f"rham_beam{bm}"] = success(ri, rw, tgt); row[f"rham_beam{bm}_top1"] = float((ri == tgt).mean())
                row[f"cost_beam{bm}"] = float(cost.mean())
                vi, vw, _ = rham_retrieve(R, Q, beta, beam=bm, mode="shared")
                row[f"v1shared_beam{bm}"] = success(vi, vw, tgt); row[f"v1shared_beam{bm}_top1"] = float((vi == tgt).mean())
            rows.append(row)
            print("E1c", {k: round(v, 2) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
    return rows


def exp_E3(seeds, d=64, configs=((8, 2), (6, 3), (4, 4), (3, 5)), beta=16):
    rows = []
    for K, h in configs:
        for s in seeds:
            rng = np.random.default_rng(s)
            X, _ = make_tree(K, h, d, rng)
            R, log = grow_hierarchy(X, beta, seed=s)
            rows.append({"data": f"Baum K={K}, h={h}", "N": X.shape[0], "true_levels": h,
                         "found_levels": R.depth, "seed": s, "log": log})
            print(f"E3 K={K} h={h} N={X.shape[0]} -> Ebenen {R.depth} (Soll {h})", flush=True)
    # Kontrolle P3: i.i.d.-Daten ohne Struktur
    for N in (256, 1024):
        for s in seeds:
            rng = np.random.default_rng(s)
            X = normalize(rng.standard_normal((N, d)))
            R, log = grow_hierarchy(X, beta, seed=s)
            rows.append({"data": f"i.i.d. Rauschen", "N": N, "true_levels": 1,
                         "found_levels": R.depth, "seed": s, "log": log})
            print(f"E3 iid N={N} -> Ebenen {R.depth} (Soll 1)", flush=True)
        # i.i.d. in niedriger Dimension: Interferenz hoch, aber keine Struktur
        for s in seeds:
            rng = np.random.default_rng(s)
            X = normalize(rng.standard_normal((N, 8)))
            R, log = grow_hierarchy(X, beta, seed=s)
            rows.append({"data": "i.i.d. Rauschen d=8", "N": N, "true_levels": 1,
                         "found_levels": R.depth, "seed": s, "log": log})
            print(f"E3 iid d=8 N={N} -> Ebenen {R.depth} (Soll 1)", flush=True)
    return rows


def exp_E6(seeds, d=64, beta=16,
           thetas=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7),
           gaps=(0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5)):
    """Sensitivität der Wachstumsregel. Beide Schwellen entscheiden nur über den
    Stopp; der Pfad bis dahin ist deterministisch. Daher: vollständigen Pfad
    einmal berechnen (Schwellen aus), dann Tiefe für jede Kombination ableiten."""
    datasets = [("Baum K=8,h=2", lambda r: make_tree(8, 2, d, r)[0], 2),
                ("Baum K=6,h=3", lambda r: make_tree(6, 3, d, r)[0], 3),
                ("Baum K=4,h=4", lambda r: make_tree(4, 4, d, r)[0], 4),
                ("Baum K=3,h=5", lambda r: make_tree(3, 5, d, r)[0], 5),
                ("iid d=64 N=256", lambda r: normalize(r.standard_normal((256, d))), 1),
                ("iid d=64 N=1024", lambda r: normalize(r.standard_normal((1024, d))), 1),
                ("iid d=8 N=1024", lambda r: normalize(r.standard_normal((1024, 8))), 1)]
    paths = []
    for name, gen, true_h in datasets:
        for s in seeds:
            X = gen(np.random.default_rng(s))
            _, log = grow_hierarchy(X, beta, seed=s, theta=-1.0, gap_margin=-1e9)
            path = [(e["interference"], e.get("gap", -np.inf)) for e in log if "gap" in e]
            paths.append({"data": name, "true_h": true_h, "seed": s, "path": path})
            print(f"E6 {name} seed={s}: Pfad {[(round(i,2), round(g,2)) for i, g in path]}", flush=True)
    grid = []
    for th in thetas:
        for gm in gaps:
            res = {}
            for p in paths:
                depth = 1
                for I, g in p["path"]:
                    if I > th and g >= gm:
                        depth += 1
                    else:
                        break
                res.setdefault(p["data"], []).append((depth, p["true_h"]))
            tree_err = np.mean([abs(dp - th_) for k, v in res.items() if k.startswith("Baum") for dp, th_ in v])
            tree_exact = np.mean([dp == th_ for k, v in res.items() if k.startswith("Baum") for dp, th_ in v])
            iid_false = np.mean([dp > 1 for k, v in res.items() if k.startswith("iid") for dp, _ in v])
            grid.append({"theta": th, "gap": gm, "tree_abs_err": float(tree_err),
                         "tree_exact": float(tree_exact), "iid_false_growth": float(iid_false),
                         "per_data": {k: [dp for dp, _ in v] for k, v in res.items()}})
    return {"paths": paths, "grid": grid}


def exp_E4(seeds, d=64, K=8, h=3, fracs=(0.0, 0.25, 0.5, 0.75)):
    """Vergessen in M0: vergessene Blätter werden beim Abruf durch ihren
    Eltern-Prototyp ersetzt ('Gist'). Gemessen: mittlere Kosinus-Treue zum
    wahren Blatt über alle Blätter."""
    rows = []
    for s in seeds:
        rng = np.random.default_rng(s)
        X, _ = make_tree(K, h, d, rng, leaf_jitter=True)
        R = build_hierarchy_fixed(X, [K ** (h - 1), K ** (h - 2)], seed=s)
        P = R.levels[1].vecs[R.levels[0].parent]
        recon = np.einsum("ij,ij->i", X, P)        # Kosinus Blatt vs. Prototyp
        for f in fracs:
            m = int(f * len(X))
            by_rule = np.argsort(-recon)[:m]        # am besten rekonstruierbare zuerst
            by_rand = rng.permutation(len(X))[:m]
            for name, forg in (("Rekonstruktionsregel", by_rule), ("Zufall", by_rand)):
                fid = np.ones(len(X))
                fid[forg] = recon[forg]
                rows.append({"seed": s, "frac": f, "policy": name, "mean_fidelity": float(fid.mean()),
                             "min_fidelity": float(fid.min()), "storage_M0": 1 - f})
        print(f"E4 seed={s} fertig", flush=True)
    return rows


if __name__ == "__main__":
    seeds = [0, 1, 2, 3, 4]
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    out = {}
    t0 = time.time()
    if which in ("all", "E1"):
        out["E1"] = exp_E1(seeds)
    if which in ("all", "E1b"):
        out["E1b"] = exp_E1b(seeds)
    if which in ("all", "E1c"):
        out["E1c"] = exp_E1c(seeds)
    if which in ("all", "E3"):
        out["E3"] = exp_E3(seeds)
    if which in ("all", "E4"):
        out["E4"] = exp_E4(seeds)
    if which == "E6":
        out["E6"] = exp_E6(seeds)
    out["runtime_s"] = round(time.time() - t0, 1)
    with open(f"results_{which}.json", "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print("fertig", out["runtime_s"], "s")
