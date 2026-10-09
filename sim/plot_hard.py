"""Abbildung E7 (Härtetest). Palette wie plot_results.py (validiert)."""
import json, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#d6d5d0", "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "lines.linewidth": 2})


def lab(ax, x, y, text, dy=0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", color=INK, va="center", fontsize=9)


def by(rows, key, f):
    g = collections.defaultdict(list)
    for r in rows:
        g[r[key]].append(r[f])
    xs = sorted(g)
    return np.array(xs), np.array([np.mean(g[x]) for x in xs])


S1 = json.load(open("results_E7_S1.json"))["S1"]
S1s = json.load(open("results_E7_S1split.json"))["S1split"]
S2 = json.load(open("results_E7_S2.json"))["S2"]
S3 = json.load(open("results_E7_S3.json"))["S3"]

fig, axs = plt.subplots(2, 2, figsize=(11, 8.2))

# A Genauigkeit vs. Überlappung (x = Kosinus zwischen Geschwisterkonzepten)
ax = axs[0, 0]
x, y = by(S1, "scale", "sibling_cos"); order = np.argsort(y)
cos = y[order]
for rows, f, c, mk, text, dy in ((S1, "flat", BLUE, "o", "flach (alle Episoden)", 8),
                                 (S1s, "rham", AQUA, "^", "RHAM + Teilungsregel", -2),
                                 (S1, "rham", ORANGE, "s", "RHAM ohne Teilung", 0)):
    _, v = by(rows, "scale", f); v = v[order]
    ax.plot(cos, v, color=c, marker=mk, ms=6); lab(ax, cos[-1], v[-1], text, dy)
ax.axvline(0.86, color=GRAY, ls=":", lw=1)
ax.text(0.862, 0.705, "≈ Ähnlichkeit zweier Episoden\ndesselben Konzepts", color=INK2, fontsize=8, va="bottom")
ax.set_ylim(0.7, 1.01); ax.set_xlim(0.64, 1.02)
ax.set_xlabel("Kosinus zwischen Geschwisterkonzepten (Überlappung)"); ax.set_ylabel("Konzept-Top-1")
ax.set_title("A  Überlappende Konzepte", loc="left", fontweight="bold"); ax.grid(axis="y", color="#ecebe7", lw=0.8)

# B gefundene Konzepte
ax = axs[0, 1]
for rows, c, mk, text in ((S1s, AQUA, "^", "RHAM + Teilungsregel"), (S1, ORANGE, "s", "RHAM ohne Teilung")):
    _, v = by(rows, "scale", "concepts_covered"); v = v[order]
    ax.plot(cos, v, color=c, marker=mk, ms=6); lab(ax, cos[-1], v[-1], text)
ax.axhline(216, color=GRAY, ls="--", lw=1.5); ax.text(0.645, 219, "216 wahre Konzepte", color=INK2, fontsize=8)
ax.set_ylim(0, 240); ax.set_xlim(0.64, 1.02)
ax.set_xlabel("Kosinus zwischen Geschwisterkonzepten (Überlappung)"); ax.set_ylabel("unterschiedene Konzepte")
ax.set_title("B  Konzepte getrennt erkannt", loc="left", fontweight="bold"); ax.grid(axis="y", color="#ecebe7", lw=0.8)

# C Drift: Speicher über Zeit
ax = axs[1, 0]
for dl, cap, c, mk, ls, text in ((0.2, None, ORANGE, "s", "-", "Drift, laufender Mittelwert"),
                                 (0.2, 30, AQUA, "^", "-", "Drift, gedeckelte Lernrate"),
                                 (0.0, None, GRAY, None, "--", "ohne Drift")):
    rows = [r for r in S2 if r["delta"] == dl and r["cap"] == cap]
    t = np.array([s["t"] for s in rows[0]["traj"]]) / 1000
    v = np.mean([[s["assoc"] for s in r["traj"]] for r in rows], 0)
    ax.plot(t, v, color=c, marker=mk, ms=4 if mk else 0, ls=ls, lw=2 if mk else 1.5); lab(ax, t[-1], v[-1], text)
ax.set_xlim(0, 21); ax.set_ylim(0, 850)
ax.set_xlabel("Episoden (Tausend), Driftschritt 0,2 pro Schlafphase"); ax.set_ylabel("assoziativ gespeicherte Vektoren")
ax.set_title("C  Konzeptdrift: Speicher", loc="left", fontweight="bold"); ax.grid(axis="y", color="#ecebe7", lw=0.8)

# D Seltene Episoden: exakter Abruf
ax = axs[1, 1]
kinds = ["untypisch", "typisch", "gewöhnlich"]
modes = [("assoziativ", False, "assoc_recall", ORANGE), ("assoziativ + Wichtig-Markierung", True, "assoc_recall", AQUA),
         ("mit Archiv (kalter Zugriff)", False, "deep_recall", BLUE)]
xw = np.arange(len(kinds)); bw = 0.26
for i, (name, prot, f, c) in enumerate(modes):
    vals = [np.mean([r[f] for r in S3 if r["protect"] == prot and r["kind"] == k]) for k in kinds]
    bars = ax.bar(xw + (i - 1) * bw, vals, bw - 0.03, color=c, label=name)
    for b_, v in zip(bars, vals):
        ax.text(b_.get_x() + b_.get_width() / 2, v + 0.015, f"{v:.0%}", ha="center", fontsize=8, color=INK)
ax.set_xticks(xw, ["seltene Episode,\nuntypisch", "seltene Episode,\ntypisch aussehend", "gewöhnliche\nEpisode"])
ax.set_ylim(0, 1.5); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_ylabel("exakter Episodenabruf")
ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=1)
ax.set_title("D  Seltene Einzelepisoden", loc="left", fontweight="bold"); ax.grid(axis="y", color="#ecebe7", lw=0.8)

fig.suptitle("RHAM-Härtetest · 216 Konzepte, Zipf, Schlafphase alle 1000 Episoden, 3 Seeds",
             x=0.01, ha="left", color=INK, fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig("rham_haertetest_ergebnisse.png", dpi=150)
print("ok")
