"""Abbildung E8 (Regel 7, Anisotropie). Palette validiert (4 Farben, light);
Aqua/Gelb < 3:1 Kontrast -> Direktbeschriftung + Markerformen."""
import json, glob, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, YELLOW, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8984"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#d6d5d0", "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "lines.linewidth": 2})


def lab(ax, x, y, text, dy=0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", color=INK, va="center", fontsize=9)


def load(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        d = json.load(open(f)); rows += [r for k, v in d.items() if k != "runtime_s" for r in v]
    return rows


D = load("results_E8_drift_*.json")
A = load("results_E8_aniso_*.json") + load("results_E8_gmeans_*.json")

fig, axs = plt.subplots(2, 2, figsize=(11, 8.4))
VC = {"Basis (1/n)": (BLUE, "o"), "gedeckelt": (ORANGE, "s"), "Regel 7 komplett": (AQUA, "^")}

def drift_series(dl, v, f):
    rr = [r for r in D if r["delta"] == dl and r["variant"] == v]
    t = np.array([x["t"] for x in rr[0]["traj"]]) / 1000
    return t, np.mean([[x[f] for x in r["traj"]] for r in rr], 0)

ax = axs[0, 0]
for v, (c, mk) in VC.items():
    t, y = drift_series(0.3, v, "assoc"); ax.plot(t, y, color=c, marker=mk, ms=4)
    lab(ax, t[-1], y[-1], v, {"Basis (1/n)": 6, "gedeckelt": -6, "Regel 7 komplett": 0}[v])
t, y = drift_series(0.0, "Basis (1/n)", "assoc"); ax.plot(t, y, color=GRAY, ls="--", lw=1.5); lab(ax, t[-1], y[-1], "ohne Drift")
ax.set_xlim(0, 28); ax.set_ylim(0, 1300)
ax.set_xlabel("Episoden (Tausend), Driftschritt 0,3"); ax.set_ylabel("assoziativ (heiß) gespeicherte Vektoren")
ax.set_title("A  Drift: Speicher", loc="left", fontweight="bold"); ax.grid(axis="y", color="#ecebe7", lw=0.8)

ax = axs[0, 1]
for v, (c, mk) in VC.items():
    t, y = drift_series(0.3, v, "acc_head"); ax.plot(t, y, color=c, marker=mk, ms=4)
    lab(ax, t[-1], y[-1], v, {"Basis (1/n)": -4, "gedeckelt": 0, "Regel 7 komplett": 6}[v])
ax.set_xlim(0, 28); ax.set_ylim(0.75, 1.01)
ax.set_xlabel("Episoden (Tausend), Driftschritt 0,3"); ax.set_ylabel("Top-1, häufige Konzepte")
ax.set_title("B  Drift: Genauigkeit", loc="left", fontweight="bold"); ax.grid(axis="y", color="#ecebe7", lw=0.8)

AV = {"ohne Teilung": (BLUE, "o"), "Teilung, iso-Null": (ORANGE, "s"),
      "Teilung, cov-Null": (AQUA, "^"), "Teilung, G-means": (YELLOW, "D")}

def aniso(scale, v, f):
    g = collections.defaultdict(list)
    for r in A:
        if r["scale"] == scale and r["variant"] == v:
            g[r["alpha"]].append(r[f])
    xs = sorted(g); return np.array(xs), np.array([np.mean(g[x]) for x in xs])

ax = axs[1, 0]
dy = {"ohne Teilung": -8, "Teilung, iso-Null": 0, "Teilung, cov-Null": 0, "Teilung, G-means": 8}
ypos = {"Teilung, iso-Null": None, "Teilung, cov-Null": 268, "Teilung, G-means": 243, "ohne Teilung": 218}
for v, (c, mk) in AV.items():
    x, y = aniso(1.0, v, "n_proto")
    if len(x):
        ax.plot(x, y, color=c, marker=mk, ms=6)
        yy = y[-1] if ypos[v] is None else ypos[v]
        ax.annotate(v, (x[-1], y[-1]), xytext=(x[-1] + 0.35, yy), textcoords="data", color=INK, va="center", fontsize=9)
x, y = aniso(1.0, "ohne Teilung", "covered"); ax.plot(x, y, color=GRAY, ls="--", lw=1.5)
ax.text(x[-1] + 0.35, 192, "wahre Konzepte (erkannt)", color=INK2, va="center", fontsize=9)
ax.set_xlim(0.5, 12); ax.set_ylim(0, 430)
ax.set_xlabel("Streckung α des Rauschens (1 = isotrop)"); ax.set_ylabel("aktive Prototypen")
ax.set_title("C  Getrennte Konzepte: Über-Teilung?", loc="left", fontweight="bold"); ax.grid(axis="y", color="#ecebe7", lw=0.8)

ax = axs[1, 1]
dy = {"ohne Teilung": -9, "Teilung, iso-Null": 4, "Teilung, cov-Null": 7, "Teilung, G-means": -4}
for v, (c, mk) in AV.items():
    x, y = aniso(0.6, v, "acc")
    if len(x):
        ax.plot(x, y, color=c, marker=mk, ms=6); lab(ax, x[-1], y[-1], v, dy[v])
ax.set_xlim(0.5, 12); ax.set_ylim(0.9, 1.005)
ax.set_xlabel("Streckung α des Rauschens (1 = isotrop)"); ax.set_ylabel("Konzept-Top-1")
ax.set_title("D  Überlappende Konzepte: Genauigkeit", loc="left", fontweight="bold"); ax.grid(axis="y", color="#ecebe7", lw=0.8)

fig.suptitle("RHAM Runde 4 · Regel 7 unter Drift (A, B) und Teilungstest unter anisotropem Rauschen (C, D), 3 Seeds",
             x=0.01, ha="left", color=INK, fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig("rham_runde4_ergebnisse.png", dpi=150)
print("ok")
