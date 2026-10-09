"""Figure for E8 (rule 7, anisotropy). Palette validated (4 colors, light);
Aqua/yellow < 3:1 contrast -> direct labeling + marker shapes."""
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
VC = {"baseline (1/n)": (BLUE, "o"), "capped": (ORANGE, "s"), "rule 7 complete": (AQUA, "^")}

def drift_series(dl, v, f):
    rr = [r for r in D if r["delta"] == dl and r["variant"] == v]
    t = np.array([x["t"] for x in rr[0]["traj"]]) / 1000
    return t, np.mean([[x[f] for x in r["traj"]] for r in rr], 0)

ax = axs[0, 0]
for v, (c, mk) in VC.items():
    t, y = drift_series(0.3, v, "assoc"); ax.plot(t, y, color=c, marker=mk, ms=4)
    lab(ax, t[-1], y[-1], v, {"baseline (1/n)": 6, "capped": -6, "rule 7 complete": 0}[v])
t, y = drift_series(0.0, "baseline (1/n)", "assoc"); ax.plot(t, y, color=GRAY, ls="--", lw=1.5); lab(ax, t[-1], y[-1], "no drift")
ax.set_xlim(0, 28); ax.set_ylim(0, 1300)
ax.set_xlabel("episodes (thousands), drift step 0.3"); ax.set_ylabel("associatively (hot) stored vectors")
ax.set_title("A  Drift: memory size", loc="left", fontweight="bold"); ax.grid(axis="y", color="#ecebe7", lw=0.8)

ax = axs[0, 1]
for v, (c, mk) in VC.items():
    t, y = drift_series(0.3, v, "acc_head"); ax.plot(t, y, color=c, marker=mk, ms=4)
    lab(ax, t[-1], y[-1], v, {"baseline (1/n)": -4, "capped": 0, "rule 7 complete": 6}[v])
ax.set_xlim(0, 28); ax.set_ylim(0.75, 1.01)
ax.set_xlabel("episodes (thousands), drift step 0.3"); ax.set_ylabel("top-1, frequent concepts")
ax.set_title("B  Drift: accuracy", loc="left", fontweight="bold"); ax.grid(axis="y", color="#ecebe7", lw=0.8)

AV = {"no split": (BLUE, "o"), "split, iso-null": (ORANGE, "s"),
      "split, cov-null": (AQUA, "^"), "split, G-means": (YELLOW, "D")}

def aniso(scale, v, f):
    g = collections.defaultdict(list)
    for r in A:
        if r["scale"] == scale and r["variant"] == v:
            g[r["alpha"]].append(r[f])
    xs = sorted(g); return np.array(xs), np.array([np.mean(g[x]) for x in xs])

ax = axs[1, 0]
dy = {"no split": -8, "split, iso-null": 0, "split, cov-null": 0, "split, G-means": 8}
ypos = {"split, iso-null": None, "split, cov-null": 268, "split, G-means": 243, "no split": 218}
for v, (c, mk) in AV.items():
    x, y = aniso(1.0, v, "n_proto")
    if len(x):
        ax.plot(x, y, color=c, marker=mk, ms=6)
        yy = y[-1] if ypos[v] is None else ypos[v]
        ax.annotate(v, (x[-1], y[-1]), xytext=(x[-1] + 0.35, yy), textcoords="data", color=INK, va="center", fontsize=9)
x, y = aniso(1.0, "no split", "covered"); ax.plot(x, y, color=GRAY, ls="--", lw=1.5)
ax.text(x[-1] + 0.35, 192, "true concepts (recognized)", color=INK2, va="center", fontsize=9)
ax.set_xlim(0.5, 12); ax.set_ylim(0, 430)
ax.set_xlabel("noise stretch α (1 = isotropic)"); ax.set_ylabel("active prototypes")
ax.set_title("C  Separated concepts: over-splitting?", loc="left", fontweight="bold"); ax.grid(axis="y", color="#ecebe7", lw=0.8)

ax = axs[1, 1]
dy = {"no split": -9, "split, iso-null": 4, "split, cov-null": 7, "split, G-means": -4}
for v, (c, mk) in AV.items():
    x, y = aniso(0.6, v, "acc")
    if len(x):
        ax.plot(x, y, color=c, marker=mk, ms=6); lab(ax, x[-1], y[-1], v, dy[v])
ax.set_xlim(0.5, 12); ax.set_ylim(0.9, 1.005)
ax.set_xlabel("noise stretch α (1 = isotropic)"); ax.set_ylabel("concept top-1")
ax.set_title("D  Overlapping concepts: accuracy", loc="left", fontweight="bold"); ax.grid(axis="y", color="#ecebe7", lw=0.8)

fig.suptitle("RHAM round 4 · rule 7 under drift (A, B) and split test under anisotropic noise (C, D), 3 seeds",
             x=0.01, ha="left", color=INK, fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig("rham_runde4_ergebnisse.png", dpi=150)
print("ok")
