"""Figure for the RHAM simulation (4 panels). Colors: reference palette, validated
(validate_palette.js, light); Aqua < 3:1 contrast -> direct labeling + marker shapes."""
import json
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
B = (8, 16, 32, 64, 128, 256)


def mean_by(rows, key, field):
    xs = sorted({r[key] for r in rows})
    m = [np.mean([r[field] for r in rows if r[key] == x]) for x in xs]
    sd = [np.std([r[field] for r in rows if r[key] == x]) for x in xs]
    return np.array(xs), np.array(m), np.array(sd)


def label_end(ax, x, y, text, color, dy=0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", color=INK,
                va="center", fontsize=9)


E1 = json.load(open("results_E1.json"))["E1"]
E1b = json.load(open("results_E1b.json"))["E1b"]
E4 = json.load(open("results_E4.json"))["E4"]
for r in E1b:
    r["rham_best_top1"] = max(r[f"rham_b{b}_top1"] for b in B)

fig, axs = plt.subplots(2, 2, figsize=(11, 8.2))

# A: sharp retrieval vs N at beta=16
ax = axs[0, 0]
for f, c, mk, lab, dy in (("flat_b16", BLUE, "o", "flat", 0), ("rham_b16", ORANGE, "s", "RHAM", 0)):
    x, m, sd = mean_by(E1, "N", f)
    ax.errorbar(x, m, yerr=sd, color=c, marker=mk, ms=6, capsize=0, elinewidth=1)
    label_end(ax, x[-1], m[-1], lab, c, dy)
ax.set_xscale("log"); ax.set_ylim(0, 1.05); ax.set_xlim(50, 9000)
ax.set_xlabel("stored patterns N"); ax.set_ylabel("fraction of sharp retrievals")
ax.set_title("A  Sharp retrieval at fixed β = 16", loc="left", fontweight="bold")
ax.grid(axis="y", color="#ecebe7", lw=0.8)

# B: compute cost
ax = axs[0, 1]
x, m, _ = mean_by(E1, "N", "flat_cost"); ax.plot(x, m, color=BLUE, marker="o", ms=6); label_end(ax, x[-1], m[-1], "flat: N", BLUE)
x, m, _ = mean_by(E1, "N", "rham_cost_b16"); ax.plot(x, m, color=ORANGE, marker="s", ms=6); label_end(ax, x[-1], m[-1], "RHAM (beam 2)", ORANGE)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(50, 12000)
ax.set_xlabel("stored patterns N"); ax.set_ylabel("dot products per retrieval")
ax.set_title("B  Compute cost per retrieval", loc="left", fontweight="bold")
ax.grid(axis="y", color="#ecebe7", lw=0.8)

# C: top-1 under noise
ax = axs[1, 0]
x, m, _ = mean_by(E1b, "eta", "nn_ceiling"); ax.plot(x, m, color=GRAY, ls="--", lw=1.5); label_end(ax, x[-1], m[-1], "ceiling (exact NN)", GRAY, 8)
x, m, _ = mean_by(E1b, "eta", "rham_best_top1"); ax.plot(x, m, color=ORANGE, marker="s", ms=6); label_end(ax, x[-1], m[-1], "RHAM v2", ORANGE, -6)
v3x = [0.35, 1.0, 1.6, 2.0]; v3y = [0.997, 0.978, 0.698, 0.453]   # results_v3.txt, beam 4
ax.plot(v3x, v3y, color=AQUA, marker="^", ms=7); label_end(ax, v3x[-1], v3y[-1], "RHAM v3 (beam 4)", AQUA, 4)
ax.set_ylim(0, 1.05); ax.set_xlim(0.2, 2.75)
ax.set_xlabel("query noise η (N = 1000)"); ax.set_ylabel("top-1 hit rate")
ax.set_title("C  Accuracy under noise", loc="left", fontweight="bold")
ax.grid(axis="y", color="#ecebe7", lw=0.8)

# D: forgetting
ax = axs[1, 1]
for pol, c, mk in (("reconstruction_rule", ORANGE, "s"), ("random", BLUE, "o")):
    sub = [r for r in E4 if r["policy"] == pol]
    x, m, _ = mean_by(sub, "frac", "mean_fidelity"); ax.plot(x * 100, m, color=c, marker=mk, ms=6)
    label_end(ax, x[-1] * 100, m[-1], pol, c)
ax.set_ylim(0.8, 1.01); ax.set_xlim(-3, 105)
ax.set_xlabel("forgotten leaves in M0 (%)"); ax.set_ylabel("mean fidelity (cosine to original)")
ax.set_title("D  Selective forgetting", loc="left", fontweight="bold")
ax.grid(axis="y", color="#ecebe7", lw=0.8)

fig.suptitle("RHAM minimal simulation · synthetic tree data (d = 64, h = 3), 5 seeds", x=0.01, ha="left", color=INK, fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig("rham_sim_ergebnisse.png", dpi=150)
print("ok")
