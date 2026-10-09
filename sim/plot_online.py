"""Abbildung E5 (Online-Strom). Palette wie plot_results.py (validiert)."""
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

E = json.load(open("results_E5.json"))["E5"]
seeds = list(E)
t = np.array([e["t"] for e in E[seeds[0]]]) / 1000


def series(k):
    a = np.array([[np.nan if e[k] is None else e[k] for e in E[s]] for s in seeds], float)
    return np.nanmean(a, 0)


def lab(ax, x, y, text, dy=0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", color=INK, va="center", fontsize=9)


def shade(ax, y=0.97, va="top"):
    ax.axvspan(10, 20.5, color="#efeee9", zorder=0)
    ax.text(10.2, y, "neue Äste aktiv", transform=ax.get_xaxis_transform(), color=INK2, fontsize=8, va=va)


fig, axs = plt.subplots(2, 2, figsize=(11, 8.2))

ax = axs[0, 0]; shade(ax)
ax.plot(t, series("flat_store"), color=BLUE, marker="o", ms=4); lab(ax, t[-1], series("flat_store")[-1], "flach (alle Episoden)")
ax.plot(t, series("assoc_store"), color=ORANGE, marker="s", ms=4); lab(ax, t[-1], series("assoc_store")[-1], "RHAM (assoziativ)")
ax.set_yscale("log"); ax.set_xlim(0, 25.5)
ax.set_xlabel("Episoden (Tausend)"); ax.set_ylabel("gespeicherte Vektoren")
ax.set_title("A  Assoziativer Speicher", loc="left", fontweight="bold"); ax.grid(axis="y", color="#ecebe7", lw=0.8)

ax = axs[0, 1]; shade(ax)
ax.plot(t, series("bits_raw"), color=BLUE, marker="o", ms=4); lab(ax, t[-1], series("bits_raw")[-1], "roh", 4)
ax.plot(t, series("bits_rham"), color=ORANGE, marker="s", ms=4); lab(ax, t[-1], series("bits_rham")[-1], "RHAM (Zwei-Teile-Code)", 6)
ax.plot(t, series("h_true"), color=GRAY, ls="--", lw=1.5); lab(ax, t[-1], series("h_true")[-1], "Entropierate der Quelle", -8)
ax.set_ylim(0, 200); ax.set_xlim(0, 25.5)
ax.set_xlabel("Episoden (Tausend)"); ax.set_ylabel("Bits pro neuer Episode")
ax.set_title("B  P4: Grenzkosten pro Episode", loc="left", fontweight="bold"); ax.grid(axis="y", color="#ecebe7", lw=0.8)

ax = axs[1, 0]; shade(ax, 0.03, "bottom")
ax.plot(t, series("acc_old"), color=ORANGE, marker="s", ms=4); lab(ax, t[-1], series("acc_old")[-1], "alte Konzepte", 6)
m = ~np.isnan(series("acc_new"))
ax.plot(t[m], series("acc_new")[m], color=AQUA, marker="^", ms=5); lab(ax, t[-1], series("acc_new")[-1], "neue Konzepte", -6)
ax.set_ylim(0.9, 1.005); ax.set_xlim(0, 25.5)
ax.set_xlabel("Episoden (Tausend)"); ax.set_ylabel("Konzept-Top-1")
ax.set_title("C  Abruf alter und neuer Konzepte", loc="left", fontweight="bold"); ax.grid(axis="y", color="#ecebe7", lw=0.8)

ax = axs[1, 1]; shade(ax)
ax.plot(t, series("flat_store"), color=BLUE, marker="o", ms=4); lab(ax, t[-1], series("flat_store")[-1], "flach: N")
ax.plot(t, series("probe_cost"), color=ORANGE, marker="s", ms=4); lab(ax, t[-1], series("probe_cost")[-1], "RHAM")
ax.set_yscale("log"); ax.set_xlim(0, 25.5)
ax.set_xlabel("Episoden (Tausend)"); ax.set_ylabel("Skalarprodukte pro Abruf")
ax.set_title("D  Rechenaufwand pro Abruf", loc="left", fontweight="bold"); ax.grid(axis="y", color="#ecebe7", lw=0.8)

fig.suptitle("RHAM online · 20 000 Episoden, 216 Konzepte (Zipf), Schlafphase alle 1000, 5 Seeds",
             x=0.01, ha="left", color=INK, fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig("rham_online_ergebnisse.png", dpi=150)
print("ok")
