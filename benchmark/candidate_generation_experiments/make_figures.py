"""Generate the dissertation figures from the committed result CSVs.

Four figures, each carrying something the tables cannot:

  fig1_bottleneck_migration  the stage priority inverting when Stage 2 improves
  fig2_two_tails             the tail splitting into a learnable and an
                             unlearnable half, with bootstrap intervals
  fig3_triage                errors captured against review budget
  fig4_dose_response         the blind spot as a function of training mismatch

Design constraints, in order of priority, because these are printed in a thesis
and may be read in greyscale:

  * categorical hues are assigned in fixed slot order and never cycled;
    the 5-slot palette was validated (adjacent-pair CVD dE 9.1, normal-vision
    19.6) against a white surface
  * three of the slots fall below 3:1 contrast on white, so the relief rule
    applies and every mark carries a visible direct label
  * every series also carries a non-colour channel, hatch on bars, marker
    shape and dash pattern on lines, so nothing depends on hue alone
  * one axis per figure, recessive grid and spines, no chartjunk

    python make_figures.py
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "thesis_drafts" / "figures"
SRC = HERE / "stage2_out"

# validated 5-slot categorical palette (light surface #ffffff)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
HATCH = ["", "///", "...", "\\\\\\", "xxx"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK, "text.color": INK,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def tidy(ax, ygrid=True):
    ax.spines[["top", "right"]].set_visible(False)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)


def parse_ci(s):
    """'0.4810 [0.4356, 0.5295]' -> (point, lo, hi)"""
    m = re.match(r"([\d.]+)\s*\[([\d.]+),\s*([\d.]+)\]", str(s))
    return tuple(float(x) for x in m.groups()) if m else (float(s), np.nan, np.nan)


# ---------------------------------------------------------------- figure 1
def fig_bottleneck():
    df = pd.read_csv(SRC / "oracle_substitution_lono_vanilla.csv")
    df = df[df.scope == "all"]
    stages = ["detection", "retrieval", "ranking"]
    cols = ["d_detection", "d_retrieval", "d_ranking"]
    rankers = ["gated", "supervised, LONO"]
    labels = ["Gated heuristic", "Leave-one-note-out reranker"]

    x = np.arange(len(stages)); w = 0.36
    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    for i, (r, lab) in enumerate(zip(rankers, labels)):
        row = df[df.ranker == r].iloc[0]
        vals = [row[c] for c in cols]
        pos = x + (i - 0.5) * (w + 0.02)
        bars = ax.bar(pos, vals, w, label=lab, color=C[i], hatch=HATCH[i],
                      edgecolor="white", linewidth=1.2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([s.capitalize() for s in stages])
    ax.set_ylabel("Gain in end-to-end recall\nif this stage were perfect")
    ax.set_ylim(0, 0.245)
    tidy(ax)
    ax.legend(frameon=False, loc="upper center", ncol=1, fontsize=8)
    fig.savefig(OUT / "fig1_bottleneck_migration.png")
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def fig_two_tails():
    df = pd.read_csv(SRC / "stage2_bootstrap_estimates.csv")
    order = ["rrf_rank1", "gated", "supervised, standard",
             "supervised, hygiene", "supervised, LONO"]
    nice = ["No ranking\n(pool order)", "Gated heuristic", "Supervised,\nstandard",
            "Supervised,\nfeatures deleted", "Supervised,\nleave-one-note-out"]
    df = df.set_index("ranker").loc[order].reset_index()

    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    y = np.arange(len(order))[::-1]
    series = [("tail_in_index", "Rare, concept seen 1-2x in training", C[0], "o", 0.16),
              ("tail_out_of_index", "Rare, concept never seen", C[1], "s", -0.16)]
    for col, lab, colr, mk, off in series:
        pts = [parse_ci(v) for v in df[col]]
        p = [a for a, _, _ in pts]
        lo = [a - b for a, b, _ in pts]
        hi = [c - a for a, _, c in pts]
        ax.errorbar(p, y + off, xerr=[lo, hi], fmt=mk, color=colr, label=lab,
                    markersize=6, capsize=3, linewidth=1.6, linestyle="none",
                    markeredgecolor="white", markeredgewidth=0.8)
        for xv, yv in zip(p, y + off):
            ax.text(xv, yv + 0.20, f"{xv:.3f}", ha="center", va="bottom",
                    fontsize=7.5, color=INK)
    for v, colr, lab in [(0.9475, C[0], None), (0.7761, C[1], None)]:
        ax.axvline(v, color=colr, linestyle=":", linewidth=1.2, alpha=0.8)
    ax.text(0.9475, len(order) - 0.35, "ceiling", ha="center", fontsize=7, color=C[0])
    ax.text(0.7761, len(order) - 0.35, "ceiling", ha="center", fontsize=7, color=C[1])
    ax.set_yticks(y); ax.set_yticklabels(nice, fontsize=8)
    ax.set_xlabel("Top-1 accuracy (95% note-clustered bootstrap interval)")
    ax.set_xlim(0, 1.0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0.0, 0.0))
    fig.savefig(OUT / "fig2_two_tails.png")
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
def fig_triage():
    df = pd.read_csv(SRC / "stage2_abstention_budget.csv")
    fig, ax = plt.subplots(figsize=(5.0, 3.1))
    spec = [("supervised, LONO", "margin", "Leave-one-note-out reranker", C[0], "o", "-"),
            ("gated", "maxprob", "Gated heuristic", C[1], "s", "--"),
            ("supervised, LONO", "oracle (ceiling)", "Perfect confidence (ceiling)",
             MUTED, "^", ":")]
    for rk, sig, lab, colr, mk, ls in spec:
        s = df[(df.ranker == rk) & (df.signal == sig)].sort_values("review_budget")
        ax.plot(s.review_budget * 100, s.errors_captured * 100, marker=mk,
                linestyle=ls, color=colr, label=lab, linewidth=2, markersize=6,
                markeredgecolor="white", markeredgewidth=0.8)
    s = df[(df.ranker == "supervised, LONO") & (df.signal == "margin")]
    for b, e in zip(s.review_budget, s.errors_captured):
        if b in (0.10, 0.20):
            ax.annotate(f"{e*100:.0f}%", (b * 100, e * 100), textcoords="offset points",
                        xytext=(4, -12), fontsize=8, color=INK)
    ax.set_xlabel("Share of mentions sent to human review (%)")
    ax.set_ylabel("Share of all errors captured (%)")
    ax.set_xlim(0, 52); ax.set_ylim(0, 104)
    tidy(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.savefig(OUT / "fig3_triage.png")
    plt.close(fig)


# ---------------------------------------------------------------- figure 4
def fig_dose():
    p = SRC / "stage2_index_dose_response.csv"
    if not p.exists():
        print("  (dose-response CSV not present, skipping fig4)")
        return
    df = pd.read_csv(p)
    syn = df[df.condition.str.startswith("synthetic")].sort_values("train_out_of_index_rate")
    lono = df[df.condition.str.startswith("leave-one")]

    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    spec = [("tail_out_of_index", "Rare, concept never seen", C[0], "o", "-"),
            ("head", "Frequent concepts", C[1], "s", "--"),
            ("all", "All mentions", C[2], "^", ":")]
    for col, lab, colr, mk, ls in spec:
        ax.plot(syn.train_out_of_index_rate * 100, syn[col], marker=mk, linestyle=ls,
                color=colr, label=lab, linewidth=2, markersize=5.5,
                markeredgecolor="white", markeredgewidth=0.8)
    if len(lono):
        r = lono.iloc[0]
        ax.scatter([r.train_out_of_index_rate * 100], [r.tail_out_of_index],
                   marker="*", s=190, color=C[4], zorder=5,
                   edgecolor="white", linewidth=0.8,
                   label="Leave-one-note-out (actual)")
    ax.axvline(7.89, color=MUTED, linestyle="-.", linewidth=1)
    ax.text(8.6, 0.03, "test-set rate", fontsize=7.5, color=MUTED, rotation=90,
            va="bottom")
    ax.set_xlabel("Training positives made out-of-index (%)")
    ax.set_ylabel("Top-1 accuracy")
    ax.set_ylim(0, 1.0)
    tidy(ax)
    ax.legend(frameon=False, fontsize=7.5, loc="center right")
    fig.savefig(OUT / "fig4_dose_response.png")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_bottleneck(); print("  fig1 ok")
    fig_two_tails(); print("  fig2 ok")
    fig_triage(); print("  fig3 ok")
    fig_dose()
    print(f"figures -> {OUT}")
