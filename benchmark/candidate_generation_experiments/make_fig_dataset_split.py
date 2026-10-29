"""Dataset split and pipeline diagram for the Methods chapter.

Kept separate from make_figures.py because that script derives every mark from
a result CSV, whereas this one is a schematic: the counts are constants, and
they are the counts printed by dataset_stats.py, which recomputes the partition
from the annotations under the same seed the experiments use. If the split ever
changes, re-run dataset_stats.py and update COUNTS below.

The diagram answers three questions a reader needs before Results: how the
corpus divides, what each division is allowed to be used for, and where the
three stages sit relative to that division.

    python make_fig_dataset_split.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "thesis_drafts" / "figures"

# palette mirrored from make_figures.py so the figure set stays one system
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "font.size": 8.5, "text.color": INK,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})

# from dataset_stats.py
CORPUS = ("SNOMED CT Entity Linking Challenge release\n"
          "272 de-identified MIMIC-IV discharge summaries\n"
          "75,209 annotated mentions  ·  6,595 distinct SNOMED CT concepts")
COUNTS = [
    ("Train", "204 notes", "52,078 mentions", "5,317 distinct concepts",
     "fits the span detector, the\ntraining prior and the reranker", C[0]),
    ("Development", "13 notes", "4,455 mentions", "1,476 distinct concepts",
     "sets fusion weights, ranker\nweights and gate thresholds", C[3]),
    ("Test", "55 notes", "18,676 mentions", "3,496 distinct concepts",
     "every reported result;\nnever used for fitting", C[2]),
]
STAGES = [
    ("Stage 0\nspan detection", "free text in,\ncharacter spans out"),
    ("Stage 1\ncandidate retrieval", "one span in,\n60 candidates out"),
    ("Stage 2\ncandidate ranking", "60 candidates in,\none concept out"),
]


def box(ax, x, y, w, h, text, ec, fc="#ffffff", lw=1.0, weight=None, size=8.5,
        ls="solid"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=size, fontweight=weight, color=INK, linespacing=1.45)


def arrow(ax, xy_from, xy_to, color=MUTED, ls="solid", lw=0.9):
    ax.add_patch(FancyArrowPatch(xy_from, xy_to, arrowstyle="-|>", mutation_scale=9,
                                 linewidth=lw, color=color, linestyle=ls,
                                 shrinkA=0, shrinkB=0))


def main():
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ---- corpus -----------------------------------------------------------
    box(ax, 0.02, 0.845, 0.96, 0.125, CORPUS, MUTED, fc="#f7f7f4", size=7.9)

    # ---- split boxes ------------------------------------------------------
    xs = [0.02, 0.35, 0.68]
    w, y, h = 0.30, 0.575, 0.20
    for (name, notes, ments, concs, _use, col), x in zip(COUNTS, xs):
        arrow(ax, (x + w / 2, 0.855), (x + w / 2, y + h))
        body = "%s\n%s\n%s" % (notes, ments, concs)
        box(ax, x, y, w, h, "", col, lw=1.4)
        ax.text(x + w / 2, y + h - 0.045, name, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=col)
        ax.text(x + w / 2, y + h / 2 - 0.032, body, ha="center", va="center",
                fontsize=8.5, color=INK, linespacing=1.5)

    # ---- what each split may be used for ---------------------------------
    for (name, _n, _m, _c, use, col), x in zip(COUNTS, xs):
        ax.text(x + w / 2, 0.525, use, ha="center", va="center", fontsize=7.6,
                color=MUTED, style="italic", linespacing=1.4)

    # ---- pipeline ---------------------------------------------------------
    py, ph, pw = 0.315, 0.145, 0.265
    pxs = [0.055, 0.3675, 0.68]
    for (title, io), x in zip(STAGES, pxs):
        box(ax, x, py, pw, ph, "", INK, lw=1.0)
        ax.text(x + pw / 2, py + ph - 0.042, title, ha="center", va="center",
                fontsize=8.8, fontweight="bold", color=INK, linespacing=1.35)
        ax.text(x + pw / 2, py + 0.038, io, ha="center", va="center",
                fontsize=7.6, color=MUTED, linespacing=1.4)
    for a, b in zip(pxs, pxs[1:]):
        arrow(ax, (a + pw, py + ph / 2), (b, py + ph / 2), color=INK, lw=1.1)

    # Which split is allowed to touch which stage, and in what capacity. Shown
    # as rows aligned under each stage column rather than as arrows: training
    # annotations fit all three stages and the test notes are read by all three
    # and fit none, so an arrow diagram of the true relation crosses itself and
    # becomes unreadable. "fit" learns parameters, "tune" only selects
    # hyperparameters, "evaluate" never influences either.
    USE = {
        "Train":       ["fit detector", "fit training prior", "fit reranker"],
        "Development": ["—", "tune fusion weights", "tune weights, thresholds"],
        "Test":        ["evaluate", "evaluate", "evaluate"],
    }
    for r, (name, _n, _m, _c, _u, col) in enumerate(COUNTS):
        ry = py - 0.045 - r * 0.042
        for s, x in enumerate(pxs):
            ax.plot([x + 0.012], [ry], marker="o", markersize=3.4, color=col,
                    clip_on=False)
            ax.text(x + 0.030, ry, USE[name][s], ha="left", va="center",
                    fontsize=7.3, color=INK if name != "Development" else MUTED)

    ax.text(0.5, 0.055, "Retrieval needs no tuning on test notes, so it is evaluated on all "
                        "23,131 test-side mentions;\nranking and end-to-end results use the "
                        "18,676 mentions of the 55 test notes.",
            ha="center", va="center", fontsize=7.6, color=MUTED, linespacing=1.5)

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "fig5_dataset_split.png"
    fig.savefig(p)
    plt.close(fig)
    print("wrote %s" % p)


if __name__ == "__main__":
    main()
