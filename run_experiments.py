"""End-to-end runner for the four-group ontology experiment + charts.

    python run_experiments.py

Pipeline:
  load -> split -> fit candidate generator -> build test mentions ->
  run Baseline / A / B / C -> metrics + stratified analysis -> charts.

Outputs a console table and PNG charts under benchmark/figures/.

NOTE: ships with MockOntology (synthetic structure) and PriorRanker (offline LLM
stand-in), so it runs with zero credentials. Numbers demonstrate the harness;
swap in SnomedOntology + a real LLM ranker for reportable results.
"""
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import CandidateGenerator, build_test_mentions, load, split_notes
from ontology import MockOntology
from pipeline import LLMRanker, PriorRanker, SemTypePredictor, Strategies, GROUPS
from evaluate import evaluate_all, recall_at_k

FIG_DIR = Path(__file__).resolve().parent / "figures"
COLORS = {"Baseline": "#7f8c8d", "A: Pre-filter": "#2980b9",
          "B: Rerank": "#27ae60", "C: Filter+Rerank": "#8e44ad"}


def _bar_color(label):
    return COLORS.get(label, "#34495e")


def chart_headline(summary, recall, title=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    labels = summary.group.tolist()
    colors = [_bar_color(l) for l in labels]

    axes[0].bar(labels, summary.accuracy, color=colors)
    axes[0].set_title("Entity-level accuracy (top-1)")
    axes[0].set_ylabel("accuracy")

    axes[1].bar(labels, summary.char_iou, color=colors)
    axes[1].set_title("Macro char-IoU (gold spans)")
    axes[1].set_ylabel("macro IoU")

    for ax in axes[:2]:
        ax.set_ylim(0, max(0.01, summary[["accuracy", "char_iou"]].values.max()) * 1.25)
        ax.tick_params(axis="x", rotation=20)
        for i, v in enumerate(ax.containers[0].datavalues):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ks = sorted(recall)
    axes[2].plot(ks, [recall[k] for k in ks], "o-", color="#c0392b")
    axes[2].set_title("Candidate recall@k (shared ceiling)")
    axes[2].set_xlabel("k"); axes[2].set_ylabel("recall"); axes[2].set_ylim(0, 1)
    for k in ks:
        axes[2].annotate(f"{recall[k]:.2f}", (k, recall[k]),
                         textcoords="offset points", xytext=(0, 6), fontsize=8)

    fig.suptitle(title or "SNOMED CT entity linking — four-group comparison "
                 "(MockOntology)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "headline.png", dpi=130)
    plt.close(fig)


def chart_strata(strata, title=None):
    panels = [("ambiguity", ["unambiguous", "sibling", "cross-branch"]),
              ("frequency", ["head", "torso", "tail"]),
              ("depth", ["shallow", "deep"])]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (name, order) in zip(axes, panels):
        data = strata[name]
        groups = list(data.keys())
        cats = [c for c in order if any(c in data[g] for g in groups)]
        x = np.arange(len(cats)); w = 0.8 / max(1, len(groups))
        for i, g in enumerate(groups):
            vals = [data[g].get(c, 0) for c in cats]
            ax.bar(x + i * w, vals, w, label=g, color=_bar_color(g))
        ax.set_xticks(x + w * (len(groups) - 1) / 2)
        ax.set_xticklabels(cats)
        ax.set_title(f"Accuracy by {name}")
        ax.set_ylabel("accuracy"); ax.set_ylim(0, 1)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(title or "Stratified accuracy — where each ontology strategy helps",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "stratified.png", dpi=130)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(exist_ok=True)
    notes, ann = load()
    train_notes, test_notes = split_notes(ann)

    cg = CandidateGenerator().fit(ann, train_notes)
    mentions = build_test_mentions(notes, ann, test_notes, cg)
    train_freq = Counter(int(c) for c in
                         ann[ann.note_id.isin(train_notes)].concept_id)

    ontology = MockOntology()
    ranker: LLMRanker = PriorRanker(cg)
    predictor = SemTypePredictor(ontology)
    strat = Strategies(ranker, predictor, ontology)

    print(f"notes: {len(notes)} (train {len(train_notes)} / test {len(test_notes)})")
    print(f"test mentions: {len(mentions)} | candidate cap: {cg.max_candidates}")

    recall = recall_at_k(mentions)
    print("\n=== Candidate recall@k (ceiling for all groups) ===")
    for k in sorted(recall):
        print(f"  recall@{k:<3} = {recall[k]:.4f}")

    summary, strata = evaluate_all(mentions, strat, ontology, train_freq, GROUPS)
    print("\n=== Headline metrics by group ===")
    print(summary.to_string(index=False,
          formatters={"accuracy": "{:.4f}".format, "char_iou": "{:.4f}".format}))

    print("\n=== Accuracy by ambiguity type ===")
    for g, d in strata["ambiguity"].items():
        cells = "  ".join(f"{k}={v:.3f}" for k, v in sorted(d.items()))
        print(f"  {g:<18} {cells}")

    chart_headline(summary, recall)
    chart_strata(strata)
    print(f"\ncharts written to: {FIG_DIR}")
    print("  - headline.png   (accuracy / char-IoU / recall@k)")
    print("  - stratified.png (accuracy by ambiguity / frequency / depth)")
    print("\n[reminder] MockOntology = synthetic structure. Swap in SnomedOntology "
          "+ a real LLM ranker before quoting any number.")


if __name__ == "__main__":
    main()
