"""Dataset statistics table for the Methods chapter.

Reproduces the exact train / development / test partition used everywhere else
(SEED and VAL_FRAC mirrored from stage2_reranker_full.py) and reports, per
split, the number of notes, the number of annotated mentions, the number of
distinct gold SNOMED CT concepts, and the mean mentions per note.

Also reports the head / torso / tail composition of the evaluation split and
how many evaluation gold concepts never appear in the training annotations,
because the Methods text quotes both.

    python dataset_stats.py
"""
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from data import load, split_notes  # noqa: E402

SEED = 20260723          # mirrored from stage2_reranker_full.py
VAL_FRAC = 0.2


def main():
    notes, ann = load()
    train_notes, test_notes = split_notes(ann)

    tn = sorted(test_notes)
    random.Random(SEED).shuffle(tn)
    dev_notes = set(tn[:max(1, int(len(tn) * VAL_FRAC))])
    eval_notes = set(tn) - dev_notes

    splits = [
        ("Train", train_notes),
        ("Development", dev_notes),
        ("Test (evaluation)", eval_notes),
    ]

    # Same convention as build_test_mentions(): a note joins the test side if it
    # carries any 'test' annotation, and only its 'test' annotations are scored.
    # Training notes carry train annotations only, so no filter applies there.
    rows = []
    for name, ns in splits:
        sub = ann[ann.note_id.isin(ns)]
        if name != "Train":
            sub = sub[sub.annotation_type == "test"]
        n_notes = len(ns)
        n_ment = len(sub)
        n_conc = sub.concept_id.nunique()
        rows.append((name, n_notes, n_ment, n_conc, n_ment / n_notes))

    w = max(len(r[0]) for r in rows)
    print(f"{'Split':<{w}} {'Notes':>7} {'Mentions':>10} {'Unique gold':>12} {'Ment/note':>10}")
    for name, a, b, c, d in rows:
        print(f"{name:<{w}} {a:>7,} {b:>10,} {c:>12,} {d:>10.1f}")

    tot_notes = sum(r[1] for r in rows)
    tot_ment = sum(r[2] for r in rows)
    print(f"{'Total':<{w}} {tot_notes:>7,} {tot_ment:>10,} "
          f"{ann.concept_id.nunique():>12,} {tot_ment / tot_notes:>10.1f}")

    # composition of the evaluation split, quoted in Methods
    train_freq = Counter(int(c) for c in ann[ann.note_id.isin(train_notes)].concept_id)
    ev = ann[(ann.note_id.isin(eval_notes)) & (ann.annotation_type == "test")]
    strat = Counter()
    for c in ev.concept_id:
        f = train_freq.get(int(c), 0)
        strat["head" if f >= 20 else "torso" if f >= 3 else "tail"] += 1
    ev_conc = set(int(c) for c in ev.concept_id)
    unseen = sum(1 for c in ev_conc if train_freq.get(c, 0) == 0)

    print("\nEvaluation split composition")
    for k in ("head", "torso", "tail"):
        print(f"  {k:<6} {strat[k]:>7,} mentions")
    print(f"  distinct gold concepts        {len(ev_conc):>7,}")
    print(f"  of which absent from training {unseen:>7,} "
          f"({unseen / len(ev_conc):.1%})")

    # overlap of concept inventories, used for the "index" definition
    tr_conc = set(int(c) for c in ann[ann.note_id.isin(train_notes)].concept_id)
    print(f"\nDistinct gold concepts in training  {len(tr_conc):,}")
    print(f"Shared with the evaluation split    {len(tr_conc & ev_conc):,}")


if __name__ == "__main__":
    main()
