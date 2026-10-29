"""Experiment 1a, feature-hygiene ablation of the supervised reranker.

stage2_reranker_full.py produced the negative result this starts from. A
HistGradientBoosting reranker trained on all 52,078 in-domain train annotations
is the best ranker overall (0.7885) and on head (0.9395) while being the worst on
tail (0.2472, below the unranked RRF top-1 of 0.4134).

The mechanism proposed for it, the in-index blind spot, is that four of the
thirteen features are functions of the training annotation set and not of the
text:

    train_rr  reciprocal rank in the train-derived retriever list
    prior     P(concept | surface form) from train counts        (data.py:146)
    log_freq  log1p(train frequency of the candidate)            (data.py:121)
    in_train  candidate occurs in the train annotations at all   (data.py:121)

In training these are near-perfect predictors of the gold, because a train
mention's gold is in the index by construction (gold-in-index = 1.000 in every
train stratum). At test time 51.5% of tail mentions have a gold the training
signal says is impossible. The model has never seen a positive with in_train=0,
so it cannot rank one first.

This script tests that mechanism CAUSALLY by removing the leaky features and
retraining. It is a ladder, not a single ablation, so the claim can be stated as
a dose-response rather than a single contrast:

    full (13)          reproduces the published row  (harness check)
    -in_train          removes the binary index-membership feature
    -prior             removes the form-conditioned train prior
    -prior -in_train   removes both
    hygiene (9)        removes all four train-derived features

The decisive cut is inside the tail. `tail` = train_freq < 3, which MIXES two
populations: gold seen 1-2 times (in the index) and gold never seen (out of the
index). The blind-spot mechanism predicts the full model collapses specifically
on the freq==0 half and that hygiene lifts that half, so both halves are
reported separately for every feature set.

Everything runs off the cached feature matrices written by
stage2_reranker_full.py; nothing is re-retrieved and neither SapBERT nor the
SNOMED ontology is loaded.

    KMP_DUPLICATE_LIB_OK=TRUE python stage2_feature_hygiene.py
"""
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BM = HERE.parent
sys.path.insert(0, str(BM)); sys.path.insert(0, str(HERE))

from data import CandidateGenerator, build_test_mentions, load, split_notes
from evaluate import freq_stratum
import test_candidate_generation as tcg

# --- mirrored from stage2_reranker_full.py so the split is bit-identical ------
SEED = 20260723
VAL_FRAC = 0.2
FEAT_TRAIN = HERE / ".rrfull_feat_train.npz"
FEAT_EVAL = HERE / ".rrfull_feat_eval.npz"
FNAMES = ["dense", "dense_rr", "lexical", "lex_rr", "char", "char_rr", "synonym",
          "train_rr", "prior", "log_freq", "in_train", "depth", "typeA"]
LEAKY = ["train_rr", "prior", "log_freq", "in_train"]

# ladder of feature sets: label -> features dropped
LADDER = [
    ("full (13)", []),
    ("-in_train (12)", ["in_train"]),
    ("-prior (12)", ["prior"]),
    ("-prior -in_train (11)", ["prior", "in_train"]),
    ("hygiene (9)", LEAKY),
]


def load_feats(path, tag):
    z = np.load(path, allow_pickle=True)
    X, y, meta = z["X"], z["y"], list(z["meta"])
    assert X.shape[1] == len(FNAMES), f"{tag}: {X.shape[1]} cols vs {len(FNAMES)} names"
    print(f"  features[{tag}] {X.shape} pos={int(y.sum())}", flush=True)
    return X, y, meta


def eval_mentions(t0):
    """Rebuild the canonical 18,676 eval split, identical construction to
    stage2_reranker_full.main(). Cheap: dict fits only, no retrievers."""
    notes, ann = load()
    train_notes, test_notes = split_notes(ann)
    cg = CandidateGenerator().fit(ann, train_notes)
    train_freq = Counter(int(c) for c in ann[ann.note_id.isin(train_notes)].concept_id)
    test_mentions = build_test_mentions(notes, ann, test_notes, cg)
    tn = sorted(test_notes)
    random.Random(SEED).shuffle(tn)
    val_notes = set(tn[:max(1, int(len(tn) * VAL_FRAC))])
    ev = [m for m in test_mentions if m.note_id not in val_notes]
    print(f"  eval split = {len(ev)} mentions ({time.time()-t0:.0f}s)", flush=True)
    return ev, train_freq


def accuracies(picks, mentions, train_freq):
    """Standard stratified accuracy plus the tail split by index membership.

    A mention whose key is absent from `picks` (gold unreachable, or no pool)
    counts as an error, so denominators are the full mention counts.
    """
    def a(sub):
        return None if not sub else sum(
            1 for m in sub if picks.get((m.note_id, m.start)) == m.gold_concept) / len(sub)

    by_s = {s: [m for m in mentions
                if freq_stratum(m.gold_concept, train_freq) == s]
            for s in ("head", "torso", "tail")}
    tail = by_s["tail"]
    tail_out = [m for m in tail if train_freq.get(m.gold_concept, 0) == 0]
    tail_in = [m for m in tail if train_freq.get(m.gold_concept, 0) > 0]
    return {"all": a(mentions), "head": a(by_s["head"]), "torso": a(by_s["torso"]),
            "tail": a(tail), "tail_out_of_index": a(tail_out),
            "tail_in_index": a(tail_in)}


def picks_from(scores, meta):
    """argmax per mention; ties broken by earlier pool position (stable max)."""
    best = {}
    for (nid, st, c), p in zip(meta, scores):
        k = (nid, int(st))
        cur = best.get(k)
        if cur is None or p > cur[1]:
            best[k] = (int(c), float(p))
    return {k: v[0] for k, v in best.items()}


def main():
    t0 = time.time()
    print("loading cached feature matrices...", flush=True)
    Xtr, ytr, _ = load_feats(FEAT_TRAIN, "train")
    Xev, yev, meta_ev = load_feats(FEAT_EVAL, "eval")

    print("rebuilding eval split...", flush=True)
    ev, train_freq = eval_mentions(t0)

    # ---- reference rows ------------------------------------------------------
    by_key = {}
    for (nid, st, c) in meta_ev:
        by_key.setdefault((nid, int(st)), []).append(int(c))
    missing = [m for m in ev if (m.note_id, m.start) not in by_key]
    print(f"  mentions with an empty pool: {len(missing)}", flush=True)

    # RRF rank-1: pool rows were appended in fused order, so index 0 is rank 1.
    rrf1 = {k: v[0] for k, v in by_key.items()}
    oracle = {k: None for k in by_key}
    for m in ev:
        k = (m.note_id, m.start)
        if k in by_key and m.gold_concept in set(by_key[k]):
            oracle[k] = m.gold_concept

    rows = [{"model": "oracle (pool ceiling)", **accuracies(oracle, ev, train_freq)},
            {"model": "rrf_rank1 (no ranker)", **accuracies(rrf1, ev, train_freq)}]

    from sklearn.experimental import enable_hist_gradient_boosting  # noqa
    from sklearn.ensemble import HistGradientBoostingClassifier

    imps = {}
    for label, dropped in LADDER:
        cols = [i for i, n in enumerate(FNAMES) if n not in set(dropped)]
        kept = [FNAMES[i] for i in cols]
        print(f"\n[{label}] training on {len(cols)} features "
              f"(dropped: {dropped or 'none'})...", flush=True)
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                             max_depth=6, l2_regularization=1.0,
                                             random_state=SEED)
        clf.fit(Xtr[:, cols], ytr)
        proba = clf.predict_proba(Xev[:, cols])[:, 1]
        acc = accuracies(picks_from(proba, meta_ev), ev, train_freq)
        rows.append({"model": label, **acc})
        if hasattr(clf, "feature_importances_"):
            imps[label] = sorted(zip(kept, clf.feature_importances_),
                                 key=lambda kv: -kv[1])
        print(f"  {label}: all={acc['all']:.4f} head={acc['head']:.4f} "
              f"torso={acc['torso']:.4f} tail={acc['tail']:.4f} "
              f"(tail_out={acc['tail_out_of_index']:.4f} "
              f"tail_in={acc['tail_in_index']:.4f})  ({time.time()-t0:.0f}s)",
              flush=True)

    df = pd.DataFrame(rows).round(4)

    # ---- stratum sizes for the caption --------------------------------------
    n = {s: sum(1 for m in ev if freq_stratum(m.gold_concept, train_freq) == s)
         for s in ("head", "torso", "tail")}
    tail_out = sum(1 for m in ev if freq_stratum(m.gold_concept, train_freq) == "tail"
                   and train_freq.get(m.gold_concept, 0) == 0)
    n_tail_in = n["tail"] - tail_out

    out = HERE / "stage2_out"; out.mkdir(exist_ok=True)
    df.to_csv(out / "stage2_feature_hygiene.csv", index=False)
    lines = [
        "# Experiment 1a — feature-hygiene ablation of the supervised reranker", "",
        f"train = all {Xtr.shape[0]:,} candidate rows from 52,078 train mentions · "
        f"eval = canonical split {len(ev):,} mentions "
        f"(head {n['head']:,} / torso {n['torso']:,} / tail {n['tail']:,}) · "
        f"pool = weighted RRF@60 · model = HistGradientBoosting "
        "(max_iter=300, lr=0.08, depth=6, l2=1.0), identical to "
        "`stage2_reranker_full.py`.", "",
        f"The tail is split by index membership: **tail_out_of_index** "
        f"({tail_out:,} mentions, train_freq = 0, gold absent from the training "
        f"annotations) and **tail_in_index** ({n_tail_in:,} mentions, "
        f"train_freq 1-2).", "",
        "Leaky features (functions of the training annotation set, not of the "
        "text): `" + "`, `".join(LEAKY) + "`.", "",
        tcg.df_to_markdown(df), "",
    ]
    if imps:
        lines += ["### Feature importance", ""]
        for label, imp in imps.items():
            lines += [f"**{label}**", "", "| feature | importance |", "| --- | --- |"]
            lines += [f"| {nm} | {v:.4f} |" for nm, v in imp]
            lines.append("")
    else:
        lines += [
            "HistGradientBoostingClassifier exposes no `feature_importances_`, so "
            "no importance table is reported. The ladder itself is the stronger "
            "substitute: it measures each feature group by its effect on the "
            "evaluation metric that matters, under a controlled retrain, rather "
            "than by an internal split-count proxy.", ""]
    (out / "STAGE2_FEATURE_HYGIENE.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + df.to_string(index=False))
    print(f"\nwrote {out/'STAGE2_FEATURE_HYGIENE.md'} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
