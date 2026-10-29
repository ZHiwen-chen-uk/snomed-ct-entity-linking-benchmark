"""Experiment 1b, leave-one-note-out (LONO) training of the supervised reranker.

Experiment 1a (`stage2_feature_hygiene.py`) showed the in-index blind spot is
causal: the four train-derived features (`train_rr`, `prior`, `log_freq`,
`in_train`) drive out-of-index tail accuracy to 0.0305, and deleting all four
lifts it 16x to 0.4993, but at a cost of 11.6pp of head accuracy, because those
features are genuinely informative wherever the gold IS in the index.

So hygiene BUYS the tail by SACRIFICING the head. 1b asks whether you can have
both. The blind spot exists because a train mention's gold is in the index by
construction. That annotation is what put it there, so gold-in-index is 1.000
in every training stratum and the model never observes a positive with
`in_train = 0`. The training signal has zero variance on the tail's defining
property.

LONO creates that variance. For every train mention, the four train-derived
features are recomputed from a CandidateGenerator fitted on the other 203 train
notes, so a gold concept that occurs only in its own note becomes out-of-index
exactly as a test tail gold is. The model can then learn WHEN to trust the prior
instead of being denied it.

One design note. LONO is applied to the features and not to the pool, which is
the right choice and not a shortcut. At deployment the full training index is
available for retrieval, so a test-time tail mention already gets a pool built
from the full index; there is no train/test mismatch in pool construction. The
mismatch being repaired is purely in the supervision signal. The regime this
reproduces is exactly the test one: the gold is usually still reachable (77.6%
of out-of-index tail golds are in the pool) but unsupported by the train-derived
features.

Evaluation uses the ORIGINAL eval features, unchanged, because at test time the
full training index is what a deployed system would have.

Runs off the cached matrices from stage2_reranker_full.py; no SapBERT, no
ontology, no re-retrieval. The recomputed train matrix is cached so later
experiments can reuse it.

    KMP_DUPLICATE_LIB_OK=TRUE python stage2_lono_training.py
"""
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BM = HERE.parent
sys.path.insert(0, str(BM)); sys.path.insert(0, str(HERE))

from data import CandidateGenerator, _norm, load, split_notes
from evaluate import freq_stratum
import test_candidate_generation as tcg
from stage2_feature_hygiene import (FEAT_EVAL, FEAT_TRAIN, FNAMES, LADDER, SEED,
                                    accuracies, eval_mentions, load_feats,
                                    picks_from)

POOL = 60                      # every mention contributes exactly POOL rows
LONO_TRAIN = HERE / ".rrfull_feat_train_lono.npz"
# column indices of the four train-derived features, in FNAMES order
IDX = {n: FNAMES.index(n) for n in ("train_rr", "prior", "log_freq", "in_train")}


def build_lono_train(Xtr, meta_tr, t0):
    """Recompute the four train-derived features leave-one-note-out.

    Returns (X_lono, stats) where stats reports the manipulation check: how much
    variance on index membership LONO actually created.
    """
    if LONO_TRAIN.exists():
        z = np.load(LONO_TRAIN, allow_pickle=True)
        print(f"  LONO train features loaded from cache: {z['X'].shape}", flush=True)
        return z["X"], dict(z["stats"].item())

    notes, ann = load()
    train_notes, _ = split_notes(ann)
    tr_ann = ann[ann.note_id.isin(train_notes)]

    # mention order is the itertuples order of tr_ann, and every mention owns
    # exactly POOL consecutive rows of Xtr, asserted below.
    rows = [(r.note_id, int(r.start), str(r.span), int(r.concept_id))
            for r in tr_ann.itertuples()]
    n_mentions = len(rows)
    assert n_mentions * POOL == Xtr.shape[0], \
        f"{n_mentions} mentions x {POOL} != {Xtr.shape[0]} rows"
    for i in (0, n_mentions // 2, n_mentions - 1):
        nid, st, _, _ = rows[i]
        m_nid, m_st, _ = meta_tr[i * POOL]
        assert m_nid == nid and int(m_st) == st, f"row/meta misalignment at {i}"
    print(f"  {n_mentions} train mentions x {POOL} rows, alignment verified",
          flush=True)

    by_note = {}
    for i, (nid, _, _, _) in enumerate(rows):
        by_note.setdefault(nid, []).append(i)

    X = Xtr.copy()
    gold_in_index, gold_prior_sum = 0, 0.0
    for k, (nid, idxs) in enumerate(sorted(by_note.items())):
        cg_n = CandidateGenerator().fit(tr_ann, set(train_notes) - {nid})
        gf = cg_n.global_freq
        for i in idxs:
            _, _, span, gold = rows[i]
            rank = {c: r for r, c in enumerate(cg_n.generate(span))}
            f2c = cg_n.form_to_concepts.get(_norm(span))
            denom = float(sum(f2c.values())) if f2c else 0.0
            base = i * POOL
            for j in range(POOL):
                c = int(meta_tr[base + j][2])
                r = rank.get(c)
                X[base + j, IDX["train_rr"]] = 1.0 / (r + 1) if r is not None else 0.0
                X[base + j, IDX["prior"]] = (f2c.get(c, 0) / denom) if denom else 0.0
                X[base + j, IDX["log_freq"]] = np.log1p(gf.get(c, 0))
                X[base + j, IDX["in_train"]] = 1.0 if c in gf else 0.0
            gold_in_index += 1 if gold in gf else 0
            gold_prior_sum += (f2c.get(gold, 0) / denom) if denom else 0.0
        if (k + 1) % 25 == 0:
            print(f"    LONO note {k+1}/{len(by_note)} ({time.time()-t0:.0f}s)",
                  flush=True)

    stats = {"n_mentions": n_mentions,
             "gold_in_index_rate": gold_in_index / n_mentions,
             "mean_gold_prior": gold_prior_sum / n_mentions}
    np.savez_compressed(LONO_TRAIN, X=X, stats=np.array(stats, dtype=object))
    print(f"  LONO features built ({time.time()-t0:.0f}s)", flush=True)
    return X, stats


def fit_eval(Xtr, ytr, Xev, meta_ev, ev, train_freq, cols, label, t0):
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                         max_depth=6, l2_regularization=1.0,
                                         random_state=SEED)
    clf.fit(Xtr[:, cols], ytr)
    proba = clf.predict_proba(Xev[:, cols])[:, 1]
    acc = accuracies(picks_from(proba, meta_ev), ev, train_freq)
    print(f"  {label}: all={acc['all']:.4f} head={acc['head']:.4f} "
          f"torso={acc['torso']:.4f} tail={acc['tail']:.4f} "
          f"(tail_out={acc['tail_out_of_index']:.4f} "
          f"tail_in={acc['tail_in_index']:.4f})  ({time.time()-t0:.0f}s)", flush=True)
    return acc


def main():
    t0 = time.time()
    print("loading cached feature matrices...", flush=True)
    Xtr, ytr, meta_tr = load_feats(FEAT_TRAIN, "train")
    Xev, yev, meta_ev = load_feats(FEAT_EVAL, "eval")

    print("rebuilding eval split...", flush=True)
    ev, train_freq = eval_mentions(t0)

    print("building leave-one-note-out train features...", flush=True)
    Xlono, stats = build_lono_train(Xtr, meta_tr, t0)

    # manipulation check: how much index-membership variance did LONO create?
    orig_gold_in = 1.0     # gold-in-index is 1.000 in every train stratum
    print(f"\nMANIPULATION CHECK: train gold-in-index {orig_gold_in:.4f} -> "
          f"{stats['gold_in_index_rate']:.4f}; mean gold prior -> "
          f"{stats['mean_gold_prior']:.4f}", flush=True)

    all_cols = list(range(len(FNAMES)))
    hyg_cols = [i for i, n in enumerate(FNAMES) if n not in set(dict(LADDER)["hygiene (9)"])]

    # ---- reference rows (no training) ---------------------------------------
    by_key = {}
    for (nid, st, c) in meta_ev:
        by_key.setdefault((nid, int(st)), []).append(int(c))
    rrf1 = {k: v[0] for k, v in by_key.items()}
    oracle = {}
    for m in ev:
        k = (m.note_id, m.start)
        oracle[k] = m.gold_concept if (k in by_key and
                                       m.gold_concept in set(by_key[k])) else None

    rows = [{"model": "oracle (pool ceiling)", **accuracies(oracle, ev, train_freq)},
            {"model": "rrf_rank1 (no ranker)", **accuracies(rrf1, ev, train_freq)}]

    print("\ntraining...", flush=True)
    rows.append({"model": "full (13), standard train",
                 **fit_eval(Xtr, ytr, Xev, meta_ev, ev, train_freq, all_cols,
                            "full (13), standard train", t0)})
    rows.append({"model": "hygiene (9), standard train",
                 **fit_eval(Xtr, ytr, Xev, meta_ev, ev, train_freq, hyg_cols,
                            "hygiene (9), standard train", t0)})
    rows.append({"model": "full (13), LONO train",
                 **fit_eval(Xlono, ytr, Xev, meta_ev, ev, train_freq, all_cols,
                            "full (13), LONO train", t0)})

    df = pd.DataFrame(rows).round(4)

    n = {s: sum(1 for m in ev
                if freq_stratum(m.gold_concept, train_freq) == s)
         for s in ("head", "torso", "tail")}

    out = HERE / "stage2_out"; out.mkdir(exist_ok=True)
    df.to_csv(out / "stage2_lono_training.csv", index=False)
    lines = [
        "# Experiment 1b — leave-one-note-out (LONO) training", "",
        f"train = {stats['n_mentions']:,} train mentions x {POOL} pool rows · "
        f"eval = canonical split {len(ev):,} mentions "
        f"(head {n['head']:,} / torso {n['torso']:,} / tail {n['tail']:,}) · "
        "same HistGradientBoosting and same weighted RRF@60 pool as "
        "`stage2_reranker_full.py`.", "",
        "LONO recomputes `train_rr`, `prior`, `log_freq` and `in_train` for every "
        "train mention from a CandidateGenerator fitted on the other 203 train "
        "notes. The pool is held fixed and the eval features are unchanged: at "
        "deployment the full training index is available for retrieval, so the "
        "only train/test mismatch being repaired is in the supervision signal.", "",
        "**Manipulation check.** Under standard training a train mention's gold is "
        "in the index by construction, so gold-in-index = 1.0000 in every stratum "
        f"and the model never sees a positive with `in_train = 0`. Under LONO that "
        f"rate falls to **{stats['gold_in_index_rate']:.4f}** (mean gold prior "
        f"{stats['mean_gold_prior']:.4f}), so "
        f"{(1-stats['gold_in_index_rate'])*100:.1f}% of training positives are now "
        "out-of-index. The test tail is 51.5% out-of-index, so the distributions "
        "are still not matched, but the training signal now has non-zero variance "
        "on the property that defines the tail.", "",
        tcg.df_to_markdown(df), "",
        "`hygiene (9)` is unaffected by LONO: the four features LONO recomputes "
        "are exactly the four it drops, so a LONO-trained hygiene model is "
        "numerically identical to the standard-trained one and is not reported "
        "twice.", "",
    ]
    (out / "STAGE2_LONO_TRAINING.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + df.to_string(index=False))
    print(f"\nwrote {out/'STAGE2_LONO_TRAINING.md'} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
