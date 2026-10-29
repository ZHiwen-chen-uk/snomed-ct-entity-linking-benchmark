"""Dose-response: does the in-index blind spot appear and vanish on demand?

Experiments 1a and 1b established that a reranker trained on in-domain
annotations fails on concepts absent from the training index, and that
recomputing the four index-derived features leave-one-note-out repairs it. Both
are observations on one corpus. If the mechanism is what it is claimed to be,
then the failure should be a smooth function of ONE quantity, the share of
training positives whose gold concept is absent from the index, and should be
switchable at will by moving that quantity, without touching anything else.

This sweeps it directly. For a fraction rho of training mentions, the four
index-derived features are recomputed as though that mention's gold concept had
never been annotated. rho = 0 is the standard model; leave-one-note-out lands
near rho = 0.073 as a by-product of note structure; larger values are
synthetic. Everything else, pool, architecture, hyperparameters, evaluation, is held fixed.

The transformation is applied ANALYTICALLY to the cached feature matrix rather
than by refitting an index per concept, which is exact for three of the four
features and near-exact for the fourth:

    in_train  gold row -> 0
    log_freq  gold row -> 0
    prior     gold row -> 0; the others share the form's remaining mass, so
              p_c -> p_c / (1 - p_gold), which is exact
    train_rr  gold row -> 0; candidates ranked below the gold in the training
              retriever move up one place. Exact except that removing a concept
              can admit a new one through the retriever's per-token top-20
              truncation, which this cannot see; the effect is confined to
              candidates outside the pool.

    KMP_DUPLICATE_LIB_OK=TRUE python stage2_index_dose_response.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BM = HERE.parent
sys.path.insert(0, str(BM)); sys.path.insert(0, str(HERE))

from stage2_feature_hygiene import (FEAT_EVAL, FEAT_TRAIN, FNAMES, SEED,
                                    accuracies, eval_mentions, load_feats,
                                    picks_from)
from stage2_lono_training import LONO_TRAIN, POOL
import test_candidate_generation as tcg

RHOS = [0.0, 0.02, 0.05, 0.10, 0.20, 0.40, 0.80, 1.0]
I_RR, I_PRIOR, I_FREQ, I_IN = (FNAMES.index(n) for n in
                               ("train_rr", "prior", "log_freq", "in_train"))


def drop_gold_from_index(X, ytr, rho, rng):
    """Recompute the four index-derived features as if a rho-share of training
    mentions had never had their gold concept annotated."""
    n_mentions = X.shape[0] // POOL
    sel = np.flatnonzero(rng.random(n_mentions) < rho)
    Xd = X.copy()
    n_applied = 0
    for i in sel:
        s = i * POOL
        seg_y = ytr[s:s + POOL]
        gi = np.flatnonzero(seg_y == 1)
        if not len(gi):
            continue                      # gold not in pool; nothing to remove
        g = s + int(gi[0])

        # --- train_rr: candidates below the gold move up one place -----------
        rr = Xd[s:s + POOL, I_RR]
        rg = Xd[g, I_RR]
        if rg > 0:
            r_gold = round(1.0 / rg - 1.0)
            present = rr > 0
            ranks = np.where(present, np.round(1.0 / np.maximum(rr, 1e-12) - 1.0), 0.0)
            moved = np.where(present & (ranks > r_gold), ranks - 1.0, ranks)
            Xd[s:s + POOL, I_RR] = np.where(present, 1.0 / (moved + 1.0), 0.0)
        Xd[g, I_RR] = 0.0

        # --- prior: the form's remaining mass renormalises --------------------
        pg = float(Xd[g, I_PRIOR])
        if pg < 1.0:
            Xd[s:s + POOL, I_PRIOR] = Xd[s:s + POOL, I_PRIOR] / (1.0 - pg)
        else:
            Xd[s:s + POOL, I_PRIOR] = 0.0
        Xd[g, I_PRIOR] = 0.0

        Xd[g, I_FREQ] = 0.0
        Xd[g, I_IN] = 0.0
        n_applied += 1
    return Xd, n_applied, n_mentions


def main():
    t0 = time.time()
    Xtr, ytr, _ = load_feats(FEAT_TRAIN, "train")
    Xev, _, meta_ev = load_feats(FEAT_EVAL, "eval")
    ev, train_freq = eval_mentions(t0)

    from sklearn.experimental import enable_hist_gradient_boosting  # noqa
    from sklearn.ensemble import HistGradientBoostingClassifier

    def fit_eval(X, label):
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                             max_depth=6, l2_regularization=1.0,
                                             random_state=SEED)
        clf.fit(X, ytr)
        acc = accuracies(picks_from(clf.predict_proba(Xev)[:, 1], meta_ev),
                         ev, train_freq)
        print(f"  {label}: all={acc['all']:.4f} head={acc['head']:.4f} "
              f"tail={acc['tail']:.4f} tail_out={acc['tail_out_of_index']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        return acc

    rows = []
    rng = np.random.default_rng(SEED)
    for rho in RHOS:
        if rho == 0.0:
            Xd, applied, n_m = Xtr, 0, Xtr.shape[0] // POOL
        else:
            Xd, applied, n_m = drop_gold_from_index(Xtr, ytr, rho, rng)
        rate = applied / n_m
        acc = fit_eval(Xd, f"rho={rho:.2f} (applied {rate:.4f})")
        rows.append({"condition": f"synthetic rho={rho:.2f}",
                     "train_out_of_index_rate": round(rate, 4), **acc})
        if rho != 0.0:
            del Xd

    # the real leave-one-note-out matrix, for reference on the same axis
    Xlono = np.load(LONO_TRAIN, allow_pickle=True)["X"]
    acc = fit_eval(Xlono, "leave-one-note-out")
    rows.append({"condition": "leave-one-note-out (real)",
                 "train_out_of_index_rate": 0.0732, **acc})

    df = pd.DataFrame(rows).round(4)
    out = HERE / "stage2_out"; out.mkdir(exist_ok=True)
    df.to_csv(out / "stage2_index_dose_response.csv", index=False)
    lines = [
        "# Dose-response: the in-index blind spot as a function of training mismatch", "",
        "One quantity is varied, the share of training positives whose gold "
        "concept is absent from the training-derived index, while the pool, the "
        "architecture, the hyperparameters and the evaluation are held fixed. "
        "`rho = 0` is the standard model; the real leave-one-note-out matrix is "
        "shown on the same axis for reference. The evaluation split's own "
        "out-of-index rate is 0.0789 of all mentions.", "",
        tcg.df_to_markdown(df), "",
        "If the blind spot were an idiosyncrasy of this corpus there would be no "
        "reason for out-of-index tail accuracy to track this one quantity. If it "
        "is the claimed mechanism, the curve should rise steeply from zero and "
        "then saturate, while head accuracy erodes only once the training prior "
        "has been destroyed.", "",
    ]
    (out / "STAGE2_INDEX_DOSE_RESPONSE.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + df.to_string(index=False))
    print(f"\nwrote {out/'STAGE2_INDEX_DOSE_RESPONSE.md'} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
