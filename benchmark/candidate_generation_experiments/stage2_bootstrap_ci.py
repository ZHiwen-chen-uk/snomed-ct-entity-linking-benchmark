"""Note-clustered bootstrap confidence intervals for the Stage-2 rankers.

Every number in the thesis is currently a point estimate, while the draft
already asserts differences between rankers. Mentions are NOT independent, they are clustered within discharge summaries, and a note contributes hundreds
of correlated mentions, so resampling mentions would understate the variance.
This resamples NOTES with replacement (the cluster bootstrap) and recomputes
each metric on the resampled corpus.

Comparisons are PAIRED: the same resampled notes are used for both rankers in a
difference, so the CI is on the difference and not on two independent estimates.

Rankers compared on gold spans over the canonical 18,676-mention eval split:

    rrf_rank1             no ranker (pool order)
    gated                 the ranker the published end-to-end numbers use
    supervised, standard  the in-index blind-spot model         (1a)
    supervised, hygiene   four train-derived features deleted   (1a)
    supervised, LONO      those features recomputed leave-one-note-out (1b)

Claims specifically under test, all flagged as at-risk:
    LONO vs hygiene on out-of-index tail  (+1.8pp, probably a tie)
    LONO vs gated on tail                 (+3.3pp)
    LONO vs standard everywhere           (large, expected to survive)

Per-mention picks are cached so later analyses (error taxonomy, abstention) can
reuse them without retraining.

    KMP_DUPLICATE_LIB_OK=TRUE python stage2_bootstrap_ci.py [--reps 2000]
"""
import argparse
import pickle
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
sys.path.insert(0, str(BM / "group_a_transformer"))

from candidate_booster import norm
from data import CandidateGenerator, build_test_mentions, load, split_notes
from evaluate import freq_stratum
from stage0_chain_e2e import tune_gate
from stage2_gated import SEED, VAL_FRAC, pick
from stage2_feature_hygiene import FEAT_EVAL, FEAT_TRAIN, FNAMES, LADDER
from stage2_lono_training import LONO_TRAIN
import test_candidate_generation as tcg

PICKS_CACHE = HERE / ".stage2_picks_eval.pkl"
GOLD_POOLS = HERE / ".stage2_pools_full.pkl"
GOLD_TPROB = HERE / ".stage2_typeprob_cache.pkl"
SCOPES = ["all", "head", "torso", "tail", "tail_out_of_index", "tail_in_index"]
PAIRS = [("supervised, LONO", "supervised, hygiene"),
         ("supervised, LONO", "gated"),
         ("supervised, LONO", "supervised, standard"),
         ("supervised, hygiene", "supervised, standard"),
         ("gated", "supervised, standard")]


def scope_of(m, train_freq):
    """Every scope a mention belongs to (mentions count in several)."""
    s = freq_stratum(m.gold_concept, train_freq)
    out = ["all", s]
    if s == "tail":
        out.append("tail_out_of_index" if train_freq.get(m.gold_concept, 0) == 0
                   else "tail_in_index")
    return out


def build_picks(t0):
    if PICKS_CACHE.exists():
        with open(PICKS_CACHE, "rb") as f:
            d = pickle.load(f)
        print(f"  picks loaded from cache ({len(d['picks'])} rankers)", flush=True)
        return d

    from sklearn.experimental import enable_hist_gradient_boosting  # noqa
    from sklearn.ensemble import HistGradientBoostingClassifier
    from stage2_feature_hygiene import eval_mentions, load_feats, picks_from

    Xtr, ytr, _ = load_feats(FEAT_TRAIN, "train")
    Xev, _, meta_ev = load_feats(FEAT_EVAL, "eval")
    ev, train_freq = eval_mentions(t0)
    Xlono = np.load(LONO_TRAIN, allow_pickle=True)["X"]

    hyg = [i for i, n in enumerate(FNAMES)
           if n not in set(dict(LADDER)["hygiene (9)"])]
    allc = list(range(len(FNAMES)))
    specs = [("supervised, standard", Xtr, allc),
             ("supervised, hygiene", Xtr, hyg),
             ("supervised, LONO", Xlono, allc)]

    picks = {}
    for name, X, cols in specs:
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                             max_depth=6, l2_regularization=1.0,
                                             random_state=SEED)
        clf.fit(X[:, cols], ytr)
        picks[name] = picks_from(clf.predict_proba(Xev[:, cols])[:, 1], meta_ev)
        print(f"  picks[{name}] ({time.time()-t0:.0f}s)", flush=True)

    # pool order = no ranker
    by_key = {}
    for (nid, st, c) in meta_ev:
        by_key.setdefault((nid, int(st)), []).append(int(c))
    picks["rrf_rank1"] = {k: v[0] for k, v in by_key.items()}

    # gated, the ranker the published end-to-end numbers use
    notes, ann = load()
    train_notes, test_notes = split_notes(ann)
    cg = CandidateGenerator().fit(ann, train_notes)
    seen_forms = set(cg.form_to_concepts.keys())
    tn = sorted(test_notes); random.Random(SEED).shuffle(tn)
    val_notes = set(tn[:max(1, int(len(tn) * VAL_FRAC))])
    gold = build_test_mentions(notes, ann, test_notes, cg)
    gold_D = pickle.load(open(GOLD_POOLS, "rb"))
    gold_tprob = pickle.load(open(GOLD_TPROB, "rb"))
    for m in gold:
        k = (m.note_id, m.start)
        if k in gold_D["pools"]:
            m.candidates = gold_D["pools"][k]
    w_seen, w_novel = tune_gate(gold_D, gold_tprob,
                                [m for m in gold if m.note_id in val_notes],
                                train_freq, seen_forms)
    missing = [m for m in ev if (m.note_id, m.start) not in gold_D["pools"]]
    assert not missing, f"{len(missing)} eval mentions absent from the gold pool cache"
    picks["gated"] = {(m.note_id, m.start): pick(
        m, gold_D, gold_tprob,
        *(w_seen if norm(m.span) in seen_forms else w_novel)) for m in ev}
    print(f"  picks[gated] gate {w_seen}/{w_novel} ({time.time()-t0:.0f}s)", flush=True)

    rows = [{"note_id": m.note_id, "start": m.start, "gold": m.gold_concept,
             "scopes": scope_of(m, train_freq)} for m in ev]
    d = {"picks": picks, "rows": rows}
    with open(PICKS_CACHE, "wb") as f:
        pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10000)
    args = ap.parse_args()
    t0 = time.time()

    d = build_picks(t0)
    picks, rows = d["picks"], d["rows"]
    rankers = ["rrf_rank1", "gated", "supervised, standard",
               "supervised, hygiene", "supervised, LONO"]

    # ---- correctness matrix: rankers x mentions -----------------------------
    keys = [(r["note_id"], r["start"]) for r in rows]
    correct = {name: np.array([picks[name].get(k) == r["gold"]
                               for k, r in zip(keys, rows)], dtype=bool)
               for name in rankers}
    notes_arr = np.array([r["note_id"] for r in rows])
    uniq_notes = sorted(set(notes_arr))
    note_idx = {n: np.flatnonzero(notes_arr == n) for n in uniq_notes}
    in_scope = {s: np.array([s in r["scopes"] for r in rows], dtype=bool)
                for s in SCOPES}
    print(f"\n{len(rows)} mentions across {len(uniq_notes)} notes; "
          f"{args.reps} cluster-bootstrap replicates", flush=True)

    rng = np.random.default_rng(SEED)
    reps = []
    for b in range(args.reps):
        draw = rng.integers(0, len(uniq_notes), len(uniq_notes))
        sel = np.concatenate([note_idx[uniq_notes[i]] for i in draw])
        rec = {}
        for s in SCOPES:
            mask = in_scope[s][sel]
            n = int(mask.sum())
            for name in rankers:
                rec[(name, s)] = correct[name][sel][mask].mean() if n else np.nan
        reps.append(rec)
        if (b + 1) % 500 == 0:
            print(f"  {b+1}/{args.reps} ({time.time()-t0:.0f}s)", flush=True)

    n_comp = len(PAIRS) * len(SCOPES)          # family size for Bonferroni
    alpha_fw = 0.05 / n_comp

    def ci(vals, alpha=0.05):
        v = np.array([x for x in vals if not np.isnan(x)])
        return np.percentile(v, 100 * alpha / 2), np.percentile(v, 100 * (1 - alpha / 2))

    # ---- per-ranker point estimates + CIs -----------------------------------
    est_rows = []
    for name in rankers:
        row = {"ranker": name}
        for s in SCOPES:
            mask = in_scope[s]
            pt = correct[name][mask].mean()
            lo, hi = ci([r[(name, s)] for r in reps])
            row[s] = f"{pt:.4f} [{lo:.4f}, {hi:.4f}]"
        est_rows.append(row)
    est = pd.DataFrame(est_rows)

    # ---- paired differences --------------------------------------------------
    diff_rows = []
    for a, b_ in PAIRS:
        for s in SCOPES:
            mask = in_scope[s]
            pt = correct[a][mask].mean() - correct[b_][mask].mean()
            ds = [r[(a, s)] - r[(b_, s)] for r in reps]
            lo, hi = ci(ds)
            blo, bhi = ci(ds, alpha_fw)
            diff_rows.append({
                "comparison": f"{a} - {b_}", "scope": s,
                "diff": round(float(pt), 4),
                "95% CI": f"[{lo:.4f}, {hi:.4f}]",
                "excludes 0": "yes" if (lo > 0 or hi < 0) else "NO",
                f"Bonferroni {100*(1-alpha_fw):.2f}% CI": f"[{blo:.4f}, {bhi:.4f}]",
                "survives correction": "yes" if (blo > 0 or bhi < 0) else "NO"})
    diffs = pd.DataFrame(diff_rows)

    out = HERE / "stage2_out"; out.mkdir(exist_ok=True)
    est.to_csv(out / "stage2_bootstrap_estimates.csv", index=False)
    diffs.to_csv(out / "stage2_bootstrap_diffs.csv", index=False)
    lines = [
        "# Stage-2 rankers — note-clustered bootstrap confidence intervals", "",
        f"eval = canonical split {len(rows):,} mentions in {len(uniq_notes)} notes · "
        f"{args.reps:,} bootstrap replicates, resampling NOTES with replacement · "
        "95% percentile intervals · differences are PAIRED (same resample for both "
        "rankers).", "",
        "Mentions within a discharge summary are correlated, so resampling mentions "
        "would understate the variance; the cluster is the note. With only "
        f"{len(uniq_notes)} clusters the intervals are necessarily wide — that is "
        "the honest precision of this corpus, not a defect of the estimator.", "",
        "## Top-1 accuracy with 95% CI", "", tcg.df_to_markdown(est), "",
        "## Paired differences", "",
        f"{n_comp} comparisons are reported, so the per-comparison 95% interval "
        f"would inflate the family-wise error rate. A Bonferroni-corrected "
        f"{100*(1-alpha_fw):.2f}% interval is given alongside; `survives "
        f"correction` is the conservative verdict and is the one to quote. Paired "
        "differences are far tighter than the marginal intervals above because the "
        "same resampled notes are used for both rankers.", "",
        tcg.df_to_markdown(diffs), "",
    ]
    (out / "STAGE2_BOOTSTRAP_CI.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + est.to_string(index=False))
    print("\n" + diffs.to_string(index=False))
    print(f"\nwrote {out/'STAGE2_BOOTSTRAP_CI.md'} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
