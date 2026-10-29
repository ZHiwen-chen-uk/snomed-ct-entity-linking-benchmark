"""Ontological taxonomy of RANKING errors, plus where the gold actually sits.

The submitted proposal's Group B applied an is-a distance penalty to fix
granularity mismatch. It was dropped on two grounds. A shared-ancestor boost
fires uniformly across siblings, and a distance penalty cannot separate siblings
because siblings are already close, so the idea does not fit the ambiguity it was
meant to resolve. It was also empirically inert. Both of those are arguments
where a measurement would be better, and this script measures it.

A RANKING error is a mention whose gold IS in the RRF@60 pool but is not picked.
Each one is classified by the is-a relation between the predicted and the gold
concept:

    too_broad     predicted is a proper ancestor of gold   (granularity)
    too_specific  predicted is a proper descendant of gold (granularity)
    sibling       predicted and gold share a parent
    same_branch   common ancestor at depth >= 2 but none of the above
    cross_branch  deepest common ancestor is the root or a top-level hierarchy
    unrelated     no common ancestor at all

Those first two labels were the wrong way round in the first version of this
script. Only their sum is reported in the write-up, so no published number
moves, but the direction of any single example does.

`too_broad + too_specific` is exactly the slice a granularity fixer could
address, so its share bounds what Group B could ever have been worth. The share
is also reported as a fraction of ALL eval mentions, which converts it directly
into an accuracy ceiling: "a perfect granularity fixer is worth at most X points".

Second output: the RANK of the gold under each ranker among ranking errors. If
the gold usually sits at rank 30+, then the `perfect ranking` headroom measured
by the oracle substitution is largely unreachable in practice, a deliberately
self-critical check.

Both the published ranker (`gated`) and the new best (`supervised, LONO`) are
reported, stratified, with the tail split by index membership.

    KMP_DUPLICATE_LIB_OK=TRUE python stage2_error_taxonomy.py
"""
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
from ontology import SnomedOntology
from stage0_chain_e2e import GOLD_POOLS, GOLD_TPROB, tune_gate
from stage2_gated import SEED, SNOMED_DIR, TYPE4, VAL_FRAC
from stage2_feature_hygiene import FEAT_EVAL, FEAT_TRAIN, load_feats
from stage2_lono_training import LONO_TRAIN
import test_candidate_generation as tcg

SCOPES = ["all", "head", "torso", "tail", "tail_out_of_index", "tail_in_index"]
CATS = ["too_broad", "too_specific", "sibling", "same_branch", "cross_branch",
        "unrelated"]
GRANULARITY = ("too_broad", "too_specific")


def classify(ont, pred, gold):
    """is-a relation of a predicted concept to the gold concept."""
    pred, gold = int(pred), int(gold)
    a_p, a_g = ont._ancestors(pred), ont._ancestors(gold)
    if gold in a_p:
        return "too_specific"        # gold is an ancestor of pred -> pred is a
                                     # descendant of the gold, so it is finer
    if pred in a_g:
        return "too_broad"           # pred is an ancestor of gold, so it is coarser
    if set(ont.parents.get(pred, ())) & set(ont.parents.get(gold, ())):
        return "sibling"
    common = a_p.keys() & a_g.keys()
    if not common:
        return "unrelated"
    return "same_branch" if max(a_p[x] for x in common) >= 2 else "cross_branch"


def scopes_of(m, train_freq):
    s = freq_stratum(m.gold_concept, train_freq)
    out = ["all", s]
    if s == "tail":
        out.append("tail_out_of_index" if train_freq.get(m.gold_concept, 0) == 0
                   else "tail_in_index")
    return out


def order_from_scores(pool, scores):
    """candidates sorted by score, descending; ties keep pool order."""
    return [c for _, c in sorted(zip(range(len(pool)), pool),
                                 key=lambda ic: (-scores[ic[0]], ic[0]))]


def main():
    t0 = time.time()
    notes, ann = load()
    train_notes, test_notes = split_notes(ann)
    cg = CandidateGenerator().fit(ann, train_notes)
    train_freq = Counter(int(c) for c in ann[ann.note_id.isin(train_notes)].concept_id)
    seen_forms = set(cg.form_to_concepts.keys())
    tn = sorted(test_notes); random.Random(SEED).shuffle(tn)
    val_notes = set(tn[:max(1, int(len(tn) * VAL_FRAC))])
    eval_notes = set(tn) - val_notes
    gold = build_test_mentions(notes, ann, test_notes, cg)
    ev = [m for m in gold if m.note_id in eval_notes]

    Xev, _, meta_ev = load_feats(FEAT_EVAL, "eval")
    keys, pools, seen = [], [], set()
    for (nid, st, c) in meta_ev:
        k = (nid, int(st))
        if k not in seen:
            seen.add(k); keys.append(k); pools.append([])
        pools[-1].append(int(c))

    # ---- LONO ordering -------------------------------------------------------
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa
    from sklearn.ensemble import HistGradientBoostingClassifier
    Xtr, ytr, _ = load_feats(FEAT_TRAIN, "train")
    Xlono = np.load(LONO_TRAIN, allow_pickle=True)["X"]
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                         max_depth=6, l2_regularization=1.0,
                                         random_state=SEED)
    clf.fit(Xlono, ytr)
    del Xtr, Xlono
    proba = clf.predict_proba(Xev)[:, 1]
    del Xev
    order_lono, off = {}, 0
    for k, p in zip(keys, pools):
        order_lono[k] = order_from_scores(p, proba[off:off + len(p)])
        off += len(p)
    print(f"  LONO orderings ready ({time.time()-t0:.0f}s)", flush=True)

    # ---- gated ordering ------------------------------------------------------
    gold_D = pickle.load(open(GOLD_POOLS, "rb"))
    gold_tprob = pickle.load(open(GOLD_TPROB, "rb"))
    for m in gold:
        k = (m.note_id, m.start)
        if k in gold_D["pools"]:
            m.candidates = gold_D["pools"][k]
    w_seen, w_novel = tune_gate(gold_D, gold_tprob,
                                [m for m in gold if m.note_id in val_notes],
                                train_freq, seen_forms)
    order_gated = {}
    for m in ev:
        k = (m.note_id, m.start)
        wd, wt, wl = w_seen if norm(m.span) in seen_forms else w_novel
        tp = gold_tprob.get(k, {})
        dcos, lexsc, coarse = gold_D["dcos"][k], gold_D["lexsc"][k], gold_D["coarse"][k]
        pool = list(gold_D["pools"][k])
        sc = []
        for c in pool:
            ct = coarse[c]; ct4 = ct if ct in TYPE4 else "other"
            sc.append(wd * dcos.get(c, 0.0) + wt * tp.get(ct4, 0.0)
                      + wl * lexsc.get(c, 0.0))
        order_gated[k] = order_from_scores(pool, sc)
    del gold_D
    print(f"  gated orderings ready ({time.time()-t0:.0f}s)", flush=True)

    ont = SnomedOntology(SNOMED_DIR)
    print(f"  ontology ready ({time.time()-t0:.0f}s)", flush=True)

    n_scope = Counter()
    for m in ev:
        for s in scopes_of(m, train_freq):
            n_scope[s] += 1

    rows, rank_rows = [], []
    for rname, order in (("gated", order_gated), ("supervised, LONO", order_lono)):
        cnt = {s: Counter() for s in SCOPES}
        gold_ranks = {s: [] for s in SCOPES}
        n_rank_err = Counter()
        for m in ev:
            k = (m.note_id, m.start)
            o = order.get(k)
            if not o:
                continue
            # reachability is tested against the ranker's OWN pool: the two
            # rankers' pools differ for ~14% of mentions by tie-breaking at the
            # 60-candidate cap, and mixing them would drop real errors.
            try:
                gr = o.index(m.gold_concept) + 1
            except ValueError:
                continue                      # retrieval error, not a ranking error
            if gr == 1:
                continue                      # correct
            cat = classify(ont, o[0], m.gold_concept)
            for s in scopes_of(m, train_freq):
                cnt[s][cat] += 1
                gold_ranks[s].append(gr)
                n_rank_err[s] += 1
        for s in SCOPES:
            tot = n_rank_err[s]
            row = {"ranker": rname, "scope": s, "n_mentions": n_scope[s],
                   "n_ranking_errors": tot,
                   "err_rate": round(tot / n_scope[s], 4) if n_scope[s] else None}
            for c in CATS:
                row[c] = round(cnt[s][c] / tot, 4) if tot else None
            gr_share = sum(cnt[s][c] for c in GRANULARITY) / tot if tot else None
            row["granularity_share"] = round(gr_share, 4) if gr_share is not None else None
            row["max_gain_if_fixed"] = (round(gr_share * tot / n_scope[s], 4)
                                        if gr_share is not None and n_scope[s] else None)
            rows.append(row)

            g = np.array(gold_ranks[s]) if gold_ranks[s] else np.array([0])
            rank_rows.append({
                "ranker": rname, "scope": s, "n_ranking_errors": tot,
                "gold_rank_median": int(np.median(g)) if tot else None,
                "gold_rank_p25": int(np.percentile(g, 25)) if tot else None,
                "gold_rank_p75": int(np.percentile(g, 75)) if tot else None,
                "gold_in_top3": round(float((g <= 3).mean()), 4) if tot else None,
                "gold_in_top10": round(float((g <= 10).mean()), 4) if tot else None,
                "gold_beyond_30": round(float((g > 30).mean()), 4) if tot else None})

    df = pd.DataFrame(rows)
    dfr = pd.DataFrame(rank_rows)
    out = HERE / "stage2_out"; out.mkdir(exist_ok=True)
    df.to_csv(out / "stage2_error_taxonomy.csv", index=False)
    dfr.to_csv(out / "stage2_gold_rank.csv", index=False)
    lines = [
        "# Ranking errors — ontological taxonomy and where the gold sits", "",
        f"eval = canonical split {len(ev):,} mentions · pool = weighted RRF@60 · a "
        "RANKING error is a mention whose gold is in the pool but is not ranked "
        "first, so retrieval misses are excluded by construction.", "",
        "Reachability is tested against each ranker's OWN pool. The two pools are "
        "built by different scripts and differ for a minority of mentions by "
        "tie-breaking at the 60-candidate cap; they give the same oracle ceiling "
        "to within one mention, but mixing them would silently drop real errors.", "",
        "## 1. What kind of mistake is it?", "",
        "Categories are shares of that scope's ranking errors. `granularity_share` "
        "= `too_broad` + `too_specific`, the only slice an is-a distance penalty "
        "(the proposal's Group B) could ever address. `max_gain_if_fixed` converts "
        "it to a bound on top-1 accuracy: even a PERFECT granularity fixer cannot "
        "gain more than that.", "",
        tcg.df_to_markdown(df), "",
        "## 2. Is the ranking headroom reachable?", "",
        "The oracle substitution says perfect ranking is worth a large amount. That "
        "is only attainable if the gold is near the top to begin with; if it sits "
        "at rank 30+ the headroom is nominal.", "",
        tcg.df_to_markdown(dfr), "",
    ]
    (out / "STAGE2_ERROR_TAXONOMY.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + df.to_string(index=False))
    print("\n" + dfr.to_string(index=False))
    print(f"\nwrote {out/'STAGE2_ERROR_TAXONOMY.md'} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
