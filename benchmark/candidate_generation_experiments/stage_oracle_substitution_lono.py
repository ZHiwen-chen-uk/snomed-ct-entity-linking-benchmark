"""Stage-wise oracle substitution, recomputed for the LONO reranker.

`stage_oracle_substitution.py` established the exact factorisation

    e2e = P(detected) x P(gold in pool | detected) x P(gold picked | det, reach)

and, with the GATED ranker, the stage priority `ranking > detection >> retrieval`
at every stratum. Experiment 1b replaced the ranking stage with a much stronger
model, so the priority has to be recomputed: improving one stage changes which
stage is the bottleneck, which is precisely the "bottleneck migration" claim the
thesis makes, and this script measures it rather than asserting it.

Both rankers are reported side by side so the migration is visible in one table.

This version needs no retrievers: the predicted-span features and pools are
cached by `stage0_chain_e2e_lono.py` in `.stage0_feat_<variant>.npz`, and the
gold-span pools are recovered from the eval feature meta, so nothing is
re-retrieved and neither SapBERT nor the SNOMED ontology is loaded.

    KMP_DUPLICATE_LIB_OK=TRUE python stage_oracle_substitution_lono.py --variant vanilla
"""
import argparse
import json
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

import stage2_reranker
from candidate_booster import norm
from data import CandidateGenerator, build_test_mentions, load, split_notes
from evaluate import freq_stratum
from stage0_chain_e2e import (GOLD_POOLS, GOLD_TPROB, IOU_THR, PRED_DIR,
                              build_pred_mentions, iou, tune_gate)
from stage2_gated import SEED, TFMR_DIR, VAL_FRAC, pick
from stage2_feature_hygiene import FEAT_EVAL, FEAT_TRAIN, load_feats
from stage2_lono_training import LONO_TRAIN
from stage2_reranker import TypeProba
import test_candidate_generation as tcg

STRATA = ("head", "torso", "tail")
SCOPES = ("all",) + STRATA


def rate(num, den):
    return (num / den) if den else None


def r4(x):
    return None if x is None else round(x, 4)


def group_pools(meta):
    """meta rows -> (keys in order, pool per key). Rows of a mention are contiguous."""
    keys, pools, seen = [], [], set()
    for (nid, st, c) in meta:
        k = (nid, int(st))
        if k not in seen:
            seen.add(k); keys.append(k); pools.append([])
        pools[-1].append(int(c))
    return keys, pools


def argmax_picks(proba, keys, pools):
    out, off = {}, 0
    for k, p in zip(keys, pools):
        n = len(p)
        if n:
            out[k] = int(p[int(np.argmax(proba[off:off + n]))])
        off += n
    assert off == len(proba), f"segment mismatch {off} vs {len(proba)}"
    return out


def account(gold_eval, train_freq, pred_by_note, pred_pool, pred_pick,
            gold_pool, gold_pick):
    """Per-gold-mention outcome counters, per scope."""
    C = {s: dict(n=0, det=0, det_reach=0, det_reach_pick=0, g_reach=0,
                 g_reach_pick=0) for s in SCOPES}
    for m in gold_eval:
        st = freq_stratum(m.gold_concept, train_freq)
        best_iou, best_key = 0.0, None
        for (ps, pe, key) in pred_by_note.get(m.note_id, []):
            v = iou((m.start, m.end), (ps, pe))
            if v > best_iou:
                best_iou, best_key = v, key
        detected = best_iou >= IOU_THR
        reach = picked = False
        if detected:
            reach = m.gold_concept in pred_pool.get(best_key, ())
            picked = pred_pick.get(best_key) == m.gold_concept
        gk = (m.note_id, m.start)
        g_reach = m.gold_concept in gold_pool.get(gk, ())
        g_pick = gold_pick.get(gk) == m.gold_concept
        for sc in ("all", st):
            c = C[sc]
            c["n"] += 1
            c["det"] += detected
            c["det_reach"] += detected and reach
            c["det_reach_pick"] += detected and reach and picked
            c["g_reach"] += g_reach
            c["g_reach_pick"] += g_reach and g_pick
    return C


def tables(C, ranker):
    rows_id, rows_sub, rows_gold = [], [], []
    for sc in SCOPES:
        c = C[sc]; n = c["n"]
        p_det = rate(c["det"], n)
        p_reach = rate(c["det_reach"], c["det"])
        p_pick = rate(c["det_reach_pick"], c["det_reach"])
        e2e = rate(c["det_reach_pick"], n)
        rows_id.append({"ranker": ranker, "scope": sc, "n_gold": n,
                        "P(detected)": r4(p_det), "P(reach|det)": r4(p_reach),
                        "P(pick|det,reach)": r4(p_pick), "e2e": r4(e2e),
                        "identity_check": r4(None if None in (p_det, p_reach, p_pick)
                                             else p_det * p_reach * p_pick)})
        perfect_det = rate(c["g_reach_pick"], n)
        perfect_rank = None if None in (p_det, p_reach) else p_det * p_reach
        perfect_retr = None if None in (p_det, p_pick) else p_det * p_pick
        rows_sub.append({
            "ranker": ranker, "scope": sc, "n_gold": n, "current_e2e": r4(e2e),
            "perfect_detection (measured)": r4(perfect_det),
            "d_detection": r4(None if None in (perfect_det, e2e) else perfect_det - e2e),
            "perfect_retrieval (est.)": r4(perfect_retr),
            "d_retrieval": r4(None if None in (perfect_retr, e2e) else perfect_retr - e2e),
            "perfect_ranking (exact)": r4(perfect_rank),
            "d_ranking": r4(None if None in (perfect_rank, e2e) else perfect_rank - e2e),
            "perfect_det+retr (est.)": r4(p_pick)})
        g_reach_rate = rate(c["g_reach"], n)
        g_top1 = rate(c["g_reach_pick"], n)
        g_pick_given = rate(c["g_reach_pick"], c["g_reach"])
        rows_gold.append({
            "ranker": ranker, "scope": sc, "n_gold": n,
            "pool_reachability": r4(g_reach_rate),
            "P(pick|reachable)": r4(g_pick_given),
            "gold_span_top1": r4(g_top1),
            "headroom_retrieval (est.)": r4(None if None in (g_pick_given, g_top1)
                                            else g_pick_given - g_top1),
            "headroom_ranking (exact)": r4(None if None in (g_reach_rate, g_top1)
                                           else g_reach_rate - g_top1)})
    return rows_id, rows_sub, rows_gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="vanilla",
                    choices=["vanilla", "section_aware"])
    args = ap.parse_args()
    t0 = time.time()

    notes, ann = load()
    train_notes, test_notes = split_notes(ann)
    text_by_id = dict(zip(notes.note_id, notes.text))
    cg = CandidateGenerator().fit(ann, train_notes)
    train_freq = Counter(int(c) for c in ann[ann.note_id.isin(train_notes)].concept_id)
    seen_forms = set(cg.form_to_concepts.keys())

    tn = sorted(test_notes); random.Random(SEED).shuffle(tn)
    val_notes = set(tn[:max(1, int(len(tn) * VAL_FRAC))])
    eval_notes = set(tn) - val_notes
    gold = build_test_mentions(notes, ann, test_notes, cg)
    gold_eval = [m for m in gold if m.note_id in eval_notes]

    # ---- gold-span side: pools + both rankers' picks ------------------------
    Xev, _, meta_ev = load_feats(FEAT_EVAL, "eval")
    ev_keys, ev_pools = group_pools(meta_ev)
    gold_pool = {k: set(p) for k, p in zip(ev_keys, ev_pools)}

    from sklearn.experimental import enable_hist_gradient_boosting  # noqa
    from sklearn.ensemble import HistGradientBoostingClassifier
    Xtr, ytr, _ = load_feats(FEAT_TRAIN, "train")
    Xlono = np.load(LONO_TRAIN, allow_pickle=True)["X"]
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                         max_depth=6, l2_regularization=1.0,
                                         random_state=SEED)
    clf.fit(Xlono, ytr)
    print(f"  LONO reranker trained ({time.time()-t0:.0f}s)", flush=True)
    del Xtr, Xlono
    gold_pick_lono = argmax_picks(clf.predict_proba(Xev)[:, 1], ev_keys, ev_pools)
    del Xev

    gold_D = pickle.load(open(GOLD_POOLS, "rb"))
    gold_tprob = pickle.load(open(GOLD_TPROB, "rb"))
    for m in gold:
        k = (m.note_id, m.start)
        if k in gold_D["pools"]:
            m.candidates = gold_D["pools"][k]
    w_seen, w_novel = tune_gate(gold_D, gold_tprob,
                                [m for m in gold if m.note_id in val_notes],
                                train_freq, seen_forms)

    def gated(m, D, tprob):
        return pick(m, D, tprob, *(w_seen if norm(m.span) in seen_forms else w_novel))

    gold_pick_gated = {(m.note_id, m.start): gated(m, gold_D, gold_tprob)
                       for m in gold_eval}
    del gold_D

    # ---- predicted-span side ------------------------------------------------
    pred_recs = [json.loads(l) for l in open(
        PRED_DIR / f"stage0_pred_spans_{args.variant}.jsonl", encoding="utf-8")]
    pred_m = build_pred_mentions(pred_recs, text_by_id, cg, eval_notes)
    z = np.load(HERE / f".stage0_feat_{args.variant}.npz", allow_pickle=True)
    Xp = z["X"]
    p_keys = [(k[0], int(k[1])) for k in z["keys"]]
    p_pools = [list(p) for p in z["pools"]]
    pred_pool = {k: set(p) for k, p in zip(p_keys, p_pools)}
    pred_pick_lono = argmax_picks(clf.predict_proba(Xp)[:, 1], p_keys, p_pools)
    del Xp
    print(f"  pred side ready ({time.time()-t0:.0f}s)", flush=True)

    pred_D = pickle.load(open(HERE / f".stage0_pools_{args.variant}.pkl", "rb"))
    stage2_reranker.TPROB_CACHE = HERE / f".stage0_typeprob_{args.variant}.pkl"
    pred_tprob = TypeProba(TFMR_DIR).proba(pred_m)
    pred_pick_gated = {(m.note_id, m.start): gated(m, pred_D, pred_tprob)
                       for m in pred_m}
    del pred_D

    pred_by_note = {}
    for m in pred_m:
        pred_by_note.setdefault(m.note_id, []).append(
            (m.start, m.end, (m.note_id, m.start)))

    ids, subs, golds = [], [], []
    for ranker, gp, pp in (("gated", gold_pick_gated, pred_pick_gated),
                           ("supervised, LONO", gold_pick_lono, pred_pick_lono)):
        C = account(gold_eval, train_freq, pred_by_note, pred_pool, pp,
                    gold_pool, gp)
        a, b, c = tables(C, ranker)
        ids += a; subs += b; golds += c

    df_id, df_sub, df_gold = pd.DataFrame(ids), pd.DataFrame(subs), pd.DataFrame(golds)
    out = HERE / "stage2_out"; out.mkdir(exist_ok=True)
    v = args.variant
    df_id.to_csv(out / f"oracle_identity_lono_{v}.csv", index=False)
    df_sub.to_csv(out / f"oracle_substitution_lono_{v}.csv", index=False)
    df_gold.to_csv(out / f"oracle_goldspan_lono_{v}.csv", index=False)

    lines = [
        f"# Stage-wise oracle substitution — {v}, gated vs the LONO reranker", "",
        f"eval notes {len(eval_notes)} · gold eval mentions {len(gold_eval)} · "
        f"pred mentions {len(pred_m)} · IoU thr {IOU_THR} · "
        f"gate seen {w_seen} / novel {w_novel}", "",
        "Detection and retrieval are IDENTICAL across the two rows — only the "
        "ranking stage changes — so any change in the stage priority is caused "
        "solely by improving Stage 2.", "",
        "### 1. End-to-end factorisation (identity_check must equal e2e)", "",
        tcg.df_to_markdown(df_id), "",
        "### 2. Marginal value of perfecting ONE stage", "",
        tcg.df_to_markdown(df_sub), "",
        "`perfect_detection` is MEASURED (the gold-span pipeline). "
        "`perfect_ranking` is EXACT. `perfect_retrieval` is an optimistic ESTIMATE: "
        "it assumes currently-unreachable golds would be ranked as well as reachable "
        "ones, which they would not.", "",
        "### 3. Gold-span pipeline only — Stage 1 vs Stage 2 without detection", "",
        tcg.df_to_markdown(df_gold), "",
    ]
    (out / f"ORACLE_SUBSTITUTION_LONO_{v}.md").write_text("\n".join(lines),
                                                          encoding="utf-8")
    print("\n--- factorisation ---"); print(df_id.to_string(index=False))
    print("\n--- single-stage oracles ---"); print(df_sub.to_string(index=False))
    print("\n--- gold-span only ---"); print(df_gold.to_string(index=False))
    print(f"\nwrote {out/f'ORACLE_SUBSTITUTION_LONO_{v}.md'} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
