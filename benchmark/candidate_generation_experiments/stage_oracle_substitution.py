"""Contribution (1): stage-wise oracle substitution, the marginal value of
perfecting ONE stage while the other two stay at their measured performance.

The end-to-end pipeline is detection -> retrieval -> ranking, and end-to-end
recall factorises exactly:

    e2e = P(detected) x P(gold in pool | detected) x P(picked gold | detected, reachable)

so each stage owns one factor and the identity can be verified numerically
(reported as `identity_check`, which must equal `e2e`).

From that, three single-stage oracle substitutions, per stratum:

  perfect detection : measured, not extrapolated. It is the gold-span pipeline
                     (the gated ranker on gold spans), which we run directly.
  perfect ranking   : exact. P(det) x P(reach|det) x 1.
  perfect retrieval : an estimate. P(det) x 1 x P(pick|det,reach), which assumes
                      that the conditional ranking accuracy measured on
                      reachable mentions carries over to the ones currently out
                      of reach. Labelled an estimate wherever it is reported.

Also reports the cumulative path to 1.0 (perfect detection+retrieval, etc.) and
the gold-span pipeline's own two-factor decomposition, so Stage 1 vs Stage 2
priority can be read off without the detection stage confounding it.

Reuses every cache built by stage0_chain_e2e.py, so this is seconds.

    KMP_DUPLICATE_LIB_OK=TRUE python stage_oracle_substitution.py --variant vanilla
"""
import argparse
import json
import pickle
import random
import sys
import time
from collections import Counter
from pathlib import Path

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
                              build_pred_mentions, build_pred_pools, iou, tune_gate)
from stage2_gated import SEED, TFMR_DIR, VAL_FRAC, pick
from stage2_reranker import TypeProba
import test_candidate_generation as tcg

STRATA = ("head", "torso", "tail")
SCOPES = ("all",) + STRATA


def rate(num, den):
    return (num / den) if den else None


def r4(x):
    return None if x is None else round(x, 4)


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
    gold_val = [m for m in gold if m.note_id in val_notes]
    gold_eval = [m for m in gold if m.note_id in eval_notes]

    gold_D = pickle.load(open(GOLD_POOLS, "rb"))
    gold_tprob = pickle.load(open(GOLD_TPROB, "rb"))
    for m in gold:
        k = (m.note_id, m.start)
        if k in gold_D["pools"]:
            m.candidates = gold_D["pools"][k]
    w_seen, w_novel = tune_gate(gold_D, gold_tprob, gold_val, train_freq, seen_forms)
    print(f"gate: seen-form {w_seen} / novel-form {w_novel}", flush=True)

    def gated(m, D, tprob):
        w = w_seen if norm(m.span) in seen_forms else w_novel
        return pick(m, D, tprob, *w)

    # ---- predicted-span side (all caches already exist) ---------------------
    pred_recs = [json.loads(l) for l in
                 open(PRED_DIR / f"stage0_pred_spans_{args.variant}.jsonl",
                      encoding="utf-8")]
    pred_m = build_pred_mentions(pred_recs, text_by_id, cg, eval_notes)
    pred_D = build_pred_pools(pred_m, cg, HERE / f".stage0_pools_{args.variant}.pkl")
    for m in pred_m:
        m.candidates = pred_D["pools"][(m.note_id, m.start)]
    stage2_reranker.TPROB_CACHE = HERE / f".stage0_typeprob_{args.variant}.pkl"
    pred_tprob = TypeProba(TFMR_DIR).proba(pred_m)
    print(f"pred side ready ({time.time()-t0:.0f}s)", flush=True)

    pred_pick = {(m.note_id, m.start): gated(m, pred_D, pred_tprob) for m in pred_m}
    pred_by_note = {}
    for m in pred_m:
        key = (m.note_id, m.start)
        pred_by_note.setdefault(m.note_id, []).append((m.start, m.end, key))

    # ---- per-gold-mention outcome accounting -------------------------------
    # counters per scope: n, detected, detected&reachable, detected&reachable&picked
    C = {s: dict(n=0, det=0, det_reach=0, det_reach_pick=0,
                 g_reach=0, g_reach_pick=0) for s in SCOPES}

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
            reach = m.gold_concept in set(pred_D["pools"][best_key])
            picked = pred_pick[best_key] == m.gold_concept
        # gold-span (perfect detection) side
        gk = (m.note_id, m.start)
        g_reach = gk in gold_D["pools"] and m.gold_concept in set(gold_D["pools"][gk])
        g_pick = gated(m, gold_D, gold_tprob) == m.gold_concept
        for sc in ("all", st):
            c = C[sc]
            c["n"] += 1
            c["det"] += detected
            c["det_reach"] += detected and reach
            c["det_reach_pick"] += detected and reach and picked
            c["g_reach"] += g_reach
            c["g_reach_pick"] += g_reach and g_pick

    rows_id, rows_sub, rows_gold = [], [], []
    for sc in SCOPES:
        c = C[sc]
        n = c["n"]
        p_det = rate(c["det"], n)
        p_reach = rate(c["det_reach"], c["det"])
        p_pick = rate(c["det_reach_pick"], c["det_reach"])
        e2e = rate(c["det_reach_pick"], n)

        # --- the factorisation identity ---
        rows_id.append({
            "scope": sc, "n_gold": n,
            "P(detected)": r4(p_det),
            "P(reach|det)": r4(p_reach),
            "P(pick|det,reach)": r4(p_pick),
            "e2e": r4(e2e),
            "identity_check": r4(None if None in (p_det, p_reach, p_pick)
                                 else p_det * p_reach * p_pick),
        })

        # --- single-stage oracle substitution ---
        perfect_det = rate(c["g_reach_pick"], n)              # measured gold-span run
        perfect_rank = None if None in (p_det, p_reach) else p_det * p_reach
        perfect_retr = None if None in (p_det, p_pick) else p_det * p_pick
        det_retr = p_pick                       # cumulative: det=1 and reach=1
        rows_sub.append({
            "scope": sc, "n_gold": n,
            "current_e2e": r4(e2e),
            "perfect_detection (measured)": r4(perfect_det),
            "d_detection": r4(None if None in (perfect_det, e2e) else perfect_det - e2e),
            "perfect_retrieval (est.)": r4(perfect_retr),
            "d_retrieval": r4(None if None in (perfect_retr, e2e) else perfect_retr - e2e),
            "perfect_ranking (exact)": r4(perfect_rank),
            "d_ranking": r4(None if None in (perfect_rank, e2e) else perfect_rank - e2e),
            "perfect_det+retr (est.)": r4(det_retr),
        })

        # --- gold-span pipeline: Stage 1 vs Stage 2 without detection ---------
        g_reach_rate = rate(c["g_reach"], n)
        g_pick_given = rate(c["g_reach_pick"], c["g_reach"])
        rows_gold.append({
            "scope": sc, "n_gold": n,
            "pool_reachability": r4(g_reach_rate),
            "P(pick|reachable)": r4(g_pick_given),
            "gold_span_top1": r4(rate(c["g_reach_pick"], n)),
            "headroom_retrieval (est.)": r4(None if None in (g_pick_given,)
                                            else g_pick_given - rate(c["g_reach_pick"], n)),
            "headroom_ranking (exact)": r4(None if None in (g_reach_rate,)
                                           else g_reach_rate - rate(c["g_reach_pick"], n)),
        })

    df_id = pd.DataFrame(rows_id)
    df_sub = pd.DataFrame(rows_sub)
    df_gold = pd.DataFrame(rows_gold)

    out = HERE / "stage2_out"; out.mkdir(exist_ok=True)
    df_id.to_csv(out / f"oracle_identity_{args.variant}.csv", index=False)
    df_sub.to_csv(out / f"oracle_substitution_{args.variant}.csv", index=False)
    df_gold.to_csv(out / f"oracle_goldspan_{args.variant}.csv", index=False)

    lines = [
        f"# Stage-wise oracle substitution — {args.variant}", "",
        f"eval notes {len(eval_notes)} · gold eval mentions {len(gold_eval)} · "
        f"pred mentions {len(pred_m)} · IoU thr {IOU_THR} · "
        f"gate seen {w_seen} / novel {w_novel}", "",
        "### 1. End-to-end factorisation (identity_check must equal e2e)", "",
        tcg.df_to_markdown(df_id), "",
        "`P(detected)` = gold span matched by a predicted span at IoU>=0.5. "
        "`P(reach|det)` = gold concept present in the matched span's RRF@60 pool. "
        "`P(pick|det,reach)` = gated ranker selects it.", "",
        "### 2. Marginal value of perfecting ONE stage (others held at measured performance)", "",
        tcg.df_to_markdown(df_sub), "",
        "`perfect_detection` is MEASURED — it is the gold-span pipeline, run directly. "
        "`perfect_ranking` is EXACT (P(det)xP(reach|det)). "
        "`perfect_retrieval` is an ESTIMATE: it assumes P(pick|det,reach) also holds for "
        "mentions whose gold is currently unreachable, which is optimistic because those "
        "mentions are harder by construction.", "",
        "### 3. Gold-span pipeline only — Stage 1 vs Stage 2 priority without detection", "",
        tcg.df_to_markdown(df_gold), "",
        "`headroom_retrieval` = what perfect retrieval would add on gold spans (estimate); "
        "`headroom_ranking` = what perfect ranking would add (exact).", "",
    ]
    (out / f"ORACLE_SUBSTITUTION_{args.variant}.md").write_text("\n".join(lines),
                                                                encoding="utf-8")
    print("\n--- factorisation ---"); print(df_id.to_string(index=False))
    print("\n--- single-stage oracles ---"); print(df_sub.to_string(index=False))
    print("\n--- gold-span only ---"); print(df_gold.to_string(index=False))
    print(f"\nwrote {out/f'ORACLE_SUBSTITUTION_{args.variant}.md'} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
