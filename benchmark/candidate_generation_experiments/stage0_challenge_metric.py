"""Score the end-to-end pipeline with the CHALLENGE's own metric.

Everything else in this dissertation is measured with internal metrics
(reachability, top-1 accuracy, end-to-end recall) and compared only against its
own oracles. That makes the numbers self-consistent but leaves a reader unable
to judge whether the system is any good, because there is no external anchor.

This script closes that gap by scoring the same predictions with the official
macro character-level IoU of the SNOMED CT Entity Linking Challenge
(`benchmark/metric.py`), which averages IoU over concept ids so that every
concept counts equally regardless of how often it occurs. Published leaderboard
scores for the three prize-winning systems are 0.4202, 0.4194 and 0.3777
(Davidson et al., 2025).

One caveat to state wherever this number is quoted. The leaderboard was
computed on the challenge's held-out private test set, whereas
this runs on the 55 evaluation notes of the public release (13 test notes are
reserved here as a tuning split). The comparison is indicative of the operating
range, not a like-for-like ranking.

Reported for both detector variants and three rankers, plus per-stratum macro
IoU restricted to head / torso / tail concepts.

    KMP_DUPLICATE_LIB_OK=TRUE python stage0_challenge_metric.py
"""
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
from data import CandidateGenerator, load, split_notes
from evaluate import freq_stratum
from metric import iou_per_class
from stage0_chain_e2e import GOLD_POOLS, GOLD_TPROB, PRED_DIR, build_pred_mentions, tune_gate
from stage2_gated import SEED, TFMR_DIR, VAL_FRAC, pick
from stage2_feature_hygiene import FEAT_TRAIN, load_feats
from stage2_lono_training import LONO_TRAIN
from stage2_reranker import TypeProba
import test_candidate_generation as tcg

VARIANTS = ["vanilla", "section_aware"]


def argmax_picks(proba, keys, pools):
    out, off = {}, 0
    for k, p in zip(keys, pools):
        n = len(p)
        if n:
            out[k] = int(p[int(np.argmax(proba[off:off + n]))])
        off += n
    return out


def main():
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

    gold_df = ann[(ann.note_id.isin(eval_notes)) & (ann.annotation_type == "test")][
        ["note_id", "start", "end", "concept_id"]].copy()
    gold_df["concept_id"] = gold_df.concept_id.astype(int)
    print(f"gold rows {len(gold_df)} over {len(eval_notes)} eval notes", flush=True)

    # concept -> stratum, for per-stratum macro IoU
    strat_of_concept = {int(c): freq_stratum(int(c), train_freq)
                        for c in set(gold_df.concept_id)}

    # ---- rankers -------------------------------------------------------------
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa
    from sklearn.ensemble import HistGradientBoostingClassifier
    Xtr, ytr, _ = load_feats(FEAT_TRAIN, "train")
    models = {}
    for name, X in (("supervised, standard", Xtr),
                    ("supervised, LONO", np.load(LONO_TRAIN, allow_pickle=True)["X"])):
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                             max_depth=6, l2_regularization=1.0,
                                             random_state=SEED)
        clf.fit(X, ytr)
        models[name] = clf
        print(f"  trained [{name}] ({time.time()-t0:.0f}s)", flush=True)
    del Xtr

    gold_D = pickle.load(open(GOLD_POOLS, "rb"))
    gold_tprob = pickle.load(open(GOLD_TPROB, "rb"))
    from data import build_test_mentions
    gold_m = build_test_mentions(notes, ann, test_notes, cg)
    for m in gold_m:
        k = (m.note_id, m.start)
        if k in gold_D["pools"]:
            m.candidates = gold_D["pools"][k]
    w_seen, w_novel = tune_gate(gold_D, gold_tprob,
                                [m for m in gold_m if m.note_id in val_notes],
                                train_freq, seen_forms)
    del gold_D, gold_m

    rows = []
    for variant in VARIANTS:
        pred_recs = [json.loads(l) for l in open(
            PRED_DIR / f"stage0_pred_spans_{variant}.jsonl", encoding="utf-8")]
        pred_m = build_pred_mentions(pred_recs, text_by_id, cg, eval_notes)
        span_of = {(m.note_id, m.start): (m.start, m.end) for m in pred_m}

        z = np.load(HERE / f".stage0_feat_{variant}.npz", allow_pickle=True)
        Xp = z["X"]
        p_keys = [(k[0], int(k[1])) for k in z["keys"]]
        p_pools = [list(p) for p in z["pools"]]

        picks = {n: argmax_picks(clf.predict_proba(Xp)[:, 1], p_keys, p_pools)
                 for n, clf in models.items()}
        del Xp

        pred_D = pickle.load(open(HERE / f".stage0_pools_{variant}.pkl", "rb"))
        stage2_reranker.TPROB_CACHE = HERE / f".stage0_typeprob_{variant}.pkl"
        pred_tprob = TypeProba(TFMR_DIR).proba(pred_m)
        picks["gated"] = {(m.note_id, m.start): pick(
            m, pred_D, pred_tprob,
            *(w_seen if norm(m.span) in seen_forms else w_novel)) for m in pred_m}
        del pred_D
        print(f"  [{variant}] {len(pred_m)} predicted spans ({time.time()-t0:.0f}s)",
              flush=True)

        for rname in ("gated", "supervised, standard", "supervised, LONO"):
            pk = picks[rname]
            pred_df = pd.DataFrame(
                [{"note_id": k[0], "start": span_of[k][0], "end": span_of[k][1],
                  "concept_id": int(c)} for k, c in pk.items() if k in span_of])
            ious = iou_per_class(pred_df, gold_df)
            overall = float(np.mean(list(ious.values())))
            row = {"variant": variant, "ranker": rname,
                   "n_pred_spans": len(pred_df), "n_concepts_scored": len(ious),
                   "macro_char_iou": round(overall, 4)}
            for s in ("head", "torso", "tail"):
                sub = [v for c, v in ious.items() if strat_of_concept.get(c) == s]
                row[f"macro_iou_{s}"] = round(float(np.mean(sub)), 4) if sub else None
            rows.append(row)
            print(f"    {rname}: macro char-IoU {overall:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    out = HERE / "stage2_out"; out.mkdir(exist_ok=True)
    df.to_csv(out / "challenge_metric.csv", index=False)
    lines = [
        "# End-to-end scored with the challenge's own metric", "",
        f"{len(eval_notes)} evaluation notes · macro character-level IoU averaged "
        "over concept ids, per `benchmark/metric.py`. Published leaderboard scores "
        "for the three prize-winning systems: 0.4202, 0.4194, 0.3777 (Davidson et "
        "al., 2025).", "",
        "**Comparability caveat.** The leaderboard used the challenge's held-out "
        "private test set; this uses the 55 evaluation notes of the public release, "
        "with 13 test notes reserved as a tuning split. Treat the comparison as "
        "indicative of the operating range, not as a like-for-like ranking.", "",
        tcg.df_to_markdown(df), "",
        "Per-stratum columns restrict the macro average to concepts of that "
        "training-frequency stratum, so they are directly comparable with the "
        "stratified recall figures reported elsewhere.", "",
    ]
    (out / "CHALLENGE_METRIC.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + df.to_string(index=False))
    print(f"\nwrote {out/'CHALLENGE_METRIC.md'} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
