"""Stage-0 -> Stage-1/2 chaining, re-run with the LEARNED rerankers.

`stage0_chain_e2e.py` produced the thesis's end-to-end numbers using the GATED
ranker (all 0.5650 / tail 0.3506 for vanilla). Experiment 1b then produced a
much stronger Stage-2 model, the LONO-trained reranker (all 0.8689 / head
0.9520 / torso 0.8314 / tail 0.5939 on gold spans), so the end-to-end story has
to be recomputed with it, otherwise the thesis reports an end-to-end pipeline
that deliberately omits its own best ranker.

Three rankers are scored on the SAME predicted spans so the comparison is
controlled:

    gated                 reproduces the published row (harness check)
    supervised, standard  the blind-spot model (features from the full index)
    supervised, LONO      features recomputed leave-one-note-out

The expensive part of the original chain, SapBERT dense retrieval over the
predicted spans, is NOT repeated: `.stage0_pools_<variant>.pkl` stores `dcos`,
built as `dict(dense_pairs)`, and dict preserves insertion order, so the dense
ranking is recovered exactly from its keys. Lexical/char scores and the synonym
set were not cached, so those retrievers are rebuilt; that is the runtime cost.

    KMP_DUPLICATE_LIB_OK=TRUE python stage0_chain_e2e_lono.py
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
from candidate_booster import SnomedCandidateBooster, norm
from data import CandidateGenerator, build_test_mentions, load, split_notes
from evaluate import freq_stratum
from ontology import SnomedOntology
from retrieval_fusion import rrf_fuse
from stage0_chain_e2e import IOU_THR, PRED_DIR, build_pred_mentions, iou, tune_gate
from stage2_gated import CAP, K, RRF_K, RRF_WEIGHTS, SEED, SNOMED_DIR, TFMR_DIR, VAL_FRAC, pick
from stage2_reranker import TypeProba, features, retriever_scores
from stage2_feature_hygiene import FEAT_EVAL, FEAT_TRAIN, FNAMES, load_feats
from stage2_lono_training import LONO_TRAIN, POOL
import test_candidate_generation as tcg

GOLD_POOLS = HERE / ".stage2_pools_full.pkl"
GOLD_TPROB = HERE / ".stage2_typeprob_cache.pkl"
VARIANTS = ["vanilla", "section_aware"]
STRATA = ("head", "torso", "tail")


def train_models(t0):
    """Train the standard and LONO rerankers on the cached 13-feature matrices."""
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa
    from sklearn.ensemble import HistGradientBoostingClassifier

    Xtr, ytr, _ = load_feats(FEAT_TRAIN, "train")
    z = np.load(LONO_TRAIN, allow_pickle=True)
    Xlono = z["X"]
    assert Xlono.shape == Xtr.shape, "LONO matrix shape mismatch"

    models = {}
    for name, X in (("supervised, standard", Xtr), ("supervised, LONO", Xlono)):
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                             max_depth=6, l2_regularization=1.0,
                                             random_state=SEED)
        clf.fit(X, ytr)
        models[name] = clf
        print(f"  trained [{name}] ({time.time()-t0:.0f}s)", flush=True)
    return models


def build_feature_rows(mentions, booster, char, cg, ontology, tprob, pred_D, t0, tag):
    """13 features per (mention, candidate), mirroring stage2_reranker_full.feats.

    The candidate set is the CACHED pool, not a recomputed one, so that all three
    rankers score an identical candidate set and the comparison is controlled.
    (Recomputing `rrf_fuse` here reproduces the cached pool for ~78% of mentions;
    the rest differ only by tie-breaking at the 60-candidate cap, 1-2 candidates
    swapped. `unsupported` below counts candidates that the recomputed retriever
    lists do not cover, which therefore receive zero retrieval features.)
    """
    dcos, cached_pools = pred_D["dcos"], pred_D["pools"]
    X, keys, pools = [], [], []
    n_cand = unsupported = reproduced = 0
    for i, m in enumerate(mentions):
        key = (m.note_id, m.start)
        dense_pairs = list(dcos[key].items())          # dict preserves dense order
        lists, ranks, lex_s, ch_s, dscore, synset = retriever_scores(
            m, booster, char, dense_pairs, K)
        pool = list(cached_pools[key])
        reproduced += (rrf_fuse(lists, k=RRF_K, cap=CAP, weights=RRF_WEIGHTS) == pool)
        covered = set().union(*[set(v) for v in lists.values()]) if lists else set()
        for c in pool:
            n_cand += 1
            unsupported += c not in covered
            X.append(features(m, c, ranks, lex_s, ch_s, dscore, synset,
                              cg, booster, ontology, tprob))
        keys.append(key)
        pools.append(pool)
        if (i + 1) % 2500 == 0:
            print(f"    features[{tag}] {i+1}/{len(mentions)} ({time.time()-t0:.0f}s)",
                  flush=True)
    print(f"  [{tag}] pool exactly reproduced for {reproduced}/{len(mentions)} mentions; "
          f"{unsupported}/{n_cand} cached candidates unsupported by the recomputed "
          f"retriever lists ({100*unsupported/max(n_cand,1):.2f}%)", flush=True)
    return np.asarray(X, dtype=np.float32), keys, pools


def picks_from_model(clf, X, keys, pools):
    """argmax candidate per mention under `clf`."""
    proba = clf.predict_proba(X)[:, 1]
    out, off = {}, 0
    for key, pool in zip(keys, pools):
        n = len(pool)
        if n:
            seg = proba[off:off + n]
            out[key] = int(pool[int(np.argmax(seg))])
        off += n
    assert off == len(proba), f"segment mismatch {off} vs {len(proba)}"
    return out


def strat_table(gold_eval, pred_by_note, train_freq, oracle_line):
    """detection recall / e2e / linking|detected / detection cost, per stratum."""
    det = {s: [0, 0] for s in STRATA}
    e2e = {s: [0, 0] for s in STRATA}
    det_all, e2e_all = [0, 0], [0, 0]
    for m in gold_eval:
        st = freq_stratum(m.gold_concept, train_freq)
        best_iou, best_cpt = 0.0, None
        for (ps, pe, pc) in pred_by_note.get(m.note_id, []):
            v = iou((m.start, m.end), (ps, pe))
            if v > best_iou:
                best_iou, best_cpt = v, pc
        detected = best_iou >= IOU_THR
        correct = detected and best_cpt == m.gold_concept
        det[st][0] += detected; det[st][1] += 1
        e2e[st][0] += correct; e2e[st][1] += 1
        det_all[0] += detected; det_all[1] += 1
        e2e_all[0] += correct; e2e_all[1] += 1

    def r(p):
        return round(p[0] / p[1], 4) if p[1] else None

    rows = []
    for scope, dd, ee in [("all", det_all, e2e_all)] + [(s, det[s], e2e[s]) for s in STRATA]:
        orc, e2e_r = oracle_line[scope], r(ee)
        rows.append({"scope": scope, "n_gold": ee[1], "oracle_span_link_acc": orc,
                     "detection_recall@IoU": r(dd), "e2e_recall": e2e_r,
                     "linking|detected": round(ee[0] / dd[0], 4) if dd[0] else None,
                     "detection_cost": round(orc - e2e_r, 4)
                     if (orc is not None and e2e_r is not None) else None})
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_notes", type=int, default=0)   # 0 = all eval notes
    args = ap.parse_args()
    sfx = f"_smoke{args.max_notes}" if args.max_notes else ""
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
    if args.max_notes:
        eval_notes = set(sorted(eval_notes)[:args.max_notes])

    gold = build_test_mentions(notes, ann, test_notes, cg)
    gold_val = [m for m in gold if m.note_id in val_notes]
    gold_eval = [m for m in gold if m.note_id in eval_notes]
    print(f"gold eval mentions {len(gold_eval)} ({time.time()-t0:.0f}s)", flush=True)

    print("training rerankers...", flush=True)
    models = train_models(t0)

    # ---- oracle-span (gold-span) lines for every ranker ----------------------
    gold_D = pickle.load(open(GOLD_POOLS, "rb"))
    gold_tprob = pickle.load(open(GOLD_TPROB, "rb"))
    for m in gold:
        k = (m.note_id, m.start)
        if k in gold_D["pools"]:
            m.candidates = gold_D["pools"][k]
    w_seen, w_novel = tune_gate(gold_D, gold_tprob, gold_val, train_freq, seen_forms)
    print(f"gate: seen-form {w_seen}  novel-form {w_novel}", flush=True)

    def gated(m, D, tprob):
        return pick(m, D, tprob, *(w_seen if norm(m.span) in seen_forms else w_novel))

    def line(picks, subset):
        by = {s: [0, 0] for s in STRATA}; tot = [0, 0]
        for m in subset:
            ok = picks.get((m.note_id, m.start)) == m.gold_concept
            tot[0] += ok; tot[1] += 1
            b = by[freq_stratum(m.gold_concept, train_freq)]; b[0] += ok; b[1] += 1
        f = lambda p: round(p[0] / p[1], 4) if p[1] else None
        return {"all": f(tot), **{s: f(by[s]) for s in by}}

    oracle_lines = {"gated": line({(m.note_id, m.start): gated(m, gold_D, gold_tprob)
                                   for m in gold_eval}, gold_eval)}

    Xev, _, meta_ev = load_feats(FEAT_EVAL, "eval")
    ev_keys, ev_pools = [], []
    seen = set()
    for (nid, st, c) in meta_ev:
        k = (nid, int(st))
        if k not in seen:
            seen.add(k); ev_keys.append(k); ev_pools.append([])
        ev_pools[-1].append(int(c))
    for name, clf in models.items():
        oracle_lines[name] = line(picks_from_model(clf, Xev, ev_keys, ev_pools), gold_eval)
    for name, ln in oracle_lines.items():
        print(f"  oracle-span [{name}]: {ln}", flush=True)
    del Xev

    print("building retrievers for predicted spans...", flush=True)
    ontology = SnomedOntology(SNOMED_DIR)
    booster = SnomedCandidateBooster(cg, ontology, SNOMED_DIR, mode="boosted",
                                     max_candidates=60, expand_k=40)
    char = tcg.CharNgramRetriever(booster)
    print(f"  retrievers ready ({time.time()-t0:.0f}s)", flush=True)

    out = HERE / "stage0_ner" / "result"
    summary = []
    for variant in VARIANTS:
        print(f"\n=== {variant} ===", flush=True)
        pred_recs = [json.loads(l) for l in open(
            PRED_DIR / f"stage0_pred_spans_{variant}.jsonl", encoding="utf-8")]
        pred_m = build_pred_mentions(pred_recs, text_by_id, cg, eval_notes)
        pred_D = pickle.load(open(HERE / f".stage0_pools_{variant}.pkl", "rb"))
        print(f"pred mentions {len(pred_m)} ({time.time()-t0:.0f}s)", flush=True)

        stage2_reranker.TPROB_CACHE = HERE / f".stage0_typeprob_{variant}.pkl"
        pred_tprob = TypeProba(TFMR_DIR).proba(pred_m)

        fcache = HERE / f".stage0_feat_{variant}{sfx}.npz"
        if fcache.exists():
            z = np.load(fcache, allow_pickle=True)
            Xp, keys, pools = z["X"], [tuple(k) for k in z["keys"]], list(z["pools"])
            keys = [(k[0], int(k[1])) for k in keys]
            print(f"  features[{variant}] loaded from cache {Xp.shape}", flush=True)
        else:
            Xp, keys, pools = build_feature_rows(pred_m, booster, char, cg, ontology,
                                                 pred_tprob, pred_D, t0, variant)
            np.savez_compressed(fcache, X=Xp,
                                keys=np.array(keys, dtype=object),
                                pools=np.array(pools, dtype=object))
            print(f"  features[{variant}] {Xp.shape} ({time.time()-t0:.0f}s)", flush=True)

        # every ranker must see the identical candidate set
        assert all(list(pred_D["pools"][k]) == list(p) for k, p in zip(keys, pools)), \
            "reranker pool differs from the pool `gated` scores"

        ranker_picks = {"gated": {(m.note_id, m.start): gated(m, pred_D, pred_tprob)
                                  for m in pred_m}}
        for name, clf in models.items():
            ranker_picks[name] = picks_from_model(clf, Xp, keys, pools)

        for name, picks in ranker_picks.items():
            pred_by_note = {}
            for m in pred_m:
                k = (m.note_id, m.start)
                pred_by_note.setdefault(m.note_id, []).append((m.start, m.end, picks.get(k)))
            rows = strat_table(gold_eval, pred_by_note, train_freq, oracle_lines[name])
            for r_ in rows:
                summary.append({"variant": variant, "ranker": name, **r_})
            e = {r_["scope"]: r_["e2e_recall"] for r_ in rows}
            print(f"  [{name}] e2e all={e['all']} head={e['head']} "
                  f"torso={e['torso']} tail={e['tail']}", flush=True)
        del pred_D, Xp

    df = pd.DataFrame(summary)
    df.to_csv(out / f"STAGE0_E2E_LONO{sfx}.csv", index=False)
    lines = ["# Stage-0 end-to-end, re-run with the learned rerankers", "",
             f"eval notes {len(eval_notes)} · gold eval mentions {len(gold_eval)} · "
             f"IoU thr {IOU_THR} · gate seen-form {w_seen} / novel-form {w_novel}", "",
             "`gated` reproduces the published end-to-end row and is the harness "
             "check. `supervised, standard` is the in-index-blind-spot model. "
             "`supervised, LONO` recomputes the four train-derived features "
             "leave-one-note-out (experiment 1b).", "",
             "oracle_span = ranker on GOLD spans (detection = 1); e2e = detected at "
             "IoU>=0.5 AND linked correctly; detection_cost = oracle_span - e2e.", ""]
    for variant in VARIANTS:
        lines += [f"## {variant}", ""]
        for name in ["gated", "supervised, standard", "supervised, LONO"]:
            sub = df[(df.variant == variant) & (df.ranker == name)].drop(
                columns=["variant", "ranker"])
            lines += [f"**{name}**", "", tcg.df_to_markdown(sub), ""]
    (out / f"STAGE0_E2E_LONO{sfx}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + df.to_string(index=False))
    print(f"\nDONE {time.time()-t0:.0f}s -> {out}/STAGE0_E2E_LONO{sfx}.md", flush=True)


if __name__ == "__main__":
    main()
