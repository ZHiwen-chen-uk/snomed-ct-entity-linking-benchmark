"""Stage-0 -> Stage-1/2 chaining: END-TO-END accuracy on PREDICTED spans vs the
ORACLE-span (gold-span) pipeline, so the gap = the cost of detection, stratified
head/torso/tail.

Detected spans (from stage0_ner/result/stage0_pred_spans_<variant>.jsonl) are
wrapped as data.Mention (span = note[start:end], with context + section), pushed
through the SAME candidate generation (dense + weighted RRF) and the SAME gated
ranker (seen-form->type, novel-form->dense+type) used in stage2_gated.py. The
gate weights are tuned on the GOLD-span validation split (the only legitimate
training-time resource) and applied to the predicted-span eval split.

Metric (mention-level, eval split notes):
  detection_recall(stratum) = |gold matched by a pred span at IoU>=THR| / |gold|
  e2e_recall(stratum)       = |gold detected AND its best pred span links to the
                               correct concept| / |gold|
  linking|detected          = e2e_recall / detection_recall
  oracle-span link acc       = gated ranker on GOLD spans (detection_recall == 1)
  cost of detection          = oracle-span link acc - e2e_recall

  python stage0_chain_e2e.py --variant vanilla        # then --variant section_aware
  python stage0_chain_e2e.py --variant vanilla --max_notes 4   # quick validation
"""
import argparse
import copy
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

import stage2_reranker                                   # for TPROB_CACHE monkeypatch
from candidate_booster import SnomedCandidateBooster, norm
from data import (CandidateGenerator, CONTEXT_WINDOW, Mention, build_test_mentions,
                  load, section_at, section_headers, split_notes)
from dense_retriever import DenseRetriever
from evaluate import freq_stratum
from ontology import SnomedOntology
from retrieval_fusion import exact_hits, rrf_fuse
from stage2_gated import (K, CAP, RRF_K, RRF_WEIGHTS, SNOMED_DIR, TFMR_DIR, SEED,
                          VAL_FRAC, pick)
from stage2_reranker import TypeProba
import test_candidate_generation as tcg

PRED_DIR = HERE / "stage0_ner" / "result"
IOU_THR = 0.5
GOLD_POOLS = HERE / ".stage2_pools_full.pkl"
GOLD_TPROB = HERE / ".stage2_typeprob_cache.pkl"


def iou(a, b):
    s = max(a[0], b[0]); e = min(a[1], b[1])
    inter = max(0, e - s)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union else 0.0


def build_pred_mentions(pred_recs, text_by_id, cg, note_filter):
    """Wrap detected spans as Mention (span=text[s:e], + context + section)."""
    hdr_cache = {}
    mentions = []
    for rec in pred_recs:
        nid = rec["note_id"]
        if nid not in note_filter:
            continue
        text = str(text_by_id.get(nid, ""))
        if nid not in hdr_cache:
            hdr_cache[nid] = section_headers(text)
        for s, e in rec["pred_spans"]:
            s, e = int(s), int(e)
            if e <= s or e > len(text):
                continue
            lo, hi = max(0, s - CONTEXT_WINDOW), min(len(text), e + CONTEXT_WINDOW)
            mentions.append(Mention(
                note_id=nid, start=s, end=e, span=text[s:e], gold_concept=-1,
                context=text[lo:hi], section=section_at(hdr_cache[nid], s),
                candidates=cg.generate(text[s:e])))
    return mentions


def build_pred_pools(mentions, cg, cache_path):
    if cache_path.exists():
        print(f"loading cached pred pools {cache_path.name}...", flush=True)
        return pickle.load(open(cache_path, "rb"))
    print("building pred pools (dense+RRF, first time)...", flush=True)
    ontology = SnomedOntology(SNOMED_DIR)
    booster = SnomedCandidateBooster(cg, ontology, SNOMED_DIR, mode="boosted",
                                     max_candidates=60, expand_k=40)
    char = tcg.CharNgramRetriever(booster)
    dense = DenseRetriever()
    spans = [m.span for m in mentions]
    dmap = {}
    for s in range(0, len(spans), 1000):
        for j, r in enumerate(dense.retrieve_batch(spans[s:s + 1000], k=K)):
            dmap[s + j] = r
        print(f"  dense {min(s+1000,len(spans))}/{len(spans)}", flush=True)
    pools, dcos, lexsc, coarse = {}, {}, {}, {}
    for i, m in enumerate(mentions):
        key = (m.note_id, m.start)
        dense_pairs = dmap[i]
        lex, lex_s = booster.retrieve([m.span], K)
        ch, _ = char.retrieve(m.span, K)
        lists = {"train": list(m.candidates), "synonym": exact_hits(booster, m.span),
                 "lexical": lex, "char": ch, "dense": [c for c, _ in dense_pairs]}
        pools[key] = rrf_fuse(lists, k=RRF_K, cap=CAP, weights=RRF_WEIGHTS)
        dcos[key] = dict(dense_pairs)
        lexsc[key] = dict(lex_s)
        coarse[key] = {c: ontology.coarse_type(c) for c in pools[key]}
        if (i + 1) % 5000 == 0:
            print(f"  pooled {i+1}/{len(mentions)}", flush=True)
    data = {"pools": pools, "dcos": dcos, "lexsc": lexsc, "coarse": coarse}
    pickle.dump(data, open(cache_path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    return data


def tune_gate(gold_D, gold_tprob, val, train_freq, seen_forms):
    grid = [(wd, wt, wl) for wd in (0, 0.5, 1, 2) for wt in (0, 0.5, 1, 2)
            for wl in (0, 0.05, 0.1) if (wd or wt or wl)]

    def acc_on(subset, w):
        if not subset:
            return -1
        hit = sum(1 for m in subset
                  if pick(m, gold_D, gold_tprob, *w) == m.gold_concept)
        return hit / len(subset)

    def best_for(subset):
        return max(grid, key=lambda w: acc_on(subset, w)) if subset else grid[0]

    w_seen = best_for([m for m in val if norm(m.span) in seen_forms])
    w_novel = best_for([m for m in val if norm(m.span) not in seen_forms])
    return w_seen, w_novel


def strat_of(m, train_freq):
    return freq_stratum(m.gold_concept, train_freq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="vanilla",
                    choices=["vanilla", "section_aware"])
    ap.add_argument("--max_notes", type=int, default=0)   # 0=all eval notes
    args = ap.parse_args()
    t0 = time.time()

    notes, ann = load()
    train_notes, test_notes = split_notes(ann)
    text_by_id = dict(zip(notes.note_id, notes.text))
    cg = CandidateGenerator().fit(ann, train_notes)
    train_freq = Counter(int(c) for c in ann[ann.note_id.isin(train_notes)].concept_id)
    seen_forms = set(cg.form_to_concepts.keys())

    # same note-level val/eval split as stage2_gated
    tn = sorted(test_notes); random.Random(SEED).shuffle(tn)
    val_notes = set(tn[:max(1, int(len(tn) * VAL_FRAC))])
    eval_notes = set(tn) - val_notes
    if args.max_notes:
        eval_notes = set(sorted(eval_notes)[:args.max_notes])

    gold = build_test_mentions(notes, ann, test_notes, cg)
    gold_val = [m for m in gold if m.note_id in val_notes]
    gold_eval = [m for m in gold if m.note_id in eval_notes]

    print("loading gold caches for gate tuning + oracle-span line...", flush=True)
    gold_D = pickle.load(open(GOLD_POOLS, "rb"))
    gold_tprob = pickle.load(open(GOLD_TPROB, "rb"))
    for m in gold:
        k = (m.note_id, m.start)
        if k in gold_D["pools"]:
            m.candidates = gold_D["pools"][k]
    w_seen, w_novel = tune_gate(gold_D, gold_tprob, gold_val, train_freq, seen_forms)
    print(f"gate: seen-form {w_seen}  novel-form {w_novel}", flush=True)

    def gated(m, D, tprob):
        w = w_seen if norm(m.span) in seen_forms else w_novel
        return pick(m, D, tprob, *w)

    # ---- oracle-span line: gated ranker on GOLD spans (detection recall == 1) --
    def link_acc(subset, D, tprob):
        by = {s: [0, 0] for s in ("head", "torso", "tail")}
        tot = [0, 0]
        for m in subset:
            ok = gated(m, D, tprob) == m.gold_concept
            tot[0] += ok; tot[1] += 1
            b = by[strat_of(m, train_freq)]; b[0] += ok; b[1] += 1
        f = lambda p: round(p[0] / p[1], 4) if p[1] else None
        return {"all": f(tot), **{s: f(by[s]) for s in by}}

    oracle_line = link_acc(gold_eval, gold_D, gold_tprob)

    # ---- predicted-span line ------------------------------------------------
    pred_recs = [json.loads(l) for l in
                 open(PRED_DIR / f"stage0_pred_spans_{args.variant}.jsonl", encoding="utf-8")]
    pred_m = build_pred_mentions(pred_recs, text_by_id, cg, eval_notes)
    print(f"pred mentions (eval notes): {len(pred_m)}  ({time.time()-t0:.0f}s)", flush=True)

    cache = HERE / f".stage0_pools_{args.variant}.pkl"
    if args.max_notes:
        cache = HERE / f".stage0_pools_{args.variant}_smoke{args.max_notes}.pkl"
    pred_D = build_pred_pools(pred_m, cg, cache)
    for m in pred_m:
        m.candidates = pred_D["pools"][(m.note_id, m.start)]

    stage2_reranker.TPROB_CACHE = HERE / (
        f".stage0_typeprob_{args.variant}"
        + (f"_smoke{args.max_notes}" if args.max_notes else "") + ".pkl")
    pred_tprob = TypeProba(TFMR_DIR).proba(pred_m)
    print(f"pred type probs ready ({time.time()-t0:.0f}s)", flush=True)

    pred_concept = {(m.note_id, m.start): gated(m, pred_D, pred_tprob) for m in pred_m}

    # index predicted spans by note for IoU matching
    pred_by_note = {}
    for m in pred_m:
        pred_by_note.setdefault(m.note_id, []).append(
            (m.start, m.end, pred_concept[(m.note_id, m.start)]))

    det = {s: [0, 0] for s in ("head", "torso", "tail")}
    e2e = {s: [0, 0] for s in ("head", "torso", "tail")}
    det_all = [0, 0]; e2e_all = [0, 0]
    for m in gold_eval:
        st = strat_of(m, train_freq)
        cands = pred_by_note.get(m.note_id, [])
        best_iou, best_cpt = 0.0, None
        for (ps, pe, pc) in cands:
            v = iou((m.start, m.end), (ps, pe))
            if v > best_iou:
                best_iou, best_cpt = v, pc
        detected = best_iou >= IOU_THR
        correct = detected and best_cpt == m.gold_concept
        det[st][0] += detected; det[st][1] += 1; det_all[0] += detected; det_all[1] += 1
        e2e[st][0] += correct; e2e[st][1] += 1; e2e_all[0] += correct; e2e_all[1] += 1

    def r(p):
        return round(p[0] / p[1], 4) if p[1] else None

    rows = []
    for scope, dd, ee in [("all", det_all, e2e_all)] + \
            [(s, det[s], e2e[s]) for s in ("head", "torso", "tail")]:
        orc = oracle_line[scope]
        e2e_r = r(ee)
        rows.append({
            "scope": scope, "n_gold": ee[1],
            "oracle_span_link_acc": orc,
            "detection_recall@IoU": r(dd),
            "e2e_recall": e2e_r,
            "linking|detected": round(ee[0] / dd[0], 4) if dd[0] else None,
            "detection_cost": round(orc - e2e_r, 4) if (orc is not None and e2e_r is not None) else None,
        })
    df = pd.DataFrame(rows)
    out = HERE / "stage0_ner" / "result"
    df.to_csv(out / f"STAGE0_E2E_{args.variant}.csv", index=False)
    lines = [f"# Stage-0 end-to-end — {args.variant}", "",
             f"eval notes {len(eval_notes)} · gold eval mentions {len(gold_eval)} · "
             f"pred mentions {len(pred_m)} · IoU thr {IOU_THR}",
             f"gate seen-form {w_seen} / novel-form {w_novel}", "",
             "oracle_span = gated ranker on GOLD spans (detection=1); "
             "e2e = detected@IoU>=0.5 AND linked correctly; "
             "detection_cost = oracle_span - e2e", "",
             tcg.df_to_markdown(df)]
    (out / f"STAGE0_E2E_{args.variant}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + df.to_string(index=False))
    print(f"\nDONE {time.time()-t0:.0f}s -> {out}/STAGE0_E2E_{args.variant}.md", flush=True)


if __name__ == "__main__":
    main()
