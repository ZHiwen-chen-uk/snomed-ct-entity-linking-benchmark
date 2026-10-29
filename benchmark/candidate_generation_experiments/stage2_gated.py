"""Stage 2 #1: fixed grid + GATED ranker over the full 23k RRF pool.

Fixes the earlier grid (which forced w_dense=1 so it never found that type_only
was best overall) and adds a routing gate for the head-vs-tail regime split we
observed: head/torso want the transformer-A type signal, tail wants dense cosine.

The gate is a ROUTING rule on an observable property (does the mention's surface
form carry any training prior), NOT a learned weight, so it cannot overfit the
prior the way the supervised HistGBM did:

    seen-in-train form  -> type-based pick   (head/torso regime)
    novel form          -> dense+type pick   (tail regime)

Pools / dense / lexical scores are cached to disk on first run (~30 min for 23k),
instant afterwards; transformer-A probs reuse .stage2_typeprob_cache.pkl.

    python stage2_gated.py
"""
import os
import pickle
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

from candidate_booster import SnomedCandidateBooster, norm
from data import CandidateGenerator, build_test_mentions, load, split_notes
from dense_retriever import DenseRetriever
from evaluate import freq_stratum
from ontology import SnomedOntology
from retrieval_fusion import per_retriever_lists, rrf_fuse, exact_hits
from stage2_reranker import TypeProba, TYPE4
import test_candidate_generation as tcg

RRF_WEIGHTS = {"synonym": 3.0, "train": 2.0, "lexical": 1.0, "char": 1.0, "dense": 1.5}
K = CAP = RRF_K = 60
SNOMED_DIR = str(BM.parent / "SnomedCT_InternationalRF2_PRODUCTION_20260601T120000Z")
TFMR_DIR = BM / "group_a_transformer" / "outputs" / "pubmedbert_semtype" / "best_model"
POOL_CACHE = HERE / ".stage2_pools_full.pkl"
VAL_FRAC = 0.2
SEED = 20260723


def build_or_load_pools(mentions, cg):
    if POOL_CACHE.exists():
        print("loading cached pools...", flush=True)
        with open(POOL_CACHE, "rb") as f:
            return pickle.load(f)
    print("building pools (first time, ~30min)...", flush=True)
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
        pool = rrf_fuse(lists, k=RRF_K, cap=CAP, weights=RRF_WEIGHTS)
        pools[key] = pool
        dcos[key] = dict(dense_pairs)
        lexsc[key] = dict(lex_s)
        coarse[key] = {c: ontology.coarse_type(c) for c in pool}
        if (i + 1) % 5000 == 0:
            print(f"  pooled {i+1}/{len(mentions)}", flush=True)
    data = {"pools": pools, "dcos": dcos, "lexsc": lexsc, "coarse": coarse}
    with open(POOL_CACHE, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return data


def pick(m, D, tprob, wd, wt, wl):
    key = (m.note_id, m.start)
    tp = tprob.get(key, {})
    dcos, lexsc, coarse = D["dcos"][key], D["lexsc"][key], D["coarse"][key]
    best_c, best_s = None, -1e9
    for c in D["pools"][key]:
        ct = coarse[c]; ct4 = ct if ct in TYPE4 else "other"
        s = wd * dcos.get(c, 0.0) + wt * tp.get(ct4, 0.0) + wl * lexsc.get(c, 0.0)
        if s > best_s:
            best_c, best_s = c, s
    return best_c


def strat(picks, mentions, train_freq):
    def a(sub):
        return None if not sub else round(sum(
            1 for m in sub if picks.get((m.note_id, m.start)) == m.gold_concept) / len(sub), 4)
    return {"all": a(mentions),
            **{s: a([m for m in mentions if freq_stratum(m.gold_concept, train_freq) == s])
               for s in ("head", "torso", "tail")}}


def main():
    t0 = time.time()
    notes, ann = load()
    train_notes, test_notes = split_notes(ann)
    cg = CandidateGenerator().fit(ann, train_notes)
    mentions = build_test_mentions(notes, ann, test_notes, cg)
    train_freq = Counter(int(c) for c in ann[ann.note_id.isin(train_notes)].concept_id)
    seen_forms = set(cg.form_to_concepts.keys())      # forms with a training prior
    print(f"test mentions: {len(mentions)}", flush=True)

    tn = sorted(test_notes)
    import random
    random.Random(SEED).shuffle(tn)
    val_notes = set(tn[:max(1, int(len(tn) * VAL_FRAC))])
    val = [m for m in mentions if m.note_id in val_notes]
    ev = [m for m in mentions if m.note_id not in val_notes]

    D = build_or_load_pools(mentions, cg)
    for m in mentions:
        m.candidates = D["pools"][(m.note_id, m.start)]
    print(f"pools ready ({time.time()-t0:.0f}s)", flush=True)

    tp = TypeProba(TFMR_DIR); tprob = tp.proba(mentions)
    print(f"type probs ready ({time.time()-t0:.0f}s)", flush=True)

    # full grid incl. dense-off, tuned on val (all-accuracy)
    grid = [(wd, wt, wl) for wd in (0, 0.5, 1, 2) for wt in (0, 0.5, 1, 2)
            for wl in (0, 0.05, 0.1) if (wd or wt or wl)]
    best, best_acc = None, -1
    for w in grid:
        acc = strat({(m.note_id, m.start): pick(m, D, tprob, *w) for m in val}, val, train_freq)["all"]
        if acc > best_acc:
            best, best_acc = w, acc
    print(f"best global weights (val all={best_acc}): dense={best[0]} type={best[1]} lex={best[2]}", flush=True)

    # per-regime best weights (tune head/torso and tail separately on val)
    def best_for(subset):
        b, ba = None, -1
        for w in grid:
            acc = strat({(m.note_id, m.start): pick(m, D, tprob, *w) for m in subset}, subset, train_freq)["all"]
            if acc > ba:
                b, ba = w, acc
        return b
    val_headish = [m for m in val if norm(m.span) in seen_forms]
    val_novel = [m for m in val if norm(m.span) not in seen_forms]
    w_seen = best_for(val_headish) if val_headish else best
    w_novel = best_for(val_novel) if val_novel else best
    print(f"gate weights: seen-form={w_seen}  novel-form={w_novel}", flush=True)

    def gated_pick(m):
        w = w_seen if norm(m.span) in seen_forms else w_novel
        return pick(m, D, tprob, *w)

    oracle = {(m.note_id, m.start): (m.gold_concept if m.gold_concept in set(m.candidates)
              else None) for m in ev}
    rows = [
        {"policy": "oracle", **strat(oracle, ev, train_freq)},
        {"policy": "type_only", **strat({(m.note_id, m.start): pick(m, D, tprob, 0, 1, 0) for m in ev}, ev, train_freq)},
        {"policy": "dense_only", **strat({(m.note_id, m.start): pick(m, D, tprob, 1, 0, 0) for m in ev}, ev, train_freq)},
        {"policy": f"best_global{best}", **strat({(m.note_id, m.start): pick(m, D, tprob, *best) for m in ev}, ev, train_freq)},
        {"policy": "gated(seen->type,novel->dense)", **strat({(m.note_id, m.start): gated_pick(m) for m in ev}, ev, train_freq)},
    ]
    df = pd.DataFrame(rows)
    out = HERE / "stage2_out"; out.mkdir(exist_ok=True)
    df.to_csv(out / "stage2_gated.csv", index=False)
    lines = ["# Stage 2 #1 — Gated ranker, full 23k (eval split)", "",
             f"eval n={len(ev)} val n={len(val)}", "",
             f"global best {best}; gate: seen-form {w_seen}, novel-form {w_novel}", "",
             tcg.df_to_markdown(df)]
    (out / "STAGE2_GATED_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + df.to_string(index=False))
    print(f"\nDONE {time.time()-t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
