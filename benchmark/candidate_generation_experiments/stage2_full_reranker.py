"""Stage 2, FULL 23k test set: unsupervised dense+type(+lexical) reranker.

The supervised reranker failed the tail through train/test prior leakage, so this
one avoids the leak by construction. It combines the two strongest signals that
do not come from the training index, SapBERT dense cosine (which wins the tail)
and the transformer type probability (Group A's soft reranker, which wins
cross-branch), as a weighted score over the RRF@60 pool:

    score(c) = w_dense * dense_cosine(mention, c)
             + w_type  * P(coarse_type(c) | mention)      # Group A soft rerank
             + w_lex   * lexical_score(c)

No prior / in_train / freq features -> no train/test mismatch. No supervised
training -> weights tuned on a val split of the test notes, reported on the rest.
Runs on CPU (local, ~1h for transformer-A over 23k, cached) or GPU (Colab).

    python stage2_full_reranker.py
"""
import os
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

from candidate_booster import SnomedCandidateBooster
from data import CandidateGenerator, build_test_mentions, load, split_notes
from dense_retriever import DenseRetriever
from evaluate import freq_stratum
from ontology import SnomedOntology
from retrieval_fusion import per_retriever_lists, rrf_fuse, exact_hits
from stage2_reranker import TypeProba, TYPE4          # reuse cached transformer-A probs
import test_candidate_generation as tcg

RRF_WEIGHTS = {"synonym": 3.0, "train": 2.0, "lexical": 1.0, "char": 1.0, "dense": 1.5}
K = CAP = RRF_K = 60
SNOMED_DIR = str(BM.parent / "SnomedCT_InternationalRF2_PRODUCTION_20260601T120000Z")
TFMR_DIR = BM / "group_a_transformer" / "outputs" / "pubmedbert_semtype" / "best_model"
VAL_FRAC = 0.2
SEED = 20260723


def build_pools(mentions, booster, char, dense, k):
    """RRF@60 pool + per-candidate dense cosine and lexical score."""
    spans = [m.span for m in mentions]
    dmap = {}
    for s in range(0, len(spans), 1000):
        for j, r in enumerate(dense.retrieve_batch(spans[s:s + 1000], k=k)):
            dmap[s + j] = r
    pools, dcos, lexsc = {}, {}, {}
    for i, m in enumerate(mentions):
        key = (m.note_id, m.start)
        dense_pairs = dmap[i]
        lex, lex_s = booster.retrieve([m.span], k)
        ch, _ = char.retrieve(m.span, k)
        lists = {"train": list(m.candidates), "synonym": exact_hits(booster, m.span),
                 "lexical": lex, "char": ch, "dense": [c for c, _ in dense_pairs]}
        pools[key] = rrf_fuse(lists, k=RRF_K, cap=CAP, weights=RRF_WEIGHTS)
        dcos[key] = dict(dense_pairs)
        lexsc[key] = dict(lex_s)
    return pools, dcos, lexsc


def score_pick(m, pool, dcos, lexsc, tprob, ontology, w_dense, w_type, w_lex):
    key = (m.note_id, m.start)
    tp = tprob.get(key, {})
    best_c, best_s = None, -1e9
    for c in pool:
        ct = ontology.coarse_type(c)
        ct4 = ct if ct in TYPE4 else "other"
        s = (w_dense * dcos[key].get(c, 0.0)
             + w_type * tp.get(ct4, 0.0)
             + w_lex * lexsc[key].get(c, 0.0))
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
    print("loading full test set...", flush=True)
    notes, ann = load()
    train_notes, test_notes = split_notes(ann)
    cg = CandidateGenerator().fit(ann, train_notes)
    mentions = build_test_mentions(notes, ann, test_notes, cg)
    train_freq = Counter(int(c) for c in ann[ann.note_id.isin(train_notes)].concept_id)
    print(f"test mentions: {len(mentions)}", flush=True)

    # val/eval split by note (weights tuned on val, reported on eval)
    tn = sorted(test_notes)
    import random
    random.Random(SEED).shuffle(tn)
    val_notes = set(tn[:max(1, int(len(tn) * VAL_FRAC))])
    val = [m for m in mentions if m.note_id in val_notes]
    ev = [m for m in mentions if m.note_id not in val_notes]
    print(f"val={len(val)} eval={len(ev)}", flush=True)

    print("retrievers + ontology...", flush=True)
    ontology = SnomedOntology(SNOMED_DIR)
    booster = SnomedCandidateBooster(cg, ontology, SNOMED_DIR, mode="boosted",
                                     max_candidates=60, expand_k=40)
    char = tcg.CharNgramRetriever(booster)
    dense = DenseRetriever()

    print("building pools (dense+lexical scores)...", flush=True)
    pools, dcos, lexsc = build_pools(mentions, booster, char, dense, K)
    for m in mentions:
        m.candidates = pools[(m.note_id, m.start)]
    print(f"  pools done ({time.time()-t0:.0f}s)", flush=True)

    print("transformer-A type probs (cached; ~1h first time)...", flush=True)
    tp = TypeProba(TFMR_DIR)
    tprob = tp.proba(mentions)
    print(f"  type probs done ({time.time()-t0:.0f}s)", flush=True)

    # baselines (leakage-free)
    def picks_for(w_d, w_t, w_l, subset):
        return {(m.note_id, m.start): score_pick(m, m.candidates, dcos, lexsc, tprob,
                                                  ontology, w_d, w_t, w_l) for m in subset}
    oracle = {(m.note_id, m.start): (m.gold_concept if m.gold_concept in set(m.candidates)
              else None) for m in mentions}

    # tune (w_type, w_lex) on val with w_dense=1
    grid = [(1.0, wt, wl) for wt in (0.0, 0.1, 0.25, 0.5, 1.0) for wl in (0.0, 0.02, 0.05, 0.1)]
    best, best_acc = None, -1
    for (wd, wt, wl) in grid:
        acc = strat(picks_for(wd, wt, wl, val), val, train_freq)["all"]
        if acc > best_acc:
            best, best_acc = (wd, wt, wl), acc
    print(f"best weights (val): dense={best[0]} type={best[1]} lex={best[2]}  val_all={best_acc}", flush=True)

    rows = []
    rows.append({"policy": "oracle", **strat(oracle, ev, train_freq)})
    rows.append({"policy": "dense_only", **strat(picks_for(1, 0, 0, ev), ev, train_freq)})
    rows.append({"policy": "type_only", **strat(picks_for(0, 1, 0, ev), ev, train_freq)})
    rows.append({"policy": f"dense+type+lex({best[1]},{best[2]})",
                 **strat(picks_for(*best, ev), ev, train_freq)})
    df = pd.DataFrame(rows)
    out = HERE / "stage2_out"; out.mkdir(exist_ok=True)
    df.to_csv(out / "stage2_full_reranker.csv", index=False)
    lines = ["# Stage 2 — Full-23k dense+type reranker (leakage-free)", "",
             f"eval n={len(ev)} (val n={len(val)}) · best weights dense={best[0]} "
             f"type={best[1]} lex={best[2]}", "", tcg.df_to_markdown(df)]
    (out / "STAGE2_FULL_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + df.to_string(index=False))
    print(f"\nDONE {time.time()-t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
