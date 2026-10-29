"""Full-dataset candidate-generation eval, LOCAL (reuses the already-built
SapBERT index; no GPU/Colab needed). All 23k held-out test mentions, stratified
head/torso/tail, weighted RRF vs naive concat.

    python run_full_local.py            # writes full_eval_out_local/
"""
import copy
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from candidate_booster import SnomedCandidateBooster, ordered_unique
from colab_full_candidate_eval import StubOntology, summarize, RRF_WEIGHTS
from data import CandidateGenerator, build_test_mentions, load, split_notes
from dense_retriever import DenseRetriever
from retrieval_fusion import per_retriever_lists, rrf_fuse
import test_candidate_generation as tcg

K = CAP = RRF_K = 60
DENSE_QUERY_CHUNK = 1000     # keep the query x forms matmul under ~0.4GB/block


def main():
    t0 = time.time()
    print("loading data + full test mentions...", flush=True)
    notes, ann = load()
    train_notes, test_notes = split_notes(ann)
    cg = CandidateGenerator().fit(ann, train_notes)
    mentions = build_test_mentions(notes, ann, test_notes, cg)
    train_freq = Counter(int(c) for c in ann[ann.note_id.isin(train_notes)].concept_id)
    print(f"test mentions: {len(mentions)}  ({time.time()-t0:.0f}s)", flush=True)

    print("building lexical retrievers (cached) + dense index...", flush=True)
    booster = SnomedCandidateBooster(cg, StubOntology(), release_dir=".",
                                     mode="boosted", max_candidates=60, expand_k=40)
    char_retriever = tcg.CharNgramRetriever(booster)
    retriever = DenseRetriever()

    print("dense retrieval (chunked)...", flush=True)
    dense_map = {}
    spans = [m.span for m in mentions]
    for s in range(0, len(spans), DENSE_QUERY_CHUNK):
        chunk = spans[s : s + DENSE_QUERY_CHUNK]
        res = retriever.retrieve_batch(chunk, k=K)
        for j, r in enumerate(res):
            dense_map[s + j] = [c for c, _ in r]
        print(f"  dense {s+len(chunk)}/{len(spans)}  ({time.time()-t0:.0f}s)", flush=True)

    print("per-retriever lists + fusion...", flush=True)
    concat, rrf_w = [], []
    for i, m in enumerate(mentions):
        lists = per_retriever_lists(m, booster, char_retriever, dense_map[i], K)
        c1 = copy.copy(m)
        c1.candidates = ordered_unique(
            lists["train"] + lists["synonym"] + lists["char"] + lists["lexical"] + lists["dense"], CAP)
        c2 = copy.copy(m)
        c2.candidates = rrf_fuse(lists, k=RRF_K, cap=CAP, weights=RRF_WEIGHTS)
        concat.append(c1); rrf_w.append(c2)
        if (i + 1) % 5000 == 0:
            print(f"  fused {i+1}/{len(mentions)}  ({time.time()-t0:.0f}s)", flush=True)

    out = HERE / "full_eval_out_local"; out.mkdir(exist_ok=True)
    rows = summarize("concat", concat, train_freq) + summarize("rrf_weighted", rrf_w, train_freq)
    df = pd.DataFrame(rows)
    df.to_csv(out / "full_candidate_eval.csv", index=False)

    lines = ["# Full-Dataset Candidate Generation — LOCAL (all test mentions)", "",
             f"n={len(mentions)} · k={K} · cap={CAP} · RRF weights {RRF_WEIGHTS}", ""]
    for scope in ("all", "head", "torso", "tail"):
        sub = df[df.scope == scope][["policy", "n", "reach", "r@1", "r@10", "r@30", "r@60", "mean_cand"]]
        lines += [f"## {scope}", "", tcg.df_to_markdown(sub), ""]
    (out / "FULL_CANDIDATE_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + df.to_string(index=False))
    print(f"\nDONE {time.time()-t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
