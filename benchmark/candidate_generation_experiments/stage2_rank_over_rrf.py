"""Stage 2 entry: rank over the RRF@60 candidate pool to decide top-1.

Candidate generation is closed out (dense + weighted RRF). The crux question now:
does the recovered tail gold (pool reachability ~0.86) get SELECTED? i.e. does
tail ACCURACY rise, and which ranker exploits the pool?

This harness fixes the RRF@60 pool as the shared candidate set and reports
top-1 accuracy, stratified head/torso/tail, for:
  * oracle       gold-in-pool -> picked        (accuracy ceiling = reachability)
  * rrf_rank1    take RRF position 1           (free retrieval ranker)
  * dense_rank1  best SapBERT cosine in pool    (semantic-only ranker)
  * prior_rank1  best train prior P(concept|span) (frequency prior baseline)
  * llm          Ollama qwen2.5:3b over the pool (proposal's Stage-1 ranker) [RUN_LLM=1]

The gap oracle - {rrf,llm} per stratum defines what a learned reranker must close.

    python stage2_rank_over_rrf.py            # cheap rankers only
    RUN_LLM=1 python stage2_rank_over_rrf.py  # + Ollama LLM baseline (slow)
"""
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from candidate_booster import SnomedCandidateBooster, norm
from colab_full_candidate_eval import StubOntology, RRF_WEIGHTS
from data import CandidateGenerator, build_test_mentions, load, split_notes
from dense_retriever import DenseRetriever
from evaluate import freq_stratum
from retrieval_fusion import per_retriever_lists, rrf_fuse
import test_candidate_generation as tcg

K = CAP = RRF_K = 60
SAMPLE_FILE = os.environ.get("SAMPLE_FILE", ".sample600.json")
RUN_LLM = os.environ.get("RUN_LLM", "0") == "1"


def load_sample(cg):
    notes, ann = load()
    train_notes, test_notes = split_notes(ann)
    mentions = build_test_mentions(notes, ann, test_notes, cg)
    want = None
    p = HERE.parent / SAMPLE_FILE
    if p.exists():
        import json
        want = {(d["note_id"], int(d["start"])) for d in json.loads(p.read_text())}
        mentions = [m for m in mentions if (m.note_id, m.start) in want]
    train_freq = Counter(int(c) for c in ann[ann.note_id.isin(train_notes)].concept_id)
    return mentions, train_freq


def strat_accuracy(mentions, picks, train_freq):
    """picks: dict key->concept_id (or None). Accuracy overall + head/torso/tail."""
    def acc(subset):
        if not subset:
            return None
        ok = sum(1 for m in subset if picks.get((m.note_id, m.start)) == m.gold_concept)
        return ok / len(subset)
    rows = {"all": acc(mentions)}
    for s in ("head", "torso", "tail"):
        rows[s] = acc([m for m in mentions if freq_stratum(m.gold_concept, train_freq) == s])
    return rows


def main():
    t0 = time.time()
    print("loading + building RRF@60 pool...", flush=True)
    notes, ann = load()
    train_notes, test_notes = split_notes(ann)
    cg = CandidateGenerator().fit(ann, train_notes)
    mentions, train_freq = load_sample(cg)
    print(f"mentions: {len(mentions)}", flush=True)

    booster = SnomedCandidateBooster(cg, StubOntology(), release_dir=".",
                                     mode="boosted", max_candidates=60, expand_k=40)
    char = tcg.CharNgramRetriever(booster)
    dense = DenseRetriever()
    dres = dense.retrieve_batch([m.span for m in mentions], k=K)
    dense_map = {(m.note_id, m.start): dict(r) for m, r in zip(mentions, dres)}   # cid->cosine
    dense_rank = {(m.note_id, m.start): [c for c, _ in r] for m, r in zip(mentions, dres)}

    # fix the RRF pool as the shared candidate set
    for m in mentions:
        lists = per_retriever_lists(m, booster, char, dense_rank[(m.note_id, m.start)], K)
        m.candidates = rrf_fuse(lists, k=RRF_K, cap=CAP, weights=RRF_WEIGHTS)

    reach = {(m.note_id, m.start): (m.gold_concept in set(m.candidates)) for m in mentions}

    # --- cheap rankers -----------------------------------------------------
    picks = {}
    picks["oracle"] = {k: (m.gold_concept if reach[k] else None)
                       for m in mentions for k in [(m.note_id, m.start)]}
    picks["rrf_rank1"] = {(m.note_id, m.start): (m.candidates[0] if m.candidates else None)
                          for m in mentions}
    def best_dense(m):
        key = (m.note_id, m.start); sc = dense_map[key]
        pool = [c for c in m.candidates if c in sc]
        return max(pool, key=lambda c: sc[c]) if pool else (m.candidates[0] if m.candidates else None)
    picks["dense_rank1"] = {(m.note_id, m.start): best_dense(m) for m in mentions}
    def best_prior(m):
        pool = m.candidates
        scored = [(c, cg.prior(m.span, c)) for c in pool]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored[0][0] if scored else None
    picks["prior_rank1"] = {(m.note_id, m.start): best_prior(m) for m in mentions}

    # --- LLM baseline (optional, slow) ------------------------------------
    if RUN_LLM:
        from pipeline import OllamaRanker
        print("running Ollama LLM baseline over the RRF pool...", flush=True)
        ranker = OllamaRanker(booster, cache_file=str(HERE / ".stage2_llm_cache.json"))
        llm = {}
        for i, m in enumerate(mentions):
            scores = ranker.score(m, list(m.candidates))
            llm[(m.note_id, m.start)] = max(scores, key=scores.get) if scores else None
            if (i + 1) % 50 == 0:
                print(f"  llm {i+1}/{len(mentions)}  ({time.time()-t0:.0f}s)", flush=True)
        picks["llm"] = llm
        print(f"LLM calls={ranker.calls} errors={ranker.errors}", flush=True)

    # --- report ------------------------------------------------------------
    rows = []
    for name, p in picks.items():
        a = strat_accuracy(mentions, p, train_freq)
        rows.append({"ranker": name, **a})
    df = pd.DataFrame(rows)
    out = HERE / "stage2_out"; out.mkdir(exist_ok=True)
    df.to_csv(out / "stage2_rank_accuracy.csv", index=False)
    lines = ["# Stage 2 — Ranking over the RRF@60 pool (fixed 600)", "",
             f"n={len(mentions)} · pool=weighted RRF cap {CAP}", "",
             tcg.df_to_markdown(df.round(4))]
    (out / "STAGE2_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n=== top-1 accuracy by stratum (RRF@60 pool) ===")
    print(df.to_string(index=False, formatters={c: "{:.4f}".format
          for c in ("all", "head", "torso", "tail")}))
    print(f"\nDONE {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
