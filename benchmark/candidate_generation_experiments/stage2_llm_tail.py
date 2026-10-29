"""Item 3: can a context LLM beat the simple signals on the tail?

On a sample of tail eval mentions, rank the RRF pool (capped to top-K) with an
LLM and compare, on the SAME subset, against the leakage-free simple rankers.
Target to beat: gated tail 0.561 (the current tail winner).

    RANKER=claude LIMIT=300 python stage2_llm_tail.py
    RANKER=ollama LIMIT=300 python stage2_llm_tail.py
"""
import os
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BM = HERE.parent
sys.path.insert(0, str(BM)); sys.path.insert(0, str(HERE))

from candidate_booster import SnomedCandidateBooster, norm
from colab_full_candidate_eval import StubOntology
from data import CandidateGenerator, build_test_mentions, load, split_notes
from evaluate import freq_stratum

TYPE4 = {"body structure", "finding", "procedure"}
VAL_FRAC = 0.2; SEED = 20260723
POOL_CACHE = HERE / ".stage2_pools_full.pkl"
TPROB_CACHE = HERE / ".stage2_typeprob_cache.pkl"
BACKEND = os.environ.get("RANKER", "claude")
LIMIT = int(os.environ.get("LIMIT", "300"))
TOPK = int(os.environ.get("TOPK", "20"))


def coarse4(t):
    return t if t in TYPE4 else "other"


def main():
    t0 = time.time()
    notes, ann = load()
    tr, te = split_notes(ann)
    cg = CandidateGenerator().fit(ann, tr)
    ms = build_test_mentions(notes, ann, te, cg)
    tf = Counter(int(c) for c in ann[ann.note_id.isin(tr)].concept_id)
    seen = set(cg.form_to_concepts.keys())

    D = pickle.load(open(POOL_CACHE, "rb")); tp = pickle.load(open(TPROB_CACHE, "rb"))
    pools, dcos, coarse, lexsc = D["pools"], D["dcos"], D["coarse"], D["lexsc"]

    tn = sorted(te); import random; random.Random(SEED).shuffle(tn)
    val_notes = set(tn[: max(1, int(len(tn) * VAL_FRAC))])
    tail = [m for m in ms if m.note_id not in val_notes
            and freq_stratum(m.gold_concept, tf) == "tail"
            and (m.note_id, m.start) in pools and (m.note_id, m.start) in tp]
    random.Random(SEED).shuffle(tail)
    tail = tail[:LIMIT]
    # cap each pool to top-K for the LLM
    for m in tail:
        m.candidates = pools[(m.note_id, m.start)][:TOPK]
    print(f"tail sample: {len(tail)}  top-K={TOPK}  backend={BACKEND}", flush=True)

    booster = SnomedCandidateBooster(cg, StubOntology(), release_dir=".", mode="synonym")

    if BACKEND == "claude":
        from pipeline import ClaudeRanker
        ranker = ClaudeRanker(booster, cache_file=str(HERE / ".llm_tail_claude_cache.json"))
    else:
        from pipeline import OllamaRanker
        ranker = OllamaRanker(booster, cache_file=str(HERE / ".llm_tail_ollama_cache.json"))

    gold = {(m.note_id, m.start): m.gold_concept for m in tail}

    def score_pick(m, wd, wt, wl):
        key = (m.note_id, m.start); dc, lx, cz, pr = dcos[key], lexsc[key], coarse[key], tp[key]
        return max(m.candidates, key=lambda c: wd*dc.get(c,0.0) + wt*pr.get(coarse4(cz[c]),0.0) + wl*lx.get(c,0.0)) if m.candidates else None
    def gated(m):
        w = (0,0.5,0) if norm(m.span) in seen else (0.5,0.5,0); return score_pick(m,*w)

    picks = {"oracle": {}, "dense": {}, "type": {}, "gated": {}, "llm": {}}
    for i, m in enumerate(tail):
        key = (m.note_id, m.start)
        picks["oracle"][key] = m.gold_concept if m.gold_concept in set(m.candidates) else None
        picks["dense"][key] = score_pick(m, 1, 0, 0)
        picks["type"][key] = score_pick(m, 0, 1, 0)
        picks["gated"][key] = gated(m)
        scores = ranker.score(m, list(m.candidates))
        picks["llm"][key] = max(scores, key=scores.get) if scores else None
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(tail)}  llm_calls={ranker.calls} err={ranker.errors} ({time.time()-t0:.0f}s)", flush=True)

    def acc(p): return round(sum(1 for k in gold if p.get(k) == gold[k]) / len(gold), 4)
    print("\n=== tail ranking on the same subset (RRF top-{} pool) ===".format(TOPK))
    for name in ("oracle", "dense", "type", "gated", "llm"):
        print(f"  {name:8} {acc(picks[name]):.4f}")
    print(f"llm backend={BACKEND} calls={ranker.calls} errors={ranker.errors} ({time.time()-t0:.0f}s)")

    out = HERE / "stage2_out"; out.mkdir(exist_ok=True)
    with open(out / "STAGE2_LLM_TAIL.md", "w", encoding="utf-8") as f:
        f.write(f"# Stage 2 — LLM tail ranking ({BACKEND}, n={len(tail)}, RRF top-{TOPK})\n\n")
        f.write("| ranker | tail accuracy |\n|---|---:|\n")
        for name in ("oracle", "dense", "type", "gated", "llm"):
            f.write(f"| {name} | {acc(picks[name]):.4f} |\n")
    print(f"saved -> {out/'STAGE2_LLM_TAIL.md'}")


if __name__ == "__main__":
    main()
