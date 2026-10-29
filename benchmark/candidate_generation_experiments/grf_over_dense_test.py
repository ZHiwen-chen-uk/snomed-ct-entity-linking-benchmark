"""Does GRF (LLM query rewrite) recover tail gold that dense+RRF misses?

Target = tail mentions whose gold is NOT in the current dense+RRF@60 pool (522).
For each, we ask two things:
  A) is the gold recoverable simply by widening dense retrieval on the ORIGINAL
     mention to k=200?  (the cheap alternative to GRF)
  B) is the gold recoverable by GRF: rewrite the mention with a local LLM, then
     retrieve on the rewrites with dense + lexical + synonym-exact?

GRF's *true* added value is the golds recovered by B but not by A, i.e. cases a
rewrite reaches that plain (wider) dense retrieval cannot.

    python grf_over_dense_test.py
"""
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
BM = HERE.parent
sys.path.insert(0, str(BM)); sys.path.insert(0, str(HERE))

from candidate_booster import SnomedCandidateBooster
from colab_full_candidate_eval import StubOntology
from data import CandidateGenerator, build_test_mentions, load, split_notes
from dense_retriever import DenseRetriever
from evaluate import freq_stratum
from retrieval_fusion import exact_hits

SNOMED_DIR = str(BM.parent / "SnomedCT_InternationalRF2_PRODUCTION_20260601T120000Z")
POOL_CACHE = HERE / ".stage2_pools_full.pkl"


def main():
    t0 = time.time()
    notes, ann = load()
    tr, te = split_notes(ann)
    cg = CandidateGenerator().fit(ann, tr)
    ms = build_test_mentions(notes, ann, te, cg)
    tf = Counter(int(c) for c in ann[ann.note_id.isin(tr)].concept_id)
    pools = pickle.load(open(POOL_CACHE, "rb"))["pools"]

    target = [m for m in ms
              if freq_stratum(m.gold_concept, tf) == "tail"
              and (m.note_id, m.start) in pools
              and m.gold_concept not in set(pools[(m.note_id, m.start)])]
    print(f"unreachable-tail targets: {len(target)}", flush=True)

    booster = SnomedCandidateBooster(cg, StubOntology(), release_dir=".", mode="grf",
                                     max_candidates=60, expand_k=30, grf_mode="ollama")
    dense = DenseRetriever()

    # A) widen dense on the ORIGINAL mention to k=200
    print("A) dense@200 on original spans...", flush=True)
    d200 = dense.retrieve_batch([m.span for m in target], k=200, top_forms=1200)
    recov_dense200 = {}
    for m, r in zip(target, d200):
        recov_dense200[(m.note_id, m.start)] = m.gold_concept in {c for c, _ in r}

    # B) GRF: generate rewrites (Ollama), then retrieve on them
    print("B) generating GRF rewrites (Ollama)...", flush=True)
    rewrites = {}
    for i, m in enumerate(target):
        rw = booster.grf_rewrite(m)
        rewrites[(m.note_id, m.start)] = [t for t in rw if str(t).strip()]
        if (i + 1) % 50 == 0:
            print(f"  grf {i+1}/{len(target)} ({time.time()-t0:.0f}s) calls={booster.grf_calls} err={booster.grf_errors}", flush=True)

    # dense-retrieve on all rewrite terms at once, then map back
    flat, owner = [], []
    for m in target:
        for t in rewrites[(m.note_id, m.start)]:
            flat.append(t); owner.append((m.note_id, m.start))
    print(f"B) dense on {len(flat)} rewrite terms...", flush=True)
    dense_rw = {}
    if flat:
        res = dense.retrieve_batch(flat, k=30)
        for (key, r) in zip(owner, res):
            dense_rw.setdefault(key, set()).update(c for c, _ in r)

    recov_grf = {}
    for m in target:
        key = (m.note_id, m.start)
        rw = rewrites[key]
        cands = set(dense_rw.get(key, set()))
        if rw:
            lex, _ = booster.retrieve(rw, 30)
            cands.update(lex)
            for t in rw:
                cands.update(exact_hits(booster, t))
        recov_grf[key] = m.gold_concept in cands

    n = len(target)
    a = sum(recov_dense200.values())
    b = sum(recov_grf.values())
    grf_unique = sum(1 for m in target
                     if recov_grf[(m.note_id, m.start)] and not recov_dense200[(m.note_id, m.start)])
    both = sum(1 for m in target
               if recov_grf[(m.note_id, m.start)] and recov_dense200[(m.note_id, m.start)])

    print("\n=== GRF-over-dense incremental test (unreachable-tail) ===")
    print(f"targets (tail gold missing from dense+RRF@60): {n}")
    print(f"A) recovered by widening dense to k=200 on original: {a} ({a/n:.1%})")
    print(f"B) recovered by GRF rewrites (dense+lexical+synonym): {b} ({b/n:.1%})")
    print(f"   GRF-unique (B but not A): {grf_unique} ({grf_unique/n:.1%})")
    print(f"   recovered by both: {both}")
    print(f"ollama calls={booster.grf_calls} errors={booster.grf_errors}  ({time.time()-t0:.0f}s)")

    # a few GRF-unique examples
    ex = []
    for m in target:
        key = (m.note_id, m.start)
        if recov_grf[key] and not recov_dense200[key]:
            ex.append((m.span, booster.label(m.gold_concept), rewrites[key][:4]))
    print("\nGRF-unique examples (span -> gold_term | rewrites):")
    for span, gold, rw in ex[:15]:
        print(f"  {span!r} -> {gold!r} | {rw}")

    out = HERE / "stage2_out"; out.mkdir(exist_ok=True)
    with open(out / "GRF_OVER_DENSE_RESULTS.md", "w", encoding="utf-8") as f:
        f.write("# GRF over dense+RRF — incremental tail-recovery test\n\n")
        f.write(f"- targets (tail gold missing from dense+RRF@60): {n}\n")
        f.write(f"- recovered by widening dense to k=200 on original: {a} ({a/n:.1%})\n")
        f.write(f"- recovered by GRF rewrites: {b} ({b/n:.1%})\n")
        f.write(f"- **GRF-unique (not reachable by wider dense): {grf_unique} ({grf_unique/n:.1%})**\n")
        f.write(f"- recovered by both: {both}\n\n")
        f.write("GRF-unique examples (span -> gold | rewrites):\n\n")
        for span, gold, rw in ex[:20]:
            f.write(f"- `{span}` -> `{gold}` | {rw}\n")
    print(f"\nsaved -> {out/'GRF_OVER_DENSE_RESULTS.md'}")


if __name__ == "__main__":
    main()
