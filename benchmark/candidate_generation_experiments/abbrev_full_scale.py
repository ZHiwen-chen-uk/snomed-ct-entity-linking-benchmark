"""Full-scale (all 23,131 test mentions) A/B for case-sensitive acronym expansion.

Replaces the fixed-600 version in test_abbrev_expansion.py, and does so against
the FINAL pipeline rather than the old concat pools: the question is whether
adding a sixth "abbrev" retriever to the weighted RRF fusion changes what the
Stage-1 pool can reach.

One retrieval pass over every mention, then two fusions of the same lists:

    base    = {train, synonym, lexical, char, dense}                 (shipped pipeline)
    +abbrev = base + {abbrev}   (precise mode: mined concepts + EXACT synonym
                                 lookup of their expansion phrases)

Reported on all / head / torso / tail / acronym-slice: reachability, recall@k,
mean pool size, and the count of gold concepts reachable ONLY with the expander.

Resumable: the per-mention retriever lists are cached, so a re-run after a crash
is instant.  Output: ABBREV_FULLSCALE_RESULTS.md + abbrev_fullscale.csv

    KMP_DUPLICATE_LIB_OK=TRUE python abbrev_full_scale.py
"""
import os
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
BM = HERE.parent
sys.path.insert(0, str(BM)); sys.path.insert(0, str(HERE))

from abbrev_expander import AbbrevExpander
from candidate_booster import SnomedCandidateBooster
from data import CandidateGenerator, build_test_mentions, load, split_notes
from dense_retriever import DenseRetriever
from evaluate import freq_stratum
from ontology import SnomedOntology
from retrieval_fusion import exact_hits, rrf_fuse
import test_candidate_generation as tcg

RRF_WEIGHTS = {"synonym": 3.0, "train": 2.0, "lexical": 1.0, "char": 1.0, "dense": 1.5}
ABBREV_WEIGHT = 1.5          # same weight class as dense: a semantic, not exact, route
K = CAP = RRF_K = 60
SNOMED_DIR = str(BM.parent / "SnomedCT_InternationalRF2_PRODUCTION_20260601T120000Z")
LIST_CACHE = HERE / ".abbrev_fullscale_lists.pkl"
STRATA = ("head", "torso", "tail")


def build_or_load_lists(mentions, cg, ann, train_notes):
    """Per-mention retriever lists for all six retrievers (cached)."""
    if LIST_CACHE.exists():
        print("loading cached retriever lists...", flush=True)
        with open(LIST_CACHE, "rb") as f:
            return pickle.load(f)

    print("building retrievers...", flush=True)
    ontology = SnomedOntology(SNOMED_DIR)
    booster = SnomedCandidateBooster(cg, ontology, SNOMED_DIR, mode="boosted",
                                     max_candidates=60, expand_k=40)
    char = tcg.CharNgramRetriever(booster)
    dense = DenseRetriever()
    expander = AbbrevExpander.from_training(ann, train_notes)
    print(f"mined acronyms={len(expander.mined)} curated={len(expander.curated)}", flush=True)

    t0 = time.time()
    spans = [m.span for m in mentions]
    dmap = {}
    for s in range(0, len(spans), 1000):
        for j, r in enumerate(dense.retrieve_batch(spans[s:s + 1000], k=K)):
            dmap[s + j] = r
        print(f"  dense {min(s+1000,len(spans))}/{len(spans)} ({time.time()-t0:.0f}s)", flush=True)

    out = {}
    for i, m in enumerate(mentions):
        lex, _ = booster.retrieve([m.span], K)
        ch, _ = char.retrieve(m.span, K)
        # precise abbrev layer: mined concepts + EXACT synonym lookup of expansions
        direct, terms = expander.expand(m.span, labeler=booster.label)
        abbrev = list(direct)
        for t in terms:
            abbrev.extend(exact_hits(booster, t))
        out[(m.note_id, m.start)] = {
            "train": list(m.candidates),
            "synonym": exact_hits(booster, m.span),
            "lexical": lex,
            "char": ch,
            "dense": [c for c, _ in dmap[i]],
            "abbrev": abbrev,
            "has_acronym": expander.has_acronym(m.span),
        }
        if (i + 1) % 2000 == 0:
            print(f"  lists {i+1}/{len(mentions)} ({time.time()-t0:.0f}s)", flush=True)
    with open(LIST_CACHE, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out


def fuse(lists, use_abbrev):
    src = {k: v for k, v in lists.items()
           if k in RRF_WEIGHTS or (use_abbrev and k == "abbrev")}
    w = dict(RRF_WEIGHTS)
    if use_abbrev:
        w["abbrev"] = ABBREV_WEIGHT
    return rrf_fuse(src, k=RRF_K, cap=CAP, weights=w)


def metrics(subset, pools):
    n = len(subset)
    got = lambda m, k: m.gold_concept in set(pools[(m.note_id, m.start)][:k])
    return {
        "n": n,
        "reachability": round(sum(1 for m in subset if got(m, CAP)) / n, 4),
        "recall@10": round(sum(1 for m in subset if got(m, 10)) / n, 4),
        "recall@30": round(sum(1 for m in subset if got(m, 30)) / n, 4),
        "mean_candidates": round(sum(len(pools[(m.note_id, m.start)]) for m in subset) / n, 2),
    }


def main():
    t0 = time.time()
    notes, ann = load()
    train_notes, test_notes = split_notes(ann)
    cg = CandidateGenerator().fit(ann, train_notes)
    mentions = build_test_mentions(notes, ann, test_notes, cg)
    train_freq = Counter(int(c) for c in ann[ann.note_id.isin(train_notes)].concept_id)
    print(f"full test mentions: {len(mentions)}", flush=True)

    L = build_or_load_lists(mentions, cg, ann, train_notes)
    print(f"lists ready ({time.time()-t0:.0f}s)", flush=True)

    base, plus = {}, {}
    for m in mentions:
        key = (m.note_id, m.start)
        base[key] = fuse(L[key], False)
        plus[key] = fuse(L[key], True)
    print(f"fused ({time.time()-t0:.0f}s)", flush=True)

    scopes = [("all", mentions)]
    scopes += [(s, [m for m in mentions
                    if freq_stratum(m.gold_concept, train_freq) == s]) for s in STRATA]
    acro = [m for m in mentions if L[(m.note_id, m.start)]["has_acronym"]]
    scopes.append(("acronym_slice", acro))
    scopes.append(("acronym_tail", [m for m in acro
                                    if freq_stratum(m.gold_concept, train_freq) == "tail"]))

    rows = []
    for scope, subset in scopes:
        if not subset:
            continue
        b, p = metrics(subset, base), metrics(subset, plus)
        only = sum(1 for m in subset
                   if m.gold_concept not in set(base[(m.note_id, m.start)])
                   and m.gold_concept in set(plus[(m.note_id, m.start)]))
        lost = sum(1 for m in subset
                   if m.gold_concept in set(base[(m.note_id, m.start)])
                   and m.gold_concept not in set(plus[(m.note_id, m.start)]))
        rows.append({
            "scope": scope, "n": b["n"],
            "reach_base": b["reachability"], "reach_abbrev": p["reachability"],
            "d_reach": round(p["reachability"] - b["reachability"], 4),
            "r10_base": b["recall@10"], "r10_abbrev": p["recall@10"],
            "d_r10": round(p["recall@10"] - b["recall@10"], 4),
            "r30_base": b["recall@30"], "r30_abbrev": p["recall@30"],
            "d_r30": round(p["recall@30"] - b["recall@30"], 4),
            "gold_only_with_abbrev": only, "gold_lost_with_abbrev": lost,
            "mean_cand_base": b["mean_candidates"], "mean_cand_abbrev": p["mean_candidates"],
        })

    df = pd.DataFrame(rows)
    out = HERE / "stage2_out"; out.mkdir(exist_ok=True)
    df.to_csv(out / "abbrev_fullscale.csv", index=False)
    lines = [
        "# Abbreviation expansion — FULL SCALE (all test mentions)", "",
        f"n={len(mentions)} · pool = weighted RRF@{CAP} · abbrev added as a 6th "
        f"retriever (weight {ABBREV_WEIGHT}, precise mode)", "",
        "`gold_only_with_abbrev` = gold reachable ONLY when the expander is on; "
        "`gold_lost_with_abbrev` = gold displaced out of the pool by the extra list.", "",
        tcg.df_to_markdown(df),
    ]
    (out / "ABBREV_FULLSCALE_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(df.to_string(index=False))
    print(f"\nwrote {out/'ABBREV_FULLSCALE_RESULTS.md'} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
