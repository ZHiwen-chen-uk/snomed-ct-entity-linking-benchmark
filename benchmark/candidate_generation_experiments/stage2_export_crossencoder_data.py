"""Export cross-encoder training/inference pairs for #2 (run LOCALLY).

Produces two small CSVs so Colab only needs HF + the text pairs (no SNOMED/dense
/booster on Colab):

  ce_mentions.csv : key, split(train/test), val_or_eval, mention_text, gold,
                    freq_stratum
  ce_pairs.csv    : key, cand_id, cand_term, label   (label=1 if cand==gold)

text_a (mention) = "Section | <<span>> | marked-context"
text_b (candidate) = "preferred term (semantic tag)"

Train pairs come from tail-oversampled train-note mentions with RRF@60 pools
(hard negatives = the other pool candidates). Test pairs are the full 23k pool,
reusing the cached .stage2_pools_full.pkl. Leakage-free: only text vs text, no
prior/in_train features.

    python stage2_export_crossencoder_data.py           # TRAIN_N=8000 default
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

from candidate_booster import SnomedCandidateBooster, norm
from data import (CandidateGenerator, Mention, build_test_mentions, load, split_notes,
                  section_headers, section_at, CONTEXT_WINDOW)
from dense_retriever import DenseRetriever
from evaluate import freq_stratum
from ontology import SnomedOntology
from retrieval_fusion import rrf_fuse, exact_hits
import test_candidate_generation as tcg

RRF_WEIGHTS = {"synonym": 3.0, "train": 2.0, "lexical": 1.0, "char": 1.0, "dense": 1.5}
K = CAP = RRF_K = 60
SNOMED_DIR = str(BM.parent / "SnomedCT_InternationalRF2_PRODUCTION_20260601T120000Z")
POOL_CACHE = HERE / ".stage2_pools_full.pkl"
OUT = HERE / "crossencoder_data"
TRAIN_N = int(os.environ.get("TRAIN_N", "8000"))
VAL_FRAC = 0.2
SEED = 20260723


def mark(context, span):
    return context.replace(span, f"<< {span} >>", 1) if span else context


def mention_text(m):
    return f"Section: {m.section or 'NA'} | Mention: {m.span} | Context: {mark(m.context, m.span)}"


def cand_text(cid, ontology, booster):
    term = ontology.preferred_term(cid)
    if not term or str(term).startswith("concept:"):
        term = booster.label(cid)
    tag = ontology.semantic_type(cid)
    return f"{term} ({tag})" if tag and tag != "unknown" else str(term)


def build_mentions(notes, ann_sub, cg):
    text_by_id = dict(zip(notes.note_id, notes.text)); hdr = {}
    out = []
    for row in ann_sub.itertuples():
        text = str(text_by_id.get(row.note_id, ""))
        if row.note_id not in hdr:
            hdr[row.note_id] = section_headers(text)
        lo, hi = max(0, row.start - CONTEXT_WINDOW), min(len(text), row.end + CONTEXT_WINDOW)
        out.append(Mention(note_id=row.note_id, start=int(row.start), end=int(row.end),
                           span=str(row.span), gold_concept=int(row.concept_id),
                           context=text[lo:hi], section=section_at(hdr[row.note_id], row.start),
                           candidates=cg.generate(str(row.span))))
    return out


def main():
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    notes, ann = load()
    train_notes, test_notes = split_notes(ann)
    cg = CandidateGenerator().fit(ann, train_notes)
    train_freq = Counter(int(c) for c in ann[ann.note_id.isin(train_notes)].concept_id)

    test_mentions = build_test_mentions(notes, ann, test_notes, cg)
    tr_pool = ann[ann.note_id.isin(train_notes)].copy()
    w = tr_pool.concept_id.map(lambda c: 1.0 / np.sqrt(train_freq.get(int(c), 1) + 1.0))
    tr_sub = tr_pool.sample(n=min(TRAIN_N, len(tr_pool)), weights=w, random_state=SEED)
    train_mentions = build_mentions(notes, tr_sub, cg)
    print(f"train={len(train_mentions)} test={len(test_mentions)}", flush=True)

    ontology = SnomedOntology(SNOMED_DIR)
    booster = SnomedCandidateBooster(cg, ontology, SNOMED_DIR, mode="boosted",
                                     max_candidates=60, expand_k=40)
    char = tcg.CharNgramRetriever(booster)
    dense = DenseRetriever()

    # test pools from cache; train pools built here
    with open(POOL_CACHE, "rb") as f:
        cached = pickle.load(f)
    test_pools = cached["pools"]

    def build_pools(ms):
        spans = [m.span for m in ms]; dmap = {}
        for s in range(0, len(spans), 1000):
            for j, r in enumerate(dense.retrieve_batch(spans[s:s + 1000], k=K)):
                dmap[s + j] = [c for c, _ in r]
            print(f"  train dense {min(s+1000,len(spans))}/{len(spans)} ({time.time()-t0:.0f}s)", flush=True)
        pools = {}
        for i, m in enumerate(ms):
            lex, _ = booster.retrieve([m.span], K)
            ch, _ = char.retrieve(m.span, K)
            lists = {"train": list(m.candidates), "synonym": exact_hits(booster, m.span),
                     "lexical": lex, "char": ch, "dense": dmap[i]}
            pools[(m.note_id, m.start)] = rrf_fuse(lists, k=RRF_K, cap=CAP, weights=RRF_WEIGHTS)
        return pools
    print("building train pools...", flush=True)
    train_pools = build_pools(train_mentions)

    # val/eval note split for test (consistent with earlier runs)
    tn = sorted(test_notes); import random; random.Random(SEED).shuffle(tn)
    val_notes = set(tn[:max(1, int(len(tn) * VAL_FRAC))])

    term_cache = {}
    def term(cid):
        if cid not in term_cache:
            term_cache[cid] = cand_text(cid, ontology, booster)
        return term_cache[cid]

    m_rows, p_rows = [], []
    def emit(ms, pools, split):
        for m in ms:
            key = f"{m.note_id}|{m.start}"
            pool = pools[(m.note_id, m.start)]
            voe = ("val" if m.note_id in val_notes else "eval") if split == "test" else ""
            m_rows.append({"key": key, "split": split, "val_or_eval": voe,
                           "mention_text": mention_text(m), "gold": m.gold_concept,
                           "freq_stratum": freq_stratum(m.gold_concept, train_freq)})
            for c in pool:
                p_rows.append({"key": key, "cand_id": int(c), "cand_term": term(c),
                               "label": int(c == m.gold_concept)})
    emit(train_mentions, train_pools, "train")
    emit(test_mentions, test_pools, "test")

    pd.DataFrame(m_rows).to_csv(OUT / "ce_mentions.csv", index=False)
    pd.DataFrame(p_rows).to_csv(OUT / "ce_pairs.csv", index=False)
    n_train_m = sum(1 for m in m_rows if m["split"] == "train")
    n_train_pos = sum(r["label"] for r in p_rows)  # positives exist only where gold in pool
    print(f"mentions={len(m_rows)} (train={n_train_m}) pairs={len(p_rows)} positives={n_train_pos}", flush=True)
    print(f"DONE {time.time()-t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
