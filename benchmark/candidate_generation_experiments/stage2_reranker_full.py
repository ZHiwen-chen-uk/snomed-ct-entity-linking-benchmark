"""Full-scale version of the supervised-reranker negative result.

stage2_reranker.py trained on a 6,000-mention weighted sample of train notes and
evaluated on the fixed 600. Only one evaluation protocol is reported, so this
rebuilds the same experiment at full scale:

    train : ALL 52,078 train-note annotations (no sampling, no tail oversampling)
    eval  : the canonical 18,676-mention eval split (same split as stage2_gated /
            stage2_unified, so the reranker row is directly comparable to the
            oracle / gated / cross-encoder rows)

Same 13 features, same HistGradientBoosting model, same RRF@60 pool. The claim
under test is unchanged: a reranker trained on in-domain annotations inherits a
training set in which the gold concept is ALWAYS in the index (1.000 in every
stratum) and therefore cannot generalise to the test tail, where it is in the
index only 48.5% of the time.

Everything expensive is cached and the script is resumable, dense retrieval,
transformer-A type probabilities and the two feature matrices are each written
to disk as soon as they are complete, so a killed run resumes at the last stage.

    KMP_DUPLICATE_LIB_OK=TRUE python stage2_reranker_full.py
"""
import os
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

from candidate_booster import SnomedCandidateBooster
from data import CandidateGenerator, build_test_mentions, load, split_notes
from dense_retriever import DenseRetriever
from evaluate import freq_stratum
from ontology import SnomedOntology
from retrieval_fusion import rrf_fuse
import test_candidate_generation as tcg
from stage2_reranker import (RRF_WEIGHTS, TypeProba, build_mentions, features,
                             retriever_scores, strat_acc)

K = CAP = RRF_K = 60
SEED = 20260723
VAL_FRAC = 0.2
SNOMED_DIR = str(BM.parent / "SnomedCT_InternationalRF2_PRODUCTION_20260601T120000Z")
TFMR_DIR = BM / "group_a_transformer" / "outputs" / "pubmedbert_semtype" / "best_model"

DENSE_CACHE = HERE / ".rrfull_dense.pkl"
FEAT_TRAIN = HERE / ".rrfull_feat_train.npz"
FEAT_EVAL = HERE / ".rrfull_feat_eval.npz"
FNAMES = ["dense", "dense_rr", "lexical", "lex_rr", "char", "char_rr", "synonym",
          "train_rr", "prior", "log_freq", "in_train", "depth", "typeA"]


def dense_for(mentions, tag, t0):
    """Dense top-K per mention, cached across runs and appended incrementally."""
    cache = {}
    if DENSE_CACHE.exists():
        with open(DENSE_CACHE, "rb") as f:
            cache = pickle.load(f)
    todo = [m for m in mentions if (m.note_id, m.start) not in cache]
    if not todo:
        print(f"  dense[{tag}] fully cached", flush=True)
        return cache
    print(f"  dense[{tag}] {len(todo)} to compute", flush=True)
    dense = DenseRetriever()
    spans = [m.span for m in todo]
    for s in range(0, len(spans), 1000):
        res = dense.retrieve_batch(spans[s:s + 1000], k=K)
        for j, r in enumerate(res):
            m = todo[s + j]
            cache[(m.note_id, m.start)] = r
        with open(DENSE_CACHE, "wb") as f:
            pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"    dense[{tag}] {min(s+1000,len(spans))}/{len(spans)} ({time.time()-t0:.0f}s)",
              flush=True)
    return cache


def tprob_for(mentions, tag, t0, chunk=4000):
    """Transformer-A coarse-type probabilities; chunked so the disk cache is
    written every `chunk` mentions rather than only at the very end."""
    tp = TypeProba(TFMR_DIR)
    todo = [m for m in mentions if (m.note_id, m.start) not in tp.cache]
    print(f"  tprob[{tag}] {len(todo)} to compute", flush=True)
    for s in range(0, len(todo), chunk):
        tp.proba(todo[s:s + chunk])
        print(f"    tprob[{tag}] {min(s+chunk,len(todo))}/{len(todo)} ({time.time()-t0:.0f}s)",
              flush=True)
    return {(m.note_id, m.start): tp.cache[(m.note_id, m.start)] for m in mentions}


def feats(mentions, path, tag, booster, char, dense_map, cg, ontology, tprob, t0):
    if path.exists():
        z = np.load(path, allow_pickle=True)
        print(f"  features[{tag}] loaded from cache: {z['X'].shape}", flush=True)
        return z["X"], z["y"], list(z["meta"])
    X, y, meta = [], [], []
    for i, m in enumerate(mentions):
        key = (m.note_id, m.start)
        lists, ranks, lex_s, ch_s, dscore, synset = retriever_scores(
            m, booster, char, dense_map[key], K)
        pool = rrf_fuse(lists, k=RRF_K, cap=CAP, weights=RRF_WEIGHTS)
        for c in pool:
            X.append(features(m, c, ranks, lex_s, ch_s, dscore, synset,
                              cg, booster, ontology, tprob))
            y.append(1 if c == m.gold_concept else 0)
            meta.append((key[0], int(key[1]), int(c)))
        if (i + 1) % 5000 == 0:
            print(f"    features[{tag}] {i+1}/{len(mentions)} ({time.time()-t0:.0f}s)", flush=True)
    X = np.asarray(X, dtype=np.float32); y = np.asarray(y, dtype=np.int8)
    np.savez_compressed(path, X=X, y=y, meta=np.array(meta, dtype=object))
    print(f"  features[{tag}] {X.shape} pos={int(y.sum())} ({time.time()-t0:.0f}s)", flush=True)
    return X, y, meta


def main():
    t0 = time.time()
    notes, ann = load()
    train_notes, test_notes = split_notes(ann)
    cg = CandidateGenerator().fit(ann, train_notes)
    train_freq = Counter(int(c) for c in ann[ann.note_id.isin(train_notes)].concept_id)

    # ---- train = ALL train-note annotations, no sampling ---------------------
    tr_sub = ann[ann.note_id.isin(train_notes)]
    train_mentions = build_mentions(notes, tr_sub, cg)

    # ---- eval = canonical 18,676 split (identical construction to stage2_gated)
    test_mentions = build_test_mentions(notes, ann, test_notes, cg)
    tn = sorted(test_notes)
    random.Random(SEED).shuffle(tn)
    val_notes = set(tn[:max(1, int(len(tn) * VAL_FRAC))])
    ev = [m for m in test_mentions if m.note_id not in val_notes]

    tr_strata = Counter(freq_stratum(m.gold_concept, train_freq) for m in train_mentions)
    ev_strata = Counter(freq_stratum(m.gold_concept, train_freq) for m in ev)
    print(f"train={len(train_mentions)} {dict(tr_strata)}", flush=True)
    print(f"eval ={len(ev)} {dict(ev_strata)}", flush=True)

    print("building retrievers...", flush=True)
    ontology = SnomedOntology(SNOMED_DIR)
    booster = SnomedCandidateBooster(cg, ontology, SNOMED_DIR, mode="boosted",
                                     max_candidates=60, expand_k=40)
    char = tcg.CharNgramRetriever(booster)
    print(f"  retrievers ready ({time.time()-t0:.0f}s)", flush=True)

    print("dense retrieval...", flush=True)
    dense_map = dense_for(train_mentions + ev, "all", t0)

    print("transformer-A type probs...", flush=True)
    tprob = tprob_for(train_mentions + ev, "all", t0)

    print("features...", flush=True)
    Xtr, ytr, _ = feats(train_mentions, FEAT_TRAIN, "train", booster, char,
                        dense_map, cg, ontology, tprob, t0)
    Xev, yev, meta_ev = feats(ev, FEAT_EVAL, "eval", booster, char,
                              dense_map, cg, ontology, tprob, t0)

    print("training HistGradientBoosting...", flush=True)
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                         max_depth=6, l2_regularization=1.0,
                                         random_state=SEED)
    clf.fit(Xtr, ytr)
    print(f"  trained ({time.time()-t0:.0f}s)", flush=True)

    proba = clf.predict_proba(Xev)[:, 1]
    by_key = {}
    for (nid, st, c), p in zip(meta_ev, proba):
        by_key.setdefault((nid, st), []).append((int(c), float(p)))
    picks = {k: max(v, key=lambda kv: kv[1])[0] for k, v in by_key.items()}
    oracle = {(m.note_id, m.start): (m.gold_concept if m.gold_concept in
              {c for c, _ in by_key.get((m.note_id, m.start), [])} else None) for m in ev}

    rows = [{"ranker": "oracle (ceiling)", **strat_acc(oracle, ev, train_freq)},
            {"ranker": "supervised reranker", **strat_acc(picks, ev, train_freq)}]
    df = pd.DataFrame(rows).round(4)

    # permutation-free importance proxy, same as the pilot reported
    imp = []
    if hasattr(clf, "feature_importances_"):
        imp = sorted(zip(FNAMES, clf.feature_importances_), key=lambda kv: -kv[1])

    out = HERE / "stage2_out"; out.mkdir(exist_ok=True)
    df.to_csv(out / "stage2_reranker_full.csv", index=False)
    lines = [
        "# Stage 2 — supervised reranker, FULL SCALE", "",
        f"train = all {len(train_mentions)} train-note mentions "
        f"({dict(tr_strata)}) · eval = canonical split {len(ev)} "
        f"({dict(ev_strata)}) · features={len(FNAMES)} · pool = weighted RRF@{CAP}", "",
        tcg.df_to_markdown(df), "",
    ]
    if imp:
        lines += ["### Feature importance", "",
                  "| feature | importance |", "| --- | --- |"]
        lines += [f"| {n} | {v:.4f} |" for n, v in imp]
        lines.append("")
    (out / "STAGE2_RERANKER_FULL.md").write_text("\n".join(lines), encoding="utf-8")
    print(df.to_string(index=False))
    print(f"\nwrote {out/'STAGE2_RERANKER_FULL.md'} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
