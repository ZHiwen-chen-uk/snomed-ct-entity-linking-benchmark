"""Stage 2: a learned reranker over the RRF@60 pool.

The cheap-ranker probe showed the signals are complementary, train prior wins
head, dense (SapBERT) wins tail, neither wins both, and a large tail gap to the
oracle ceiling (best cheap tail 0.57 vs 0.895). This trains a pointwise reranker
that combines them, so head keeps its prior and tail gets its semantic signal.

Trained on train-note mentions (train-only retrievers, no test leakage),
evaluated on the fixed 600 test mentions, stratified head/torso/tail.

Features per (mention, candidate):
  retrieval : dense cosine, dense/lexical/char reciprocal-rank, lexical/char
              score, exact-synonym flag, RRF score
  prior     : train prior P(concept|span), log global freq, seen-in-train flag
  ontology  : candidate depth, coarse-type one-hot
  type (A)  : transformer-A P(candidate coarse type | mention)   [soft Group A]

    python stage2_reranker.py                 # default 6000 train mentions
    TRAIN_N=3000 python stage2_reranker.py
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

from candidate_booster import SnomedCandidateBooster, norm
from data import (CandidateGenerator, Mention, load, split_notes,
                  section_headers, section_at, CONTEXT_WINDOW)
from dense_retriever import DenseRetriever
from evaluate import freq_stratum
from ontology import SnomedOntology, COARSE_LABELS
from retrieval_fusion import rrf_fuse, exact_hits
import test_candidate_generation as tcg
from transformer_predictor import transformer_text

RRF_WEIGHTS = {"synonym": 3.0, "train": 2.0, "lexical": 1.0, "char": 1.0, "dense": 1.5}
K = CAP = RRF_K = 60
SNOMED_DIR = str(BM.parent / "SnomedCT_InternationalRF2_PRODUCTION_20260601T120000Z")
TFMR_DIR = BM / "group_a_transformer" / "outputs" / "pubmedbert_semtype" / "best_model"
TRAIN_N = int(os.environ.get("TRAIN_N", "6000"))
SEED = 20260723
TYPE4 = {"body structure", "finding", "procedure"}   # transformer-A labels (+ "other")


# --------------------------------------------------------------------------
def build_mentions(notes, ann_sub, cg):
    text_by_id = dict(zip(notes.note_id, notes.text))
    hdr = {}
    out = []
    for row in ann_sub.itertuples():
        text = str(text_by_id.get(row.note_id, ""))
        if row.note_id not in hdr:
            hdr[row.note_id] = section_headers(text)
        lo, hi = max(0, row.start - CONTEXT_WINDOW), min(len(text), row.end + CONTEXT_WINDOW)
        out.append(Mention(
            note_id=row.note_id, start=int(row.start), end=int(row.end),
            span=str(row.span), gold_concept=int(row.concept_id),
            context=text[lo:hi], section=section_at(hdr[row.note_id], row.start),
            candidates=cg.generate(str(row.span))))
    return out


TPROB_CACHE = HERE / ".stage2_typeprob_cache.pkl"


class TypeProba:
    """Batched transformer-A coarse-type probabilities (4 labels), disk-cached."""
    def __init__(self, model_dir, batch=64, max_length=192):
        import json
        self.model_dir = model_dir
        self.labels = json.loads((Path(model_dir) / "label_map.json").read_text())["labels"]
        self.batch = batch
        self.max_length = max_length
        self.cache = {}
        if TPROB_CACHE.exists():
            import pickle
            self.cache = pickle.load(open(TPROB_CACHE, "rb"))

    def proba(self, mentions):
        todo = [m for m in mentions if (m.note_id, m.start) not in self.cache]
        if todo:
            import torch, pickle
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            torch.set_num_threads(max(1, os.cpu_count() or 1))
            tok = AutoTokenizer.from_pretrained(str(self.model_dir), use_fast=True)
            model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir)).eval()
            texts = [transformer_text(m) for m in todo]
            with torch.no_grad():
                for i in range(0, len(texts), self.batch):
                    enc = tok(texts[i:i + self.batch], truncation=True, padding=True,
                              max_length=self.max_length, return_tensors="pt")
                    p = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
                    for j in range(p.shape[0]):
                        m = todo[i + j]
                        self.cache[(m.note_id, m.start)] = {
                            self.labels[l]: float(p[j, l]) for l in range(len(self.labels))}
            pickle.dump(self.cache, open(TPROB_CACHE, "wb"))
        return {(m.note_id, m.start): self.cache[(m.note_id, m.start)] for m in mentions}


def retriever_scores(m, booster, char, dense_pairs, k):
    lex, lex_s = booster.retrieve([m.span], k)
    ch, ch_s = char.retrieve(m.span, k)
    syn = exact_hits(booster, m.span)
    dense = [c for c, _ in dense_pairs]
    dscore = dict(dense_pairs)
    lists = {"train": list(m.candidates), "synonym": syn, "lexical": lex, "char": ch, "dense": dense}
    ranks = {n: {c: r for r, c in enumerate(lst)} for n, lst in lists.items()}
    return lists, ranks, lex_s, ch_s, dscore, set(syn)


def features(m, cand, ranks, lex_s, ch_s, dscore, synset, cg, booster, ontology, tprob):
    def rr(name):
        r = ranks[name].get(cand)
        return 1.0 / (r + 1) if r is not None else 0.0
    ct = ontology.coarse_type(cand)
    ct4 = ct if ct in TYPE4 else "other"
    return [
        dscore.get(cand, 0.0), rr("dense"),
        lex_s.get(cand, 0.0), rr("lexical"),
        ch_s.get(cand, 0.0), rr("char"),
        1.0 if cand in synset else 0.0,
        rr("train"),
        cg.prior(m.span, cand),
        np.log1p(booster.global_freq.get(cand, 0)),
        1.0 if cand in booster.global_freq else 0.0,
        float(ontology.depth(cand)),
        tprob.get((m.note_id, m.start), {}).get(ct4, 0.0),
    ]


def build_xy(mentions, booster, char, dense_map, cg, ontology, tprob, k):
    X, y, groups, meta = [], [], [], []
    for m in mentions:
        key = (m.note_id, m.start)
        lists, ranks, lex_s, ch_s, dscore, synset = retriever_scores(
            m, booster, char, dense_map[key], k)
        pool = rrf_fuse(lists, k=RRF_K, cap=CAP, weights=RRF_WEIGHTS)
        for c in pool:
            X.append(features(m, c, ranks, lex_s, ch_s, dscore, synset, cg, booster, ontology, tprob))
            y.append(1 if c == m.gold_concept else 0)
            meta.append((key, c))
        groups.append((key, pool))
    return np.array(X, dtype=np.float32), np.array(y), groups, meta


def strat_acc(picks, mentions, train_freq):
    def a(sub):
        return None if not sub else sum(
            1 for m in sub if picks.get((m.note_id, m.start)) == m.gold_concept) / len(sub)
    return {"all": a(mentions),
            **{s: a([m for m in mentions if freq_stratum(m.gold_concept, train_freq) == s])
               for s in ("head", "torso", "tail")}}


def main():
    t0 = time.time()
    print("loading data...", flush=True)
    notes, ann = load()
    train_notes, test_notes = split_notes(ann)
    cg = CandidateGenerator().fit(ann, train_notes)
    train_freq = Counter(int(c) for c in ann[ann.note_id.isin(train_notes)].concept_id)

    # test = fixed 600
    import json
    want = {(d["note_id"], int(d["start"])) for d in json.loads((BM / ".sample600.json").read_text())}
    test_sub = ann[(ann.note_id.isin(test_notes)) & (ann.annotation_type == "test")]
    test_sub = test_sub[[(r.note_id, int(r.start)) in want for r in test_sub.itertuples()]]
    test_mentions = build_mentions(notes, test_sub, cg)

    # train = sample of train-note annotations, OVERSAMPLING low-frequency (tail-like)
    # gold so the reranker learns the no-prior regime, not just the head prior.
    tr_pool = ann[ann.note_id.isin(train_notes)].copy()
    w = tr_pool.concept_id.map(lambda c: 1.0 / np.sqrt(train_freq.get(int(c), 1) + 1.0))
    tr_sub = tr_pool.sample(n=min(TRAIN_N, len(tr_pool)), weights=w, random_state=SEED)
    train_mentions = build_mentions(notes, tr_sub, cg)
    tr_strata = Counter(freq_stratum(m.gold_concept, train_freq) for m in train_mentions)
    print(f"train={len(train_mentions)} {dict(tr_strata)}  test={len(test_mentions)}", flush=True)

    print("building retrievers...", flush=True)
    ontology = SnomedOntology(SNOMED_DIR)
    booster = SnomedCandidateBooster(cg, ontology, SNOMED_DIR, mode="boosted",
                                     max_candidates=60, expand_k=40)
    char = tcg.CharNgramRetriever(booster)
    dense = DenseRetriever()

    print("dense retrieval (train+test)...", flush=True)
    allm = train_mentions + test_mentions
    dense_map = {}
    spans = [m.span for m in allm]
    for s in range(0, len(spans), 1000):
        for j, r in enumerate(dense.retrieve_batch(spans[s:s + 1000], k=K)):
            dense_map[(allm[s + j].note_id, allm[s + j].start)] = r
        print(f"  dense {min(s+1000,len(spans))}/{len(spans)} ({time.time()-t0:.0f}s)", flush=True)

    print("transformer-A type probs...", flush=True)
    tp = TypeProba(TFMR_DIR)
    tprob = tp.proba(allm)
    print(f"  done ({time.time()-t0:.0f}s)", flush=True)

    print("features...", flush=True)
    Xtr, ytr, _, _ = build_xy(train_mentions, booster, char, dense_map, cg, ontology, tprob, K)
    Xte, yte, groups_te, meta_te = build_xy(test_mentions, booster, char, dense_map, cg, ontology, tprob, K)
    print(f"  train rows={len(ytr)} pos={int(ytr.sum())} | test rows={len(yte)} ({time.time()-t0:.0f}s)", flush=True)

    print("training reranker (HistGradientBoosting)...", flush=True)
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                         max_depth=6, l2_regularization=1.0,
                                         random_state=SEED)
    clf.fit(Xtr, ytr)

    # score test pools, pick argmax per mention
    proba = clf.predict_proba(Xte)[:, 1]
    by_key = {}
    for (key, c), p in zip(meta_te, proba):
        by_key.setdefault(key, []).append((c, p))
    picks = {key: max(cs, key=lambda kv: kv[1])[0] for key, cs in by_key.items()}

    oracle = {(m.note_id, m.start): (m.gold_concept if m.gold_concept in
              {c for c, _ in by_key.get((m.note_id, m.start), [])} else None)
              for m in test_mentions}

    rows = [{"ranker": "oracle", **strat_acc(oracle, test_mentions, train_freq)},
            {"ranker": "reranker", **strat_acc(picks, test_mentions, train_freq)}]
    df = pd.DataFrame(rows)
    out = HERE / "stage2_out"; out.mkdir(exist_ok=True)
    df.to_csv(out / "stage2_reranker_accuracy.csv", index=False)

    fnames = ["dense", "dense_rr", "lexical", "lex_rr", "char", "char_rr", "synonym",
              "train_rr", "prior", "log_freq", "in_train", "depth", "typeA"]
    imp = sorted(zip(fnames, clf.feature_importances_ if hasattr(clf, "feature_importances_")
                     else [0]*len(fnames)), key=lambda kv: -kv[1]) \
        if hasattr(clf, "feature_importances_") else []

    lines = ["# Stage 2 — Learned Reranker over RRF@60 (fixed 600)", "",
             f"train mentions={len(train_mentions)} · features={len(fnames)}", "",
             tcg.df_to_markdown(df.round(4))]
    (out / "STAGE2_RERANKER_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n=== reranker vs oracle (RRF@60 pool) ===")
    print(df.to_string(index=False, formatters={c: "{:.4f}".format
          for c in ("all", "head", "torso", "tail")}))
    print(f"\nDONE {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
