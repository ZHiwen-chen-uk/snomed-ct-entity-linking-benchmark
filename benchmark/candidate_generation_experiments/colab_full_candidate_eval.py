"""Full-dataset candidate-generation evaluation for Colab (GPU).

Runs the whole retrieval stack -- train / synonym / lexical / char / dense
(SapBERT) -- and weighted RRF fusion over ALL 23k held-out test mentions (not
just the fixed 600), so long-tail reachability/recall is measured with real
statistical power (thousands of tail mentions instead of ~86).

Why Colab: SapBERT encoding of ~665k SNOMED forms is minutes on GPU (hours on
CPU), and dense search over 23k queries is instant with FAISS-GPU.

No SNOMED RF2 release needed: the lexical indexes are shipped as cached pickles
and a stub ontology supplies concept labels from the label cache.

Expected upload layout (see COLAB_CANDIDATE_RUNBOOK.md):
    benchmark/.snomed_synindex.pkl
    benchmark/.snomed_tokenindex.pkl
    benchmark/.snomed_concept_labels.pkl
    benchmark/<code>.py ...
    snomed-ct-entity-linking-challenge-.../train_notes.csv, train_annotations.csv

Run (from the benchmark/candidate_generation_experiments dir on Colab):
    pip install faiss-gpu transformers
    python colab_full_candidate_eval.py --out-dir /content/drive/MyDrive/cand_full
"""
import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
sys.path.insert(0, str(BENCHMARK))
sys.path.insert(0, str(HERE))

from candidate_booster import SnomedCandidateBooster, build_synindex
from data import CandidateGenerator, build_test_mentions, load, split_notes
from evaluate import freq_stratum
from ontology import Ontology
from retrieval_fusion import per_retriever_lists, rrf_fuse
import test_candidate_generation as tcg  # CharNgramRetriever, recall_at_k, df_to_markdown

RRF_WEIGHTS = {"synonym": 3.0, "train": 2.0, "lexical": 1.0, "char": 1.0, "dense": 1.5}


class StubOntology(Ontology):
    """No RF2 release on Colab; labels come from the booster's concept_labels
    cache (booster.label falls back when preferred_term starts with 'concept:')."""
    def semantic_type(self, concept):
        return "unknown"
    def preferred_term(self, concept):
        return f"concept:{concept}"
    def depth(self, concept):
        return 0
    def distance(self, c1, c2):
        return 40


# ---------------------------------------------------------------------------
# dense: SapBERT encode (GPU) + FAISS inner-product search
# ---------------------------------------------------------------------------
def build_dense(syn_index, min_len=2, max_len=120, batch_size=512, device="cuda"):
    import faiss
    import torch
    from transformers import AutoModel, AutoTokenizer
    name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(device).eval()

    forms, form_concepts = [], []
    for form, concepts in syn_index.items():
        txt = str(form).strip()
        if min_len <= len(txt) <= max_len:
            forms.append(txt)
            form_concepts.append(tuple(int(c) for c in concepts))
    n = len(forms)
    print(f"[dense] encoding {n} forms on {device}...", flush=True)

    emb = np.empty((n, 768), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = forms[i : i + batch_size]
            t = tok(batch, padding=True, truncation=True, max_length=32,
                    return_tensors="pt").to(device)
            cls = model(**t).last_hidden_state[:, 0, :]
            cls = torch.nn.functional.normalize(cls, p=2, dim=1)
            emb[i : i + len(batch)] = cls.cpu().numpy()
            if i % (batch_size * 100) == 0:
                print(f"  {i}/{n}  ({time.time()-t0:.0f}s)", flush=True)
    index = faiss.IndexFlatIP(768)
    index.add(emb)
    return index, forms, form_concepts, tok, model


def dense_retrieve_all(spans, index, form_concepts, tok, model, k=60,
                       top_forms=400, batch_size=512, device="cuda"):
    import torch
    out = {}
    with torch.no_grad():
        for i in range(0, len(spans), batch_size):
            batch = spans[i : i + batch_size]
            t = tok(batch, padding=True, truncation=True, max_length=32,
                    return_tensors="pt").to(device)
            q = torch.nn.functional.normalize(
                model(**t).last_hidden_state[:, 0, :], p=2, dim=1).cpu().numpy()
            D, I = index.search(q, top_forms)
            for r, row in enumerate(I):
                cs = {}
                for j, form_row in enumerate(row):
                    for cid in form_concepts[form_row]:
                        cid = int(cid)
                        s = float(D[r, j])
                        if cid not in cs or s > cs[cid]:
                            cs[cid] = s
                    if len(cs) >= k * 4:
                        break
                top = sorted(cs.items(), key=lambda kv: kv[1], reverse=True)[:k]
                out[i + r] = [c for c, _ in top]
    return out


# ---------------------------------------------------------------------------
def scope_subsets(mentions, train_freq):
    return [("all", mentions)] + [
        (s, [m for m in mentions if freq_stratum(m.gold_concept, train_freq) == s])
        for s in ("head", "torso", "tail")
    ]


def summarize(name, mentions, train_freq):
    rows = []
    for scope, subset in scope_subsets(mentions, train_freq):
        if not subset:
            continue
        reach = sum(1 for m in subset if m.gold_concept in set(m.candidates)) / len(subset)
        rows.append({
            "policy": name, "scope": scope, "n": len(subset), "reach": reach,
            "r@1": tcg.recall_at_k(subset, 1), "r@10": tcg.recall_at_k(subset, 10),
            "r@30": tcg.recall_at_k(subset, 30), "r@60": tcg.recall_at_k(subset, 60),
            "mean_cand": np.mean([len(m.candidates) for m in subset]),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(HERE / "full_eval_out"))
    ap.add_argument("--k", type=int, default=60)
    ap.add_argument("--cap", type=int, default=60)
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="debug: cap #mentions")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    print("loading data + full test mentions...", flush=True)
    notes, ann = load()
    train_notes, test_notes = split_notes(ann)
    cg = CandidateGenerator().fit(ann, train_notes)
    mentions = build_test_mentions(notes, ann, test_notes, cg)
    if args.limit:
        mentions = mentions[: args.limit]
    train_freq = Counter(int(c) for c in ann[ann.note_id.isin(train_notes)].concept_id)
    print(f"test mentions: {len(mentions)}", flush=True)

    print("building lexical retrievers (from cached indexes)...", flush=True)
    booster = SnomedCandidateBooster(cg, StubOntology(), release_dir=".",
                                     mode="boosted", max_candidates=60, expand_k=40)
    char_retriever = tcg.CharNgramRetriever(booster)

    print("building dense index (SapBERT, GPU)...", flush=True)
    index, forms, form_concepts, tok, model = build_dense(booster.syn_index, device=args.device)
    dense_by_i = dense_retrieve_all([m.span for m in mentions], index, form_concepts,
                                    tok, model, k=args.k, device=args.device)

    print("fusing (weighted RRF) + baselines...", flush=True)
    import copy
    pol = {"concat": [], "rrf_weighted": []}
    for i, m in enumerate(mentions):
        lists = per_retriever_lists(m, booster, char_retriever, dense_by_i.get(i, []), args.k)
        from candidate_booster import ordered_unique
        c1 = copy.copy(m); c1.candidates = ordered_unique(
            lists["train"] + lists["synonym"] + lists["char"] + lists["lexical"] + lists["dense"], args.cap)
        c2 = copy.copy(m); c2.candidates = rrf_fuse(lists, k=args.rrf_k, cap=args.cap, weights=RRF_WEIGHTS)
        pol["concat"].append(c1); pol["rrf_weighted"].append(c2)

    rows = []
    for name, ms in pol.items():
        rows.extend(summarize(name, ms, train_freq))
    df = pd.DataFrame(rows)
    df.to_csv(out / "full_candidate_eval.csv", index=False)

    lines = ["# Full-Dataset Candidate Generation (23k test mentions)", "",
             f"per-retriever depth k={args.k} · cap={args.cap} · RRF weights {RRF_WEIGHTS}", ""]
    for scope in ("all", "head", "torso", "tail"):
        sub = df[df.scope == scope][["policy", "n", "reach", "r@1", "r@10", "r@30", "r@60", "mean_cand"]]
        lines += [f"## {scope}", "", tcg.df_to_markdown(sub), ""]
    (out / "FULL_CANDIDATE_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(df.to_string(index=False))
    print(f"\nsaved -> {out}", flush=True)


if __name__ == "__main__":
    main()
