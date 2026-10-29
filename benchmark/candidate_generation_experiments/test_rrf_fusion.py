"""Validate RRF fusion locally on the fixed sample.

Compares the prior naive concatenation pool against RRF over
{train, synonym, lexical, char, dense}. The question: does RRF surface the
dense-only tail golds (buried at rank 60+ by concatenation) into recall@30?

    python benchmark/candidate_generation_experiments/test_rrf_fusion.py
"""
import argparse
import copy
from pathlib import Path

import pandas as pd

import test_candidate_generation as base
from abbrev_expander import acronym_tokens
from candidate_booster import ordered_unique
from dense_retriever import DenseRetriever, EMB_PATH, META_PATH
from evaluate import freq_stratum
from retrieval_fusion import per_retriever_lists, rrf_fuse

HERE = Path(__file__).resolve().parent


def scope_subsets(mentions, train_freq):
    return [
        ("all_sample", mentions),
        ("head", [m for m in mentions if freq_stratum(m.gold_concept, train_freq) == "head"]),
        ("torso", [m for m in mentions if freq_stratum(m.gold_concept, train_freq) == "torso"]),
        ("tail", [m for m in mentions if freq_stratum(m.gold_concept, train_freq) == "tail"]),
        ("acronym", [m for m in mentions if acronym_tokens(m.span)]),
    ]


def summarize(name, mentions, train_freq, train_reached):
    rows = []
    for scope, subset in scope_subsets(mentions, train_freq):
        if not subset:
            continue
        hits = {(m.note_id, m.start) for m in subset if m.gold_concept in set(m.candidates)}
        sizes = [len(m.candidates) for m in subset]
        rows.append({
            "policy": name, "scope": scope, "n": len(subset),
            "reach": len(hits) / len(subset),
            "r@10": base.recall_at_k(subset, 10),
            "r@30": base.recall_at_k(subset, 30),
            "r@60": base.recall_at_k(subset, 60),
            "new_gold": sum(1 for k in hits if k not in train_reached),
            "mean_cand": sum(sizes) / len(sizes),
        })
    return rows


def with_candidates(base_mentions, cand_by_key):
    out = []
    for m in base_mentions:
        m2 = copy.deepcopy(m)
        m2.candidates = cand_by_key[(m.note_id, m.start)]
        out.append(m2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-file", default=".sample600.json")
    ap.add_argument("--snomed-dir", default=str(base.DEFAULT_SNOMED_DIR))
    ap.add_argument("--k", type=int, default=60, help="per-retriever depth")
    ap.add_argument("--cap", type=int, default=60, help="final pool cap")
    ap.add_argument("--rrf-k", type=int, default=60)
    args = ap.parse_args()

    if not (Path(EMB_PATH).exists() and Path(META_PATH).exists()):
        raise SystemExit("Dense index missing. Run build_dense_index.py first.")

    print("loading data + mentions...")
    cg, base_mentions, train_freq = base.load_mentions(args.sample_file)
    train_reached = {(m.note_id, m.start) for m in base_mentions
                     if m.gold_concept in set(m.candidates)}

    print("loading ontology + retrievers...")
    ontology = base.SnomedOntology(args.snomed_dir)
    _, wide_boosted = base.apply_booster(cg, ontology, args.snomed_dir,
                                         base_mentions, "boosted", 60, 40)
    char_retriever = base.CharNgramRetriever(wide_boosted)
    retriever = DenseRetriever()

    print("dense retrieval (batched)...")
    dense_res = retriever.retrieve_batch([m.span for m in base_mentions], k=args.k)
    dense_map = {(m.note_id, m.start): [c for c, _ in r]
                 for m, r in zip(base_mentions, dense_res)}

    print("building per-retriever lists + fusions...")
    concat, rrf_all, rrf_nodense, rrf_w = {}, {}, {}, {}
    for m in base_mentions:
        key = (m.note_id, m.start)
        lists = per_retriever_lists(m, wide_boosted, char_retriever, dense_map[key], args.k)
        # naive concatenation (prior hybrid+dense, dense last)
        concat[key] = ordered_unique(
            lists["train"] + lists["synonym"] + lists["char"] + lists["lexical"] + lists["dense"],
            args.cap)
        rrf_all[key] = rrf_fuse(lists, k=args.rrf_k, cap=args.cap)
        rrf_w[key] = rrf_fuse(lists, k=args.rrf_k, cap=args.cap,
                              weights={"synonym": 3.0, "train": 2.0,
                                       "lexical": 1.0, "char": 1.0, "dense": 1.5})
        rrf_nodense[key] = rrf_fuse(
            {n: lists[n] for n in ("train", "synonym", "lexical", "char")},
            k=args.rrf_k, cap=args.cap)

    policies = {
        "concat_hybrid+dense": with_candidates(base_mentions, concat),
        "rrf_lexical_only": with_candidates(base_mentions, rrf_nodense),
        "rrf_all(+dense)": with_candidates(base_mentions, rrf_all),
        "rrf_weighted(+dense)": with_candidates(base_mentions, rrf_w),
    }

    rows = []
    for name, ms in policies.items():
        rows.extend(summarize(name, ms, train_freq, train_reached))
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "rrf_fusion_eval.csv", index=False)

    lines = ["# RRF Fusion Results", "",
             f"Sample `{args.sample_file}` · per-retriever depth k={args.k} · "
             f"final cap={args.cap} · rrf_k={args.rrf_k}", ""]
    for scope in ("all_sample", "head", "torso", "tail", "acronym"):
        sub = df[df.scope == scope][["policy", "n", "reach", "r@10", "r@30", "r@60",
                                     "new_gold", "mean_cand"]]
        lines += [f"## {scope}", "", base.df_to_markdown(sub), ""]
    (HERE / "RRF_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n=== RRF fusion (policy x scope) ===")
    print(df.to_string(index=False, formatters={
        "reach": "{:.4f}".format, "r@10": "{:.4f}".format,
        "r@30": "{:.4f}".format, "r@60": "{:.4f}".format, "mean_cand": "{:.1f}".format}))
    print(f"\nsaved -> {HERE}")


if __name__ == "__main__":
    main()
