"""Evaluate SapBERT dense retrieval (#1) as a candidate-generation source.

Recall-ceiling test on the fixed sample. Compares the prior best lexical/char
pool against dense-only and hybrid+dense pools, stratified by all / tail /
acronym slice. The question: does dense retrieval put gold concepts into the
pool that string methods (lexical, char, synonym, acronym) structurally cannot
reach, especially in the tail?

Requires the SapBERT index built by build_dense_index.py.

    python benchmark/candidate_generation_experiments/test_dense_retrieval.py
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

HERE = Path(__file__).resolve().parent


def has_acronym(span):
    return bool(acronym_tokens(span))


def dense_concepts_for(mentions, retriever, k):
    """concept lists (no scores) per mention, from dense retrieval of the span."""
    spans = [m.span for m in mentions]
    res = retriever.retrieve_batch(spans, k=k)
    return [[cid for cid, _ in r] for r in res]


def build_policy(base_mentions, dense_lists, mode, cap):
    """mode: 'dense' (dense only) or 'hybrid+dense' (base pool then dense fill)."""
    out = []
    for m, dense in zip(base_mentions, dense_lists):
        m2 = copy.deepcopy(m)
        if mode == "dense":
            m2.candidates = ordered_unique(dense, cap)
        else:  # hybrid+dense
            m2.candidates = ordered_unique(list(m.candidates) + dense, cap)
        out.append(m2)
    return out


def scope_subsets(mentions, train_freq):
    return [
        ("all_sample", mentions),
        ("tail", [m for m in mentions if freq_stratum(m.gold_concept, train_freq) == "tail"]),
        ("acronym", [m for m in mentions if has_acronym(m.span)]),
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
            "gold_reachability": len(hits) / len(subset),
            "recall_at_10": base.recall_at_k(subset, 10),
            "recall_at_30": base.recall_at_k(subset, 30),
            "recall_at_60": base.recall_at_k(subset, 60),
            "new_gold_found": sum(1 for k in hits if k not in train_reached),
            "mean_candidates": sum(sizes) / len(sizes),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-file", default=".sample600.json")
    parser.add_argument("--snomed-dir", default=str(base.DEFAULT_SNOMED_DIR))
    parser.add_argument("--dense-k", type=int, default=60)
    args = parser.parse_args()

    if not (Path(EMB_PATH).exists() and Path(META_PATH).exists()):
        raise SystemExit(f"Dense index missing. Run build_dense_index.py first.\n  {EMB_PATH}")

    print("loading data + mentions...")
    cg, base_mentions, train_freq = base.load_mentions(args.sample_file)
    train_reached = {(m.note_id, m.start) for m in base_mentions
                     if m.gold_concept in set(m.candidates)}

    print("loading ontology + hybrid base pool...")
    ontology = base.SnomedOntology(args.snomed_dir)
    _, wide_boosted = base.apply_booster(cg, ontology, args.snomed_dir,
                                         base_mentions, "boosted", 60, 40)
    char_retriever = base.CharNgramRetriever(wide_boosted)
    hybrid = base.apply_hybrid(base_mentions, wide_boosted, char_retriever, 60, 40, char_first=True)

    print("loading dense index + retrieving...")
    retriever = DenseRetriever()
    dense_lists = dense_concepts_for(base_mentions, retriever, args.dense_k)

    policies = {
        "hybrid_char_first60": hybrid,
        f"dense{args.dense_k}": build_policy(base_mentions, dense_lists, "dense", args.dense_k),
        "hybrid+dense_cap60": build_policy(hybrid, dense_lists, "hybrid+dense", 60),
        "hybrid+dense_cap90": build_policy(hybrid, dense_lists, "hybrid+dense", 90),
    }

    rows = []
    for name, mentions in policies.items():
        rows.extend(summarize(name, mentions, train_freq, train_reached))
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "dense_candidate_eval.csv", index=False)

    # gold newly reachable via dense (union cap90) vs hybrid
    hset = {(m.note_id, m.start): (m.gold_concept in set(m.candidates)) for m in hybrid}
    examples = []
    for m in policies["hybrid+dense_cap90"]:
        key = (m.note_id, m.start)
        if m.gold_concept in set(m.candidates) and not hset.get(key, False):
            examples.append({
                "span": m.span,
                "gold_term": wide_boosted.label(m.gold_concept),
                "freq_stratum": freq_stratum(m.gold_concept, train_freq),
                "gold_rank": m.candidates.index(m.gold_concept) + 1,
            })
    ex_df = pd.DataFrame(examples)
    ex_df.to_csv(HERE / "dense_new_gold_examples.csv", index=False)

    _write_report(df, ex_df, args)

    print("\n=== reachability by policy x scope ===")
    print(df.to_string(index=False, formatters={
        "gold_reachability": "{:.4f}".format, "recall_at_10": "{:.4f}".format,
        "recall_at_30": "{:.4f}".format, "recall_at_60": "{:.4f}".format,
        "mean_candidates": "{:.2f}".format}))
    if not ex_df.empty:
        print(f"\n{len(ex_df)} gold concepts newly reachable via dense (vs hybrid). "
              f"See dense_new_gold_examples.csv")
    print(f"\nsaved -> {HERE}")


def _write_report(df, ex_df, args):
    def block(scope):
        sub = df[df.scope == scope][["policy", "n", "gold_reachability",
                                     "recall_at_10", "recall_at_30", "recall_at_60",
                                     "new_gold_found", "mean_candidates"]]
        return base.df_to_markdown(sub)
    lines = [
        "# Dense Retrieval (#1, SapBERT) Results", "",
        f"Sample: `{args.sample_file}`  ·  dense-k = {args.dense_k}  ·  "
        "encoder = cambridgeltl/SapBERT-from-PubMedBERT-fulltext", "",
        "## All sample", "", block("all_sample"),
        "", "## Tail", "", block("tail"),
        "", "## Acronym slice", "", block("acronym"),
        "", "## Gold newly reachable via dense (vs hybrid)", "",
        base.df_to_markdown(ex_df) if not ex_df.empty else "_None._",
    ]
    (HERE / "DENSE_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
