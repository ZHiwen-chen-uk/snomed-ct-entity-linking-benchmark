"""Evaluate case-sensitive abbreviation expansion (#2) as a candidate-generation
add-on. Recall-ceiling test only; no downstream A/B/C ranking here.

For each policy we report gold reachability and recall@k on three scopes:
  * all_sample   overall effect
  * tail         long-tail effect (train frequency < 3)
  * acronym      the slice the intervention actually targets (mentions whose
                 span contains an acronym-shaped token) -- this is where the
                 gain, and any distractor cost, should show up.

Policies compare a base candidate generator with and without abbreviation
expansion, so the marginal contribution of #2 is isolated:

  synonym45                 synonym-exact base
  synonym45+abbrev          + case-sensitive acronym expansion
  hybrid_char_first60_k40   strong lexical+char base (prior best local reach)
  hybrid_char_first...+abbrev

Run:
    python benchmark/candidate_generation_experiments/test_abbrev_expansion.py
"""

import argparse
import copy
from collections import Counter
from pathlib import Path

import pandas as pd

# reuse the existing sandbox harness so candidate generation is identical
import test_candidate_generation as base
from abbrev_expander import AbbrevExpander, acronym_tokens
from candidate_booster import ordered_unique, norm
from data import load, split_notes
from evaluate import freq_stratum

HERE = Path(__file__).resolve().parent


def _abbrev_add(expander, booster, span, expand_k, mode):
    """Concepts contributed by the acronym layer for one span.

    precise : direct mined concepts + EXACT synonym lookup of expansion phrases
              (high precision, minimal distractors).
    fuzzy   : direct + full fuzzy retrieval over expansion phrases (recall-max
              but inflates the pool; kept for contrast).
    """
    direct, terms = expander.expand(span, labeler=booster.label)
    if not direct and not terms:
        return []
    if mode == "fuzzy":
        exp_hits, _ = booster.retrieve(terms, expand_k) if terms else ([], {})
        return direct + exp_hits
    exp_hits = []
    for t in terms:
        exp_hits.extend(base.exact_hits(booster, t))
    return direct + exp_hits


def apply_hybrid_abbrev(base_mentions, booster, char_retriever, expander,
                        max_candidates, expand_k, char_first=True,
                        use_abbrev=True, abbrev_mode="precise"):
    """Prior best hybrid pool, optionally with acronym expansion prepended."""
    mentions = copy.deepcopy(base_mentions)
    for m in mentions:
        base_cands = list(m.candidates)
        syn = base.exact_hits(booster, m.span)
        lex, _ = booster.retrieve([m.span], expand_k)
        ch, _ = char_retriever.retrieve(m.span, expand_k)
        add = _abbrev_add(expander, booster, m.span, expand_k, abbrev_mode) if use_abbrev else []
        order = base_cands + add + syn + ch + lex if char_first else base_cands + add + syn + lex + ch
        m.candidates = ordered_unique(order, max_candidates)
    return mentions


def apply_synonym_abbrev(base_mentions, booster, expander, max_candidates,
                         expand_k, use_abbrev=True, abbrev_mode="precise"):
    """Synonym-exact base, optionally with acronym expansion."""
    mentions = copy.deepcopy(base_mentions)
    for m in mentions:
        base_cands = list(m.candidates)
        syn = base.exact_hits(booster, m.span)
        add = _abbrev_add(expander, booster, m.span, expand_k, abbrev_mode) if use_abbrev else []
        m.candidates = ordered_unique(base_cands + add + syn, max_candidates)
    return mentions


def scope_subsets(mentions, train_freq, expander):
    subsets = [("all_sample", mentions)]
    subsets.append(("tail", [m for m in mentions
                             if freq_stratum(m.gold_concept, train_freq) == "tail"]))
    subsets.append(("acronym", [m for m in mentions if expander.has_acronym(m.span)]))
    return subsets


def summarize(name, mentions, train_freq, train_reached, expander):
    rows = []
    for scope, subset in scope_subsets(mentions, train_freq, expander):
        if not subset:
            continue
        hits = {(m.note_id, m.start) for m in subset
                if m.gold_concept in set(m.candidates)}
        sizes = [len(m.candidates) for m in subset]
        rows.append({
            "policy": name,
            "scope": scope,
            "n": len(subset),
            "gold_reachability": len(hits) / len(subset),
            "recall_at_10": base.recall_at_k(subset, 10),
            "recall_at_30": base.recall_at_k(subset, 30),
            "recall_at_45": base.recall_at_k(subset, 45),
            "new_gold_found": sum(1 for k in hits if k not in train_reached),
            "mean_candidates": sum(sizes) / len(sizes),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-file", default=".sample600.json")
    parser.add_argument("--snomed-dir", default=str(base.DEFAULT_SNOMED_DIR))
    args = parser.parse_args()

    print("loading data + mentions...")
    cg, base_mentions, train_freq = base.load_mentions(args.sample_file)
    train_reached = {(m.note_id, m.start) for m in base_mentions
                     if m.gold_concept in set(m.candidates)}

    _, ann = load()
    train_notes, _ = split_notes(ann)
    expander = AbbrevExpander.from_training(ann, train_notes)
    n_acr = sum(1 for m in base_mentions if expander.has_acronym(m.span))
    print(f"mentions={len(base_mentions)}  acronym-slice={n_acr}  "
          f"mined_acronyms={len(expander.mined)}  curated={len(expander.curated)}")

    print("loading ontology + SNOMED indexes...")
    ontology = base.SnomedOntology(args.snomed_dir)

    # bases (reuse the sandbox booster + char index exactly)
    _, syn_booster = base.apply_booster(cg, ontology, args.snomed_dir,
                                        base_mentions, "synonym", 45, 25)
    _, wide_boosted = base.apply_booster(cg, ontology, args.snomed_dir,
                                         base_mentions, "boosted", 60, 40)
    char_retriever = base.CharNgramRetriever(wide_boosted)

    policies = {}
    policies["synonym45"] = apply_synonym_abbrev(
        base_mentions, syn_booster, expander, 45, 25, use_abbrev=False)
    policies["synonym45+abbrev_precise"] = apply_synonym_abbrev(
        base_mentions, syn_booster, expander, 45, 25, abbrev_mode="precise")
    policies["synonym45+abbrev_fuzzy"] = apply_synonym_abbrev(
        base_mentions, syn_booster, expander, 45, 25, abbrev_mode="fuzzy")
    policies["hybrid_char_first60_k40"] = apply_hybrid_abbrev(
        base_mentions, wide_boosted, char_retriever, expander, 60, 40, use_abbrev=False)
    policies["hybrid_char_first60_k40+abbrev_precise"] = apply_hybrid_abbrev(
        base_mentions, wide_boosted, char_retriever, expander, 60, 40, abbrev_mode="precise")

    rows = []
    for name, mentions in policies.items():
        rows.extend(summarize(name, mentions, train_freq, train_reached, expander))
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "abbrev_candidate_eval.csv", index=False)

    # new-gold examples surfaced only by the acronym layer (hybrid+abbrev vs hybrid)
    abbrev_policy = policies["hybrid_char_first60_k40+abbrev_precise"]
    baseh = {(m.note_id, m.start): (m.gold_concept in set(m.candidates))
             for m in policies["hybrid_char_first60_k40"]}
    examples = []
    for m in abbrev_policy:
        key = (m.note_id, m.start)
        reach = m.gold_concept in set(m.candidates)
        if reach and not baseh.get(key, False):
            examples.append({
                "span": m.span,
                "acronym_tokens": ",".join(acronym_tokens(m.span)),
                "gold_term": wide_boosted.label(m.gold_concept),
                "freq_stratum": freq_stratum(m.gold_concept, train_freq),
                "gold_rank": m.candidates.index(m.gold_concept) + 1,
                "candidate_count": len(m.candidates),
            })
    ex_df = pd.DataFrame(examples)
    ex_df.to_csv(HERE / "abbrev_new_gold_examples.csv", index=False)

    # characterise the residual: acronym-slice mentions STILL unreachable
    misses = []
    for m in abbrev_policy:
        if not expander.has_acronym(m.span):
            continue
        if m.gold_concept in set(m.candidates):
            continue
        toks = acronym_tokens(m.span)
        misses.append({
            "span": m.span,
            "acronym_tokens": ",".join(toks),
            "in_mined": any(t in expander.mined for t in toks),
            "in_curated": any(t in expander.curated for t in toks),
            "gold_term": wide_boosted.label(m.gold_concept),
            "freq_stratum": freq_stratum(m.gold_concept, train_freq),
        })
    miss_df = pd.DataFrame(misses)
    miss_df.to_csv(HERE / "abbrev_acronym_slice_misses.csv", index=False)
    print(f"acronym-slice residual misses: {len(miss_df)} (see abbrev_acronym_slice_misses.csv)")

    _write_report(df, ex_df, n_acr, expander)

    print("\n=== reachability by policy x scope ===")
    print(df.to_string(index=False, formatters={
        "gold_reachability": "{:.4f}".format,
        "recall_at_10": "{:.4f}".format,
        "recall_at_30": "{:.4f}".format,
        "recall_at_45": "{:.4f}".format,
        "mean_candidates": "{:.2f}".format,
    }))
    if not ex_df.empty:
        print(f"\n{len(ex_df)} gold concepts newly reachable via acronym layer "
              f"(hybrid+abbrev vs hybrid). See abbrev_new_gold_examples.csv")
    print(f"\nsaved -> {HERE}")


def _write_report(df, ex_df, n_acr, expander):
    def block(scope):
        sub = df[df.scope == scope][
            ["policy", "n", "gold_reachability", "recall_at_10",
             "recall_at_30", "new_gold_found", "mean_candidates"]]
        return base.df_to_markdown(sub)

    lines = [
        "# Abbreviation Expansion (#2) Results",
        "",
        f"Acronym slice: {n_acr} of the fixed sample carry an acronym-shaped "
        f"token. Mined acronyms (train-only): {len(expander.mined)}. "
        f"Curated seed: {len(expander.curated)}.",
        "",
        "## All sample", "", block("all_sample"),
        "", "## Tail", "", block("tail"),
        "", "## Acronym slice (the targeted subset)", "", block("acronym"),
        "", "## Gold newly reachable via the acronym layer", "",
        base.df_to_markdown(ex_df) if not ex_df.empty else "_None._",
    ]
    (HERE / "ABBREV_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
