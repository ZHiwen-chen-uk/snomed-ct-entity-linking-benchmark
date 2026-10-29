"""Candidate-generation supplement experiments.

This is a recall-ceiling test, not a downstream A/B/C ranking test. It keeps the
same fixed gold spans and compares several ways to enlarge the shared candidate
pool before any ontology strategy runs.

Outputs:
  * hybrid_candidate_eval.csv      scope-level reachability metrics
  * hybrid_candidate_details.csv   mention-level diagnostics
  * HYBRID_RESULTS.md              short readable summary
"""

import argparse
import copy
import json
import math
import pickle
import re
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
PROJECT = BENCHMARK.parent
DEFAULT_SNOMED_DIR = PROJECT / "SnomedCT_InternationalRF2_PRODUCTION_20260601T120000Z"
CHAR_CACHE = HERE / ".snomed_charindex_v1.pkl"

sys.path.insert(0, str(BENCHMARK))

from candidate_booster import SnomedCandidateBooster, norm, ordered_unique, tokens
from data import CandidateGenerator, build_test_mentions, load, split_notes
from evaluate import freq_stratum
from ontology import SnomedOntology


CHAR_TOKEN_RE = re.compile(r"[a-z0-9]+")


def char_text(text):
    parts = CHAR_TOKEN_RE.findall(norm(text))
    return " ".join(parts)


def char_grams(text, n_min=3, n_max=4):
    txt = char_text(text)
    if not txt:
        return set()
    padded = f" {txt} "
    grams = set()
    for n in range(n_min, n_max + 1):
        if len(padded) < n:
            continue
        grams.update(padded[i : i + n] for i in range(len(padded) - n + 1))
    return grams


def dice(a, b):
    if not a or not b:
        return 0.0
    return 2.0 * len(a & b) / (len(a) + len(b))


def load_mentions(sample_file):
    notes, ann = load()
    train_notes, test_notes = split_notes(ann)
    cg = CandidateGenerator().fit(ann, train_notes)
    mentions = build_test_mentions(notes, ann, test_notes, cg)
    train_freq = Counter(int(c) for c in ann[ann.note_id.isin(train_notes)].concept_id)

    if sample_file:
        sample_path = BENCHMARK / sample_file
        want = {
            (item["note_id"], int(item["start"]))
            for item in json.loads(sample_path.read_text())
        }
        mentions = [m for m in mentions if (m.note_id, m.start) in want]

    return cg, mentions, train_freq


class CharNgramRetriever:
    """Character n-gram backoff over SNOMED description forms.

    Token retrieval misses cases where the surface form has punctuation,
    spelling variants, or fused words. This retriever deliberately stays simple:
    it proposes extra SNOMED concepts from char n-gram overlap, then rescales
    with preferred-label similarity.
    """

    def __init__(self, booster, max_postings=9000, max_form_len=90):
        self.booster = booster
        self.max_postings = int(max_postings)
        self.max_form_len = int(max_form_len)
        self.index = self._load_or_build_index()
        self._feature_cache = {}
        self._retrieve_cache = {}

    def label(self, concept):
        return self.booster.label(concept)

    def retrieve(self, span, limit):
        cache_key = (norm(span), int(limit))
        if cache_key in self._retrieve_cache:
            hits, scores = self._retrieve_cache[cache_key]
            return list(hits), dict(scores)

        qgrams = char_grams(span)
        if not qgrams:
            return [], {}

        raw = Counter()
        for gram in qgrams:
            posting = self.index.get(gram, ())
            if not posting:
                continue
            weight = 1.0 / max(1.0, math.sqrt(len(posting)))
            for cid in posting:
                raw[int(cid)] += weight

        if not raw:
            return [], {}

        query_text = char_text(span)
        query_tokens = set(tokens(span))
        shortlist_size = max(limit * 35, 350)
        shortlist = [cid for cid, _ in raw.most_common(shortlist_size)]
        rescored = []
        for cid in shortlist:
            label_text, label_grams, label_tokens = self._label_features(cid)
            gram_score = dice(qgrams, label_grams)
            token_score = dice(query_tokens, label_tokens)
            seq_score = SequenceMatcher(None, query_text, label_text).ratio() if label_text else 0.0
            score = raw[cid] + 2.2 * gram_score + 0.7 * token_score + 0.5 * seq_score
            rescored.append((cid, score))

        rescored.sort(key=lambda kv: kv[1], reverse=True)
        top = rescored[:limit]
        max_score = max((score for _, score in top), default=1.0)
        scores = {cid: (score / max_score if max_score else 0.0) for cid, score in top}
        hits = [cid for cid, _ in top]
        self._retrieve_cache[cache_key] = (tuple(hits), dict(scores))
        return hits, scores

    def _label_features(self, concept):
        concept = int(concept)
        if concept not in self._feature_cache:
            label = self.label(concept)
            self._feature_cache[concept] = (
                char_text(label),
                char_grams(label),
                set(tokens(label)),
            )
        return self._feature_cache[concept]

    def _load_or_build_index(self):
        meta = {
            "max_postings": self.max_postings,
            "max_form_len": self.max_form_len,
            "source_forms": len(self.booster.syn_index),
        }
        if CHAR_CACHE.exists():
            with open(CHAR_CACHE, "rb") as f:
                cached = pickle.load(f)
            if cached.get("meta") == meta:
                return cached["index"]

        buckets = {}
        dropped = set()
        for form, concepts in self.booster.syn_index.items():
            txt = char_text(form)
            if len(txt) < 3 or len(txt) > self.max_form_len:
                continue
            grams = char_grams(txt)
            if not grams:
                continue
            for gram in grams:
                if gram in dropped:
                    continue
                bucket = buckets.setdefault(gram, set())
                bucket.update(int(c) for c in concepts)
                if len(bucket) > self.max_postings:
                    dropped.add(gram)
                    del buckets[gram]

        compact = {
            gram: tuple(sorted(concepts))
            for gram, concepts in buckets.items()
            if concepts and gram not in dropped
        }
        with open(CHAR_CACHE, "wb") as f:
            pickle.dump({"meta": meta, "index": compact}, f, protocol=pickle.HIGHEST_PROTOCOL)
        return compact


def exact_hits(booster, span):
    return [int(c) for c in booster.syn_index.get(norm(span), ())]


def apply_booster(cg, ontology, snomed_dir, base_mentions, mode, max_candidates, expand_k, grf_mode="cache"):
    mentions = copy.deepcopy(base_mentions)
    booster = SnomedCandidateBooster(
        cg,
        ontology,
        snomed_dir,
        mode=mode,
        max_candidates=max_candidates,
        expand_k=expand_k,
        grf_mode=grf_mode,
    )
    booster.boost_mentions(mentions)
    return mentions, booster


def apply_hybrid(base_mentions, booster, char_retriever, max_candidates, expand_k, char_first=False):
    mentions = copy.deepcopy(base_mentions)
    for mention in mentions:
        base = list(mention.candidates)
        syn = exact_hits(booster, mention.span)
        lex, _ = booster.retrieve([mention.span], expand_k)
        ch, _ = char_retriever.retrieve(mention.span, expand_k)
        if char_first:
            mention.candidates = ordered_unique(base + syn + ch + lex, max_candidates)
        else:
            mention.candidates = ordered_unique(base + syn + lex + ch, max_candidates)
    return mentions


def recall_at_k(mentions, k):
    if not mentions:
        return 0.0
    return sum(1 for m in mentions if m.gold_concept in m.candidates[:k]) / len(mentions)


def scope_subsets(mentions, train_freq, train_reached):
    subsets = [("all_sample", mentions)]
    for name in ("head", "torso", "tail"):
        subsets.append((name, [m for m in mentions if freq_stratum(m.gold_concept, train_freq) == name]))
    subsets.append(
        (
            "original_miss",
            [m for m in mentions if (m.note_id, m.start) not in train_reached],
        )
    )
    return subsets


def summarize_policy(name, mentions, train_freq, train_reached):
    rows = []
    for scope, subset in scope_subsets(mentions, train_freq, train_reached):
        if not subset:
            continue
        hits = {
            (m.note_id, m.start)
            for m in subset
            if m.gold_concept in set(m.candidates)
        }
        sizes = [len(m.candidates) for m in subset]
        rows.append(
            {
                "policy": name,
                "scope": scope,
                "n": len(subset),
                "gold_reachability": len(hits) / len(subset),
                "recall_at_1": recall_at_k(subset, 1),
                "recall_at_3": recall_at_k(subset, 3),
                "recall_at_5": recall_at_k(subset, 5),
                "recall_at_10": recall_at_k(subset, 10),
                "recall_at_30": recall_at_k(subset, 30),
                "recall_at_45": recall_at_k(subset, 45),
                "new_gold_found": sum(1 for key in hits if key not in train_reached),
                "mean_candidates": sum(sizes) / len(sizes),
                "max_candidates_seen": max(sizes),
            }
        )
    return rows


def mention_details(policy, mentions, train_freq, train_reached, labeler):
    out = []
    for m in mentions:
        key = (m.note_id, m.start)
        reachable = m.gold_concept in set(m.candidates)
        out.append(
            {
                "policy": policy,
                "note_id": m.note_id,
                "start": m.start,
                "span": m.span,
                "section": m.section,
                "gold": m.gold_concept,
                "gold_term": labeler.label(m.gold_concept),
                "freq_stratum": freq_stratum(m.gold_concept, train_freq),
                "train_reachable": key in train_reached,
                "policy_reachable": reachable,
                "new_gold_found": reachable and key not in train_reached,
                "candidate_count": len(m.candidates),
                "gold_rank_1based": (m.candidates.index(m.gold_concept) + 1) if reachable else "",
            }
        )
    return out


def df_to_markdown(df, columns=None):
    if columns:
        df = df[columns]
    if df.empty:
        return "_No rows._"
    lines = [
        "| " + " | ".join(df.columns) + " |",
        "| " + " | ".join(["---"] * len(df.columns)) + " |",
    ]
    for row in df.itertuples(index=False):
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(summary_df, details_df, args):
    overall = summary_df[summary_df.scope == "all_sample"].copy()
    tail = summary_df[summary_df.scope == "tail"].copy()
    miss = summary_df[summary_df.scope == "original_miss"].copy()
    useful = details_df[details_df.new_gold_found == True].head(12)

    lines = [
        "# Candidate Generation Hybrid Results",
        "",
        f"Sample file: `{args.sample_file}`",
        f"SNOMED dir: `{args.snomed_dir}`",
        "",
        "## Overall",
        "",
        df_to_markdown(
            overall,
            [
                "policy",
                "n",
                "gold_reachability",
                "recall_at_30",
                "recall_at_45",
                "new_gold_found",
                "mean_candidates",
            ],
        ),
        "",
        "## Tail",
        "",
        df_to_markdown(
            tail,
            [
                "policy",
                "n",
                "gold_reachability",
                "recall_at_30",
                "recall_at_45",
                "new_gold_found",
                "mean_candidates",
            ],
        ),
        "",
        "## Original Misses",
        "",
        df_to_markdown(
            miss,
            ["policy", "n", "gold_reachability", "new_gold_found", "mean_candidates"],
        ),
        "",
        "## New Gold Examples",
        "",
        df_to_markdown(
            useful,
            [
                "policy",
                "span",
                "gold_term",
                "freq_stratum",
                "candidate_count",
                "gold_rank_1based",
            ],
        ),
    ]

    ollama_summary = BENCHMARK / "long_tail_b_strategies" / "grf_audit_summary_ollama_tail.csv"
    if ollama_summary.exists():
        grf_df = pd.read_csv(ollama_summary)
        lines.extend(
            [
                "",
                "## Existing Ollama GRF Audit",
                "",
                "This row is from the earlier real local-LLM rewrite audit, included here for comparison.",
                "",
                df_to_markdown(grf_df),
            ]
        )

    (HERE / "HYBRID_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-file", default=".sample600.json")
    parser.add_argument("--snomed-dir", default=str(DEFAULT_SNOMED_DIR))
    parser.add_argument("--char-max-postings", type=int, default=9000)
    parser.add_argument("--char-max-form-len", type=int, default=90)
    args = parser.parse_args()

    print("loading data...")
    cg, base_mentions, train_freq = load_mentions(args.sample_file)
    train_reached = {
        (m.note_id, m.start)
        for m in base_mentions
        if m.gold_concept in set(m.candidates)
    }
    print(f"mentions={len(base_mentions)} train_reachable={len(train_reached)}")

    print("loading ontology and existing SNOMED indexes...")
    ontology = SnomedOntology(args.snomed_dir)

    policies = {}
    policies["train"] = copy.deepcopy(base_mentions)
    policies["synonym45"], _ = apply_booster(
        cg, ontology, args.snomed_dir, base_mentions, "synonym", 45, 25
    )
    policies["boosted45_k25"], boosted = apply_booster(
        cg, ontology, args.snomed_dir, base_mentions, "boosted", 45, 25
    )
    policies["boosted60_k40"], wide_boosted = apply_booster(
        cg, ontology, args.snomed_dir, base_mentions, "boosted", 60, 40
    )
    policies["grf_cache60_k40"], _ = apply_booster(
        cg,
        ontology,
        args.snomed_dir,
        base_mentions,
        "grf",
        60,
        40,
        grf_mode="cache",
    )

    print("building/loading char n-gram index...")
    char_retriever = CharNgramRetriever(
        wide_boosted,
        max_postings=args.char_max_postings,
        max_form_len=args.char_max_form_len,
    )
    policies["hybrid_char_after60_k40"] = apply_hybrid(
        base_mentions, wide_boosted, char_retriever, 60, 40, char_first=False
    )
    policies["hybrid_char_first60_k40"] = apply_hybrid(
        base_mentions, wide_boosted, char_retriever, 60, 40, char_first=True
    )

    print("summarising...")
    summary_rows = []
    detail_rows = []
    for name, mentions in policies.items():
        summary_rows.extend(summarize_policy(name, mentions, train_freq, train_reached))
        labeler = wide_boosted if name.startswith("hybrid") or name.startswith("boosted") else boosted
        detail_rows.extend(mention_details(name, mentions, train_freq, train_reached, labeler))

    summary_df = pd.DataFrame(summary_rows)
    details_df = pd.DataFrame(detail_rows)
    summary_df.to_csv(HERE / "hybrid_candidate_eval.csv", index=False)
    details_df.to_csv(HERE / "hybrid_candidate_details.csv", index=False)
    write_report(summary_df, details_df, args)

    view = summary_df[summary_df.scope.isin(["all_sample", "tail", "original_miss"])]
    print(view.to_string(index=False, formatters={
        "gold_reachability": "{:.4f}".format,
        "recall_at_30": "{:.4f}".format,
        "recall_at_45": "{:.4f}".format,
        "mean_candidates": "{:.2f}".format,
    }))
    print(f"\nsaved -> {HERE}")


if __name__ == "__main__":
    main()
