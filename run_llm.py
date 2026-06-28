"""Run the four-group experiment with the REAL Claude LLM as the Stage-1 ranker.

    python run_llm.py            # default sample
    LLM_SAMPLE=200 python run_llm.py

Differs from run_experiments.py only in the ranker: PriorRanker -> ClaudeRanker
(claude-opus-4-8 via the Anthropic API). Everything else — candidate generation,
the MockOntology, metrics, stratification, charts — is shared.

Because an LLM call per (mention, candidate-set) costs money and time, this runs
on a SAMPLE of the test mentions (deterministic seed) rather than all 23k. Set
LLM_SAMPLE to change the size. Charts go to benchmark/figures_llm/.

Credentials: set ANTHROPIC_API_KEY, or run `ant auth login` first.

NOTE: ranker is real; MockOntology is still synthetic, so Groups A/B/C remain a
plumbing demo. The Baseline numbers here ARE a real LLM entity-linking result.
"""
import os
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiments as rx               # reuse chart + helpers
from data import CandidateGenerator, build_test_mentions, load, split_notes
from ontology import MockOntology
from pipeline import ClaudeRanker, OllamaRanker, SemTypePredictor, Strategies, GROUPS
from evaluate import evaluate_all, recall_at_k

SAMPLE = int(os.environ.get("LLM_SAMPLE", "150"))
BACKEND = os.environ.get("RANKER", "ollama")   # "ollama" (local) | "claude" (API)
# ONLY=Baseline restricts to the real Baseline group (A/B/C need real SNOMED).
ONLY = [g.strip() for g in os.environ.get("ONLY", "").split(",") if g.strip()]
SEED = 20260626
CACHE = Path(__file__).resolve().parent / ".llm_cache.json"


def make_ranker(cg):
    if BACKEND == "claude":
        return ClaudeRanker(cg, cache_file=str(CACHE))
    return OllamaRanker(cg, cache_file=str(CACHE))


def main():
    out_dir = Path(__file__).resolve().parent / "figures_llm"
    out_dir.mkdir(exist_ok=True)
    rx.FIG_DIR = out_dir                     # redirect chart output

    notes, ann = load()
    train_notes, test_notes = split_notes(ann)
    cg = CandidateGenerator().fit(ann, train_notes)
    mentions = build_test_mentions(notes, ann, test_notes, cg)
    train_freq = Counter(int(c) for c in
                         ann[ann.note_id.isin(train_notes)].concept_id)

    # deterministic sample (keep only mentions with >=1 candidate)
    mentions = [m for m in mentions if m.candidates]
    random.Random(SEED).shuffle(mentions)
    mentions = mentions[:SAMPLE]

    ontology = MockOntology()
    ranker = make_ranker(cg)                  # local Ollama (default) or Claude API
    predictor = SemTypePredictor(ontology)
    strat = Strategies(ranker, predictor, ontology)

    groups = {k: v for k, v in GROUPS.items() if k in ONLY} if ONLY else GROUPS
    model_name = getattr(ranker, "model", "claude-opus-4-8")
    print(f"LLM ranker: {BACKEND} ({model_name}) | sample: {len(mentions)} mentions")
    print(f"groups: {list(groups)}\n")

    recall = recall_at_k(mentions)
    summary, strata = evaluate_all(mentions, strat, ontology, train_freq, groups)
    baseline_only = list(groups) == ["Baseline"]
    suffix = f"real LLM baseline ({model_name}, n={len(mentions)})"
    h_title = (f"SNOMED CT entity linking — {suffix}" if baseline_only else None)
    s_title = (f"Baseline stratified accuracy — {suffix}" if baseline_only else None)

    print(f"LLM calls: {ranker.calls} | API errors (fell back): {ranker.errors}")
    print("\n=== Candidate recall@k (sample) ===")
    for k in sorted(recall):
        print(f"  recall@{k:<3} = {recall[k]:.4f}")
    print("\n=== Headline metrics by group (real LLM baseline) ===")
    print(summary.to_string(index=False,
          formatters={"accuracy": "{:.4f}".format, "char_iou": "{:.4f}".format}))
    print("\n=== Accuracy by ambiguity type ===")
    for g, d in strata["ambiguity"].items():
        cells = "  ".join(f"{k}={v:.3f}" for k, v in sorted(d.items()))
        print(f"  {g:<18} {cells}")

    rx.chart_headline(summary, recall, title=h_title)
    rx.chart_strata(strata, title=s_title)
    print(f"\ncharts written to: {out_dir}")
    print("  - headline.png / stratified.png")


if __name__ == "__main__":
    main()
