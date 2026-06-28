# SNOMED CT Entity Linking — rough benchmark

A minimal, runnable benchmark harness for the
[DrivenData SNOMED CT Entity Linking Challenge](https://www.drivendata.org/competitions/258/competition-snomed-ct/),
built on the public release `snomed-ct-entity-linking-challenge-1.2.1`.

## Task
Given a clinical note, output character spans `[start, end)` and link each to a
SNOMED CT `concept_id`. Two sub-problems at once: **span detection** +
**concept normalization**.

## Data (public release)
- 272 annotated notes (`train_notes.csv`), ~10k chars each.
- 75,491 annotations, 6,595 unique concepts (`train_annotations.csv`).
- `annotation_type`: `train` (204 notes) / `test` (68 notes, disjoint) /
  `proposed_ACCEPTED` (model-proposed, human-accepted; scattered).

## Split used here
Notes with any `test` annotation → **test set (68)**; the rest → **train (204)**.
Note-level, so no span leaks between splits. Gold for a test note = all its
annotations.

## Metric — `metric.py`
Official-style **macro-averaged character IoU**: per concept_id, intersection /
union of character positions aggregated across all notes, then mean over
concepts (every concept weighted equally → rare concepts matter).

## Two harnesses in this folder

### 1. `run_benchmark.py` — trivial dictionary reference
Surface form → most frequent concept_id, greedy longest-phrase matching. The
"does the metric work + what does a trivial system get" anchor.
```
python run_benchmark.py
```

### 2. `run_experiments.py` — four-group ontology comparison (the dissertation skeleton)
Implements the proposal's controlled comparison on the candidate-**ranking**
sub-problem (gold spans given; pick the right SNOMED concept from a shared,
high-recall candidate set):

| group | what it does |
|---|---|
| **Baseline** | LLM ranks candidates, no ontology signal |
| **A: Pre-filter** | predicted semantic type removes incompatible candidates → rank |
| **B: Rerank** | rank → hierarchy-distance penalty (β) demotes structurally isolated candidates |
| **C: Filter+Rerank** | A then B |

```
python run_experiments.py        # prints tables, writes figures/*.png
```
Reports entity-level accuracy, macro char-IoU (spans fixed to gold, so it
reflects *linking* quality), candidate recall@k, and accuracy stratified by
**ambiguity type / concept frequency / ontology depth**.

**Module map** (each is a one-class swap point):
- `data.py` — load, split, `CandidateGenerator` (lexical; ← real SNOMED synonym index)
- `ontology.py` — `Ontology` interface; `MockOntology` (synthetic, runs today) + `SnomedOntology` (stub ← real RF2 files)
- `pipeline.py` — `LLMRanker`/`PriorRanker` (← real LLM), `SemTypePredictor` (← fine-tuned model), the 4 strategies
- `evaluate.py` — metrics + stratification · `run_experiments.py` — runner + charts

> ⚠️ **MockOntology produces SYNTHETIC types/hierarchy** — `run_experiments.py`
> numbers validate the *pipeline plumbing, metrics, stratification, and charts*,
> NOT the science. Replace `MockOntology`→`SnomedOntology` and
> `PriorRanker`→ a real LLM ranker before quoting any result.

### Swapping in real SNOMED CT + LLM
1. `SnomedOntology(release_dir)`: parse semantic tags from FSNs (`sct2_Description`)
   and the is-a graph (`sct2_Relationship`, typeId `116680003`) for `depth()`/`distance()`.
2. `LLMRanker` subclass: prompt the model (latest Claude, e.g. `claude-opus-4-8`)
   with `mention.context` + candidate preferred terms; parse its choice.
3. `SemTypePredictor.predict()`: replace the simulated draw with fine-tuned
   model inference. Everything downstream is unchanged.

## Current result
| | macro char IoU |
|---|---|
| dictionary baseline (this repo, public test split) | **0.2927** |
| ├─ head (frequent) concepts | 0.458 |
| └─ tail (rare) concepts | 0.148 |

### Official challenge numbers (private held-out test set, verified Jun 2026)
| | macro char IoU | approach |
|---|---|---|
| DrivenData/Veratai benchmark | **0.1794** | DeBERTa-v3-large span detect → link |
| median final submission | 0.2572 | (range 0.006–0.4202) |
| 🥉 MITEL-UNIUD (3rd) | 0.3777 | decoder (Faiss + Mistral) |
| 🥈 SNOBERT (2nd) | 0.4194 | encoder (fine-tuned BERT) |
| 🥇 KIRI (1st) | **0.4202** | dictionary-based |

Sources: [JAMIA / Davidson et al. 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12361850/),
[DrivenData winners blog](https://drivendata.co/blog/snomed-ct-entity-linking-challenge-winners),
[SNOBERT (arXiv 2405.16115)](https://arxiv.org/html/2405.16115v1).
KIRI vs SNOBERT 95% CIs overlap → not significantly different.

> **Not directly comparable to the live leaderboard** — that used a *private,
> held-out* test set, while this repo scores on a 68-note split of the *public*
> release. The repo's 0.2927 sitting above the official 0.1794 benchmark is
> expected: (a) easier/seen public test notes, (b) a surface-form dictionary
> is genuinely strong here — note the **winner (KIRI) was also dictionary-based**.
> The head/tail gap is the dissertation hook: an ontology-aware filter/reranker
> should lift the long tail most.

## Next steps for the dissertation
- [x] Four-group comparison harness (Baseline / A / B / C) — `run_experiments.py`
- [x] Metrics: entity-level accuracy, char-IoU, recall@k; stratified by
      ambiguity / frequency / depth; chart output
- [ ] Replace `MockOntology` with real SNOMED CT (`SnomedOntology`)
- [ ] Replace `PriorRanker` with a real LLM ranker
- [ ] Replace simulated `SemTypePredictor` with a fine-tuned semantic-type model
- [ ] Tune β and the filter confidence threshold on a validation split
