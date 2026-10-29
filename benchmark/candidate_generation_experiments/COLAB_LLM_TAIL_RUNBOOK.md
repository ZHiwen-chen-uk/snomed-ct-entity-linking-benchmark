# Colab Runbook — LLM tail ranking (item 3)

Tests whether a strong context LLM can beat the leakage-free simple signals on
the long tail — the tail is data-limited, so the question is whether world
knowledge in a large LLM breaks that ceiling. Ranks the tail candidate pool with
Qwen2.5-7B-Instruct and compares against the precomputed baselines.

Target to beat: **gated 0.546** on the RRF top-20 pool. Ceiling (gold in top-20)
is 0.769, so no ranker here can exceed ~0.77.

Colab needs only HF + two CSVs — no SNOMED, dense, or caches.

## Package `llm_tail_colab_upload.zip`

```
colab_llm_tail_ranker.py
llm_tail_mentions.csv     # key, mention_text, gold, oracle_ok, dense/type/gated picks
llm_tail_candidates.csv   # key, rank, cand_id, cand_term  (top-20 per mention)
```

## Steps

1. Runtime → GPU (T4 is enough for the 7B model at 4-bit).
2. Upload + unzip:
```python
from google.colab import files; files.upload()      # llm_tail_colab_upload.zip
!unzip -q llm_tail_colab_upload.zip -d /content/lt
```
3. Install:
```bash
!pip -q install -U transformers accelerate bitsandbytes
```
4. Run (start with a subset, then scale up):
```bash
%cd /content/lt
!python colab_llm_tail_ranker.py --data-dir . --limit 800
```
`--limit 0` (or a big number) runs all 2,864 tail mentions (~60–90 min on T4).
Swap `--model Qwen/Qwen2.5-14B-Instruct` on an A100 for a stronger test, or
`--model microsoft/Phi-3.5-mini-instruct` for a faster/smaller one.

## Output

```
llm_tail_out/LLM_TAIL_RESULTS.md      # ranker | tail accuracy
llm_tail_out/llm_tail_predictions.csv # per-mention picks, for error analysis
```

## What the result means

- If the LLM clearly beats gated (0.546), world knowledge helps the tail and is
  worth pursuing — the data-limited ceiling is not fundamental.
- If it does not, that strengthens the finding that tail ranking is genuinely
  data-limited: even a large general model with broad medical knowledge cannot
  pick the right rare SNOMED concept from context alone.

Either way it is a clean result for the write-up. Compare like-for-like: the
baselines in the table are computed on exactly the mentions the LLM ran on.
