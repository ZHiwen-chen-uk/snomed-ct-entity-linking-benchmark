# Colab Runbook — Full-Dataset Candidate Generation (dense + RRF)

Runs the full retrieval stack (train / synonym / lexical / char / dense-SapBERT)
and weighted RRF fusion over **all 23,131 held-out test mentions**, so long-tail
reachability/recall is measured with real power (thousands of tail mentions vs
the 86 in the local `.sample600`).

No SNOMED RF2 release is needed: lexical indexes ship as cached pickles and a
stub ontology supplies labels. SapBERT SNOMED encoding is minutes on GPU.

## Upload package

`cand_full_colab_upload.zip` (~24 MB) already contains, in repo layout:

```
benchmark/
  candidate_booster.py data.py evaluate.py metric.py pipeline.py ontology.py
  .snomed_synindex.pkl .snomed_tokenindex.pkl .snomed_concept_labels.pkl
  candidate_generation_experiments/
    dense_retriever.py retrieval_fusion.py abbrev_expander.py
    test_candidate_generation.py colab_full_candidate_eval.py
snomed-ct-entity-linking-challenge-1.2.1/snomed-ct-entity-linking-challenge-1.2.1/
  train_notes.csv train_annotations.csv
```

The char n-gram index is NOT shipped (100 MB); it rebuilds from the synonym
index on first run (a few minutes).

## Steps

1. Runtime → Change runtime type → **GPU** (T4 is enough).

2. Upload and unzip:

```python
from google.colab import files; files.upload()          # pick cand_full_colab_upload.zip
!unzip -q cand_full_colab_upload.zip -d /content/proj
```

3. Install deps:

```bash
!pip -q install "transformers==4.30.2" faiss-gpu
```

(If `faiss-gpu` fails on the current CUDA, use `faiss-cpu` — search over 665k
vectors is still fast.)

4. Run:

```bash
%cd /content/proj/benchmark/candidate_generation_experiments
!python colab_full_candidate_eval.py --out-dir /content/drive/MyDrive/cand_full
```

Mount Drive first (`from google.colab import drive; drive.mount('/content/drive')`)
if you want outputs persisted there; otherwise drop `--out-dir` for a local dir.

Debug fast with `--limit 1000` before the full run.

## Outputs

```
<out-dir>/full_candidate_eval.csv        # policy x {all,head,torso,tail}
<out-dir>/FULL_CANDIDATE_RESULTS.md      # readable tables
```

Policies: `concat` (naive hybrid+dense) vs `rrf_weighted` (weighted RRF over all
five retrievers). Metrics per stratum: reachability, recall@{1,10,30,60}, mean
candidates.

## What to look for

The local 600-sample result to confirm at scale (weighted RRF vs concat, cap 60):

| stratum | metric | concat | rrf_weighted |
|---|---|---:|---:|
| all  | reach | 0.960 | 0.978 |
| all  | r@10  | 0.883 | 0.888 |
| tail | reach | 0.791 | 0.895 |
| tail | r@10  | 0.372 | **0.733** |
| tail | r@30  | 0.674 | 0.791 |

The headline claim to validate on 23k mentions: **RRF over dense+lexical roughly
doubles tail recall@10 at the same pool size, while not hurting head/overall.**

## Tuning knobs

- `RRF_WEIGHTS` in `colab_full_candidate_eval.py` (default
  `synonym 3, train 2, lexical 1, char 1, dense 1.5`) — raise `synonym`/`train`
  if head r@10 needs protecting; raise `dense` to push tail recall.
- `--k` per-retriever depth, `--cap` final pool size, `--rrf-k` RRF constant.
