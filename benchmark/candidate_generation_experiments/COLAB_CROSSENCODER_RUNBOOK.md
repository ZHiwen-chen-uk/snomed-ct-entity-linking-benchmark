# Colab Runbook — #2 Cross-encoder reranker (PubMedBERT)

Fine-tunes a PubMedBERT cross-encoder that scores (mention+context) vs (candidate
SNOMED term), reranks the RRF@60 pool, and reports stratified top-1 accuracy over
the full 23k test set. Target: close the tail gap that simple ontology/dense
signals could not (gated ceiling tail 0.56 → oracle tail 0.86).

Colab needs only HF + two CSVs — no SNOMED, dense, or booster (all done locally).

## Package

`crossencoder_colab_upload.zip` contains:
```
colab_crossencoder.py
ce_mentions.csv    # key, split, val_or_eval, mention_text, gold, freq_stratum
ce_pairs.csv       # key, cand_id, cand_term, label   (RRF@60 pool, ~1.9M rows)
```

## Steps

1. Runtime → GPU (T4 fine).
2. Upload + unzip:
```python
from google.colab import files; files.upload()      # crossencoder_colab_upload.zip
!unzip -q crossencoder_colab_upload.zip -d /content/ce
```
3. Install (use a recent transformers — do NOT pin 4.30.2, its old tokenizers
   has no prebuilt wheel on Colab and fails to compile; `datasets` not needed):
```bash
!pip -q install -U transformers accelerate
```
4. Run:
```bash
%cd /content/ce
!python colab_crossencoder.py --data-dir . --out-dir /content/drive/MyDrive/ce_out
```
(mount Drive first to persist outputs, else use a local `--out-dir`).

~2 epochs over ~200k train pairs + inference on ~1.4M test pairs ≈ 15–30 min on T4.

## Outputs

```
<out-dir>/crossencoder_accuracy.csv     # stratum, n, acc  (eval split)
<out-dir>/CROSSENCODER_RESULTS.md
```

## What to compare against (full-23k eval split, established locally)

| stratum | oracle ceiling | gated simple-signal upper bound | cross-encoder (target) |
|---|---:|---:|---:|
| all  | 0.970 | 0.713 | ? |
| head | 0.994 | 0.752 | ? |
| torso| 0.979 | 0.711 | ? |
| tail | 0.859 | 0.561 | **? (aim ≫ 0.56)** |

The claim to validate: **context-aware cross-encoder reranking beats the
gated simple-signal upper bound, especially on tail, moving toward the ceiling.**

## Knobs (`colab_crossencoder.py`)

- `MODEL` — swap PubMedBERT for another biomedical encoder.
- `NEG_PER_MENTION` (24) — hard negatives per train mention.
- `--epochs` (2), `--batch` (32), `--lr` (2e-5).
- Class-weighted loss already handles the positive/negative imbalance.
