# Reproducibility package

Code and result tables for *A Stage-Wise Error Attribution for End-to-End SNOMED
CT Clinical Entity Linking*.

Section 1 lists every table and figure in the dissertation next to the script
that produced it and the file the numbers were read from. Use it to trace any
number in the text back to a command.

A note on the layout. It copies the working repository instead of grouping
scripts by topic, because each script finds its imports relative to its own
position on disk. Tidying the files into `stage1/`, `stage2/` and `analysis/`
folders would stop all of them running.

---

## 1. What produced each table and figure

| Dissertation artefact | Script | Result file |
|---|---|---|
| Table 2, corpus statistics | `dataset_stats.py` | printed, recomputed on demand |
| Table 3, methods inventory | written from the scripts below | — |
| Table 4, the four training-derived features | `stage2_feature_hygiene.py`, `stage2_lono_training.py` | `stage2_out/stage2_feature_hygiene.csv` |
| Table 5, span detection | `stage0_ner/run_ner.py` | `stage0_ner/result/STAGE0_RESULTS.md`, `stage0_metrics.json` |
| Table 6, candidate ranking | `stage2_reranker_full.py`, `stage2_feature_hygiene.py`, `stage2_lono_training.py`, `stage2_gated.py`, `stage2_unified.py` | `stage2_out/stage2_lono_training.csv`, `stage2_feature_hygiene.csv`, `stage2_unified.csv` |
| Table 6, confidence intervals | `stage2_bootstrap_ci.py` | `stage2_out/stage2_bootstrap_estimates.csv`, `stage2_bootstrap_diffs.csv` |
| Table 7, end to end | `stage0_chain_e2e_lono.py`, with `stage0_chain_e2e.py` for the published gated row | `stage0_ner/result/STAGE0_E2E_LONO.csv` |
| Table 8, BC5CDR replication | `../replication_bc5cdr/replicate_blind_spot.py` | printed by the script |
| Table 9, ranking-error taxonomy | `stage2_error_taxonomy.py` | `stage2_out/stage2_error_taxonomy.csv`, `stage2_gold_rank.csv` |
| Table 10, representative failures | `stage_error_examples.py` | withheld, see section 4 |
| Figure 1, dataset split and pipeline | `make_fig_dataset_split.py` | writes `fig5_dataset_split.png` |
| Figure 2, marginal value per stage | `make_figures.py`, from `stage_oracle_substitution_lono.py` | `stage2_out/oracle_substitution_lono_vanilla.csv` → `fig1_bottleneck_migration.png` |
| Figure 3, induced-mismatch sweep | `make_figures.py`, from `stage2_index_dose_response.py` | `stage2_out/stage2_index_dose_response.csv` → `fig4_dose_response.png` |
| Figure 4, the two halves of the rare stratum | `make_figures.py`, from `stage2_bootstrap_ci.py` | `stage2_out/stage2_bootstrap_estimates.csv` → `fig2_two_tails.png` |
| Figure 5, review triage | `make_figures.py`, from `stage2_abstention.py` | `stage2_out/stage2_abstention_budget.csv` → `fig3_triage.png` |
| §3.2, retrieval and fusion | `run_full_local.py`, `test_rrf_fusion.py`, `abbrev_full_scale.py` | `rrf_fusion_eval.csv`, `stage2_out/abbrev_fullscale.csv` |
| §3.3, semantic type as filter and as prior | `stage3_ontology_constraint.py` | `stage2_out/STAGE3_RESULTS.md` |
| §3.3, open-weight language-model probe | `stage2_export_llm_tail.py`, `stage2_llm_tail.py`, `colab_llm_tail_ranker.py` | `stage2_out/STAGE2_LLM_TAIL.md` |
| §3.5, the challenge's official metric | `stage0_challenge_metric.py` | `stage2_out/challenge_metric.csv` |
| §3.6, the three-factor identity | `stage_oracle_substitution_lono.py` | `stage2_out/oracle_identity_lono_vanilla.csv` |
| §4.5, the full-scale audit of earlier claims | `stage_audit_fullscale.py` | `stage2_out/fullscale_audit.csv` |

Watch out for the figure filenames. They do not match the figure numbers, because
the figures were made before the chapters were reordered. Go by the table above.

The five figures can be rebuilt from the tables here, without any access to the
corpus, and the PNGs come out byte-identical to the ones in the submitted
dissertation:

```
cd benchmark/candidate_generation_experiments
python make_figures.py            # figures 2 to 5
python make_fig_dataset_split.py  # figure 1
```

## 2. Order of the runs

Each stage caches what the next one reads, so they have to run in this order.

```
# stage 1, build the retrieval indexes once (slow, then cached)
python build_dense_index.py                 # SapBERT embeddings, ~1 GB cache
python test_candidate_generation.py         # builds the char n-gram index
python run_full_local.py                    # reachability and recall@k

# stage 0, span detection (needs a GPU; I ran this on Colab)
python stage0_ner/build_bio_data.py
python stage0_ner/run_ner.py

# stage 2, ranking over the fixed pool
python stage2_gated.py                      # heuristic, and the shared pool cache
python stage2_reranker_full.py              # supervised reranker, ~2.5 h on CPU
python stage2_feature_hygiene.py            # the feature-deletion repair
python stage2_lono_training.py              # the leave-one-note-out repair

# end to end
python stage0_chain_e2e_lono.py             # both detector variants, ~27 min

# analysis
python stage_oracle_substitution_lono.py
python stage2_error_taxonomy.py
python stage2_bootstrap_ci.py
python stage2_abstention.py
python stage2_index_dose_response.py
python stage0_challenge_metric.py

# figures
python make_figures.py
python make_fig_dataset_split.py
```

`run_fullscale_chain.sh` runs the expensive stage-2 sequence in the right order.
Do not run the two heavy scripts at the same time. Both load the same 1 GB
embedding matrix and the machine runs out of memory.

On Windows, set `KMP_DUPLICATE_LIB_OK=TRUE` first or the OpenMP runtime aborts.

## 3. Data and models to obtain separately

None of this can be redistributed, so it is not here. Each script sets its paths
at the top.

* **SNOMED CT Entity Linking Challenge corpus**, release 1.2.1, from PhysioNet
  under a credentialed data-use agreement. Holds the annotations and the MIMIC-IV
  discharge summaries.
* **SNOMED CT International RF2**, production release 2026-06-01, from SNOMED
  International or the UK Terminology Server, under an affiliate licence.
* **SapBERT** (`cambridgeltl/SapBERT-from-PubMedBERT-fulltext`) and
  **PubMedBERT**, from Hugging Face. Both download on first use.
* **BC5CDR** and the **CTD** vocabularies, needed only for the replication, from
  BioCreative and the Comparative Toxicogenomics Database.

## 4. Files held back under the data-use agreement

MIMIC-IV note text cannot be redistributed. Thirteen result files carry note text, or
one row per mention of a note, so they are not in this package even though the
scripts that write them are. Each one regenerates once you have the corpus.

| File | What is in it | Regenerate with |
|---|---|---|
| `stage_error_examples_all.csv` | one row per failure, with the note text around each mention | `python stage_error_examples.py` |
| `stage_error_examples_shortlist.csv` | the same, filtered | same |
| `STAGE_ERROR_EXAMPLES.md` | quotes mention text and its context | same |
| `abbrev_acronym_slice_misses.csv` | acronym surface forms taken from notes | `python abbrev_full_scale.py` |
| `abbrev_new_gold_examples.csv` | the same | same |
| `dense_new_gold_examples.csv` | mention text recovered by dense retrieval | `python run_full_local.py` |
| `hybrid_candidate_details.csv` | one row per mention, with note ids and spans | `python test_candidate_generation.py` |
| `stage0_pred_spans_vanilla.jsonl` | predicted character offsets for every held-out note | `python stage0_ner/run_ner.py` |
| `stage0_pred_spans_section_aware.jsonl` | the same, other detector variant | same |
| `GRF_OVER_DENSE_RESULTS.md` | mention text next to the concepts it was linked to | `python grf_over_dense_test.py` |
| `.llm_tail_*_cache.json`, `.stage2_llm_cache.json` | model answers keyed by note id and character offset | `python stage2_llm_tail.py` |

Table 10 of the dissertation quotes ten mentions. That is a minimal illustration
and falls under the same agreement.

## 5. Determinism

Every script fixes its seeds and breaks candidate ties deterministically, so the
reported numbers come back the same on a re-run. This is not decoration. An
earlier version of the replication took ties from `Counter.most_common` and from
set iteration order, and the headline figure moved by 13 points between runs. Two
probes now check for it, `replication_bc5cdr/probe_determinism.py` and
`diag_nondet.py`. Run both after touching anything in retrieval.

## 6. Environment

The results were produced on Windows with Python 3.8.8, CPU only, apart from the
two transformer training runs, which used a Colab GPU.

```
numpy==1.20.1
pandas==1.2.4
scikit-learn==0.24.1
scipy==1.6.2
torch==2.4.1
transformers==4.30.2
matplotlib==3.5.1
```

## 7. Included but not reported in the dissertation

These were part of the work, and no number in the dissertation rests on them.
They are here so that nothing looks hidden.

* `grf_over_dense_test.py`, a graph random-walk expansion over the dense pool. It
  recovered too few concepts to keep.
* `stage2_export_crossencoder_data.py` and `colab_crossencoder.py`, the
  cross-encoder. Its per-mention outputs were not saved, which the Table 6
  caption says.
* the `test_*.py` scripts, which are development harnesses and not unit tests.

## 8. Integrity

`MANIFEST.csv` gives the size and SHA-256 of every file in the package. It cannot
list itself.

Nothing here was assembled by hand. `build_reproducibility_package.py` works out
which local modules the scripts import, copies those, copies the aggregate result
tables, leaves out the files in section 4, and writes the manifest. Running it
again rebuilds the directory as it stands.
