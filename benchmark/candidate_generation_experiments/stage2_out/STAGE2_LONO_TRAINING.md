# Experiment 1b — leave-one-note-out (LONO) training

train = 52,078 train mentions x 60 pool rows · eval = canonical split 18,676 mentions (head 11,446 / torso 4,366 / tail 2,864) · same HistGradientBoosting and same weighted RRF@60 pool as `stage2_reranker_full.py`.

LONO recomputes `train_rr`, `prior`, `log_freq` and `in_train` for every train mention from a CandidateGenerator fitted on the other 203 train notes. The pool is held fixed and the eval features are unchanged: at deployment the full training index is available for retrieval, so the only train/test mismatch being repaired is in the supervision signal.

**Manipulation check.** Under standard training a train mention's gold is in the index by construction, so gold-in-index = 1.0000 in every stratum and the model never sees a positive with `in_train = 0`. Under LONO that rate falls to **0.9268** (mean gold prior 0.7753), so 7.3% of training positives are now out-of-index. The test tail is 51.5% out-of-index, so the distributions are still not matched, but the training signal now has non-zero variance on the property that defines the tail.

| model | all | head | torso | tail | tail_out_of_index | tail_in_index |
| --- | --- | --- | --- | --- | --- | --- |
| oracle (pool ceiling) | 0.9697 | 0.9938 | 0.9792 | 0.8593 | 0.7761 | 0.9475 |
| rrf_rank1 (no ranker) | 0.5508 | 0.5680 | 0.5957 | 0.4138 | 0.3039 | 0.5302 |
| full (13), standard train | 0.7885 | 0.9395 | 0.7476 | 0.2472 | 0.0305 | 0.4770 |
| hygiene (9), standard train | 0.7421 | 0.8235 | 0.6606 | 0.5409 | 0.4993 | 0.5849 |
| full (13), LONO train | 0.8689 | 0.9520 | 0.8314 | 0.5939 | 0.4810 | 0.7137 |

`hygiene (9)` is unaffected by LONO: the four features LONO recomputes are exactly the four it drops, so a LONO-trained hygiene model is numerically identical to the standard-trained one and is not reported twice.
