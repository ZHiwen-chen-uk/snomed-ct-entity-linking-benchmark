# Experiment 1a — feature-hygiene ablation of the supervised reranker

train = all 3,124,680 candidate rows from 52,078 train mentions · eval = canonical split 18,676 mentions (head 11,446 / torso 4,366 / tail 2,864) · pool = weighted RRF@60 · model = HistGradientBoosting (max_iter=300, lr=0.08, depth=6, l2=1.0), identical to `stage2_reranker_full.py`.

The tail is split by index membership: **tail_out_of_index** (1,474 mentions, train_freq = 0, gold absent from the training annotations) and **tail_in_index** (1,390 mentions, train_freq 1-2).

Leaky features (functions of the training annotation set, not of the text): `train_rr`, `prior`, `log_freq`, `in_train`.

| model | all | head | torso | tail | tail_out_of_index | tail_in_index |
| --- | --- | --- | --- | --- | --- | --- |
| oracle (pool ceiling) | 0.9697 | 0.9938 | 0.9792 | 0.8593 | 0.7761 | 0.9475 |
| rrf_rank1 (no ranker) | 0.5508 | 0.5680 | 0.5957 | 0.4138 | 0.3039 | 0.5302 |
| full (13) | 0.7885 | 0.9395 | 0.7476 | 0.2472 | 0.0305 | 0.4770 |
| -in_train (12) | 0.7885 | 0.9395 | 0.7476 | 0.2472 | 0.0305 | 0.4770 |
| -prior (12) | 0.8123 | 0.9450 | 0.7893 | 0.3167 | 0.0210 | 0.6302 |
| -prior -in_train (11) | 0.8123 | 0.9450 | 0.7893 | 0.3167 | 0.0210 | 0.6302 |
| hygiene (9) | 0.7421 | 0.8235 | 0.6606 | 0.5409 | 0.4993 | 0.5849 |

HistGradientBoostingClassifier exposes no `feature_importances_`, so no importance table is reported. The ladder itself is the stronger substitute: it measures each feature group by its effect on the evaluation metric that matters, under a controlled retrain, rather than by an internal split-count proxy.
