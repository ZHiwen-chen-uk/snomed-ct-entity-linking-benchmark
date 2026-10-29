# Selective prediction — accuracy-coverage and human-review triage

eval = canonical split 18,676 mentions, gold spans, weighted RRF@60 pool. `oracle (ceiling)` orders every error ahead of every correct prediction and bounds what any confidence signal could achieve.

## 1. Accuracy on the auto-accepted portion

| ranker | signal | coverage | n_accepted | accuracy_on_accepted |
| --- | --- | --- | --- | --- |
| gated | maxprob | 0.5000 | 9338 | 0.7483 |
| gated | maxprob | 0.6000 | 11206 | 0.7433 |
| gated | maxprob | 0.7000 | 13073 | 0.7385 |
| gated | maxprob | 0.8000 | 14941 | 0.7299 |
| gated | maxprob | 0.9000 | 16808 | 0.7242 |
| gated | maxprob | 1.0000 | 18676 | 0.7131 |
| gated | margin | 0.5000 | 9338 | 0.6928 |
| gated | margin | 0.6000 | 11206 | 0.6959 |
| gated | margin | 0.7000 | 13073 | 0.6964 |
| gated | margin | 0.8000 | 14941 | 0.7026 |
| gated | margin | 0.9000 | 16808 | 0.7087 |
| gated | margin | 1.0000 | 18676 | 0.7131 |
| gated | oracle (ceiling) | 0.5000 | 9338 | 1.0000 |
| gated | oracle (ceiling) | 0.6000 | 11206 | 1.0000 |
| gated | oracle (ceiling) | 0.7000 | 13073 | 1.0000 |
| gated | oracle (ceiling) | 0.8000 | 14941 | 0.8914 |
| gated | oracle (ceiling) | 0.9000 | 16808 | 0.7924 |
| gated | oracle (ceiling) | 1.0000 | 18676 | 0.7131 |
| supervised, LONO | maxprob | 0.5000 | 9338 | 0.9919 |
| supervised, LONO | maxprob | 0.6000 | 11206 | 0.9871 |
| supervised, LONO | maxprob | 0.7000 | 13073 | 0.9797 |
| supervised, LONO | maxprob | 0.8000 | 14941 | 0.9585 |
| supervised, LONO | maxprob | 0.9000 | 16808 | 0.9252 |
| supervised, LONO | maxprob | 1.0000 | 18676 | 0.8689 |
| supervised, LONO | margin | 0.5000 | 9338 | 0.9927 |
| supervised, LONO | margin | 0.6000 | 11206 | 0.9886 |
| supervised, LONO | margin | 0.7000 | 13073 | 0.9790 |
| supervised, LONO | margin | 0.8000 | 14941 | 0.9625 |
| supervised, LONO | margin | 0.9000 | 16808 | 0.9258 |
| supervised, LONO | margin | 1.0000 | 18676 | 0.8689 |
| supervised, LONO | oracle (ceiling) | 0.5000 | 9338 | 1.0000 |
| supervised, LONO | oracle (ceiling) | 0.6000 | 11206 | 1.0000 |
| supervised, LONO | oracle (ceiling) | 0.7000 | 13073 | 1.0000 |
| supervised, LONO | oracle (ceiling) | 0.8000 | 14941 | 1.0000 |
| supervised, LONO | oracle (ceiling) | 0.9000 | 16808 | 0.9655 |
| supervised, LONO | oracle (ceiling) | 1.0000 | 18676 | 0.8689 |

## 2. Review budget: what does reviewing X% buy?

`errors_captured` = share of all errors that land in the queue. `post_review_accuracy` = corpus accuracy if a reviewer corrects every queued mention. `precision_of_queue` = share of the queue that really was wrong, i.e. how much reviewer time is not wasted.

| ranker | signal | review_budget | errors_captured | post_review_accuracy | precision_of_queue |
| --- | --- | --- | --- | --- | --- |
| gated | maxprob | 0.0500 | 0.0823 | 0.7367 | 0.4722 |
| gated | maxprob | 0.1000 | 0.1349 | 0.7518 | 0.3870 |
| gated | maxprob | 0.2000 | 0.2469 | 0.7839 | 0.3542 |
| gated | maxprob | 0.3000 | 0.3619 | 0.8169 | 0.3461 |
| gated | maxprob | 0.5000 | 0.5614 | 0.8742 | 0.3221 |
| gated | margin | 0.0500 | 0.0420 | 0.7252 | 0.2409 |
| gated | margin | 0.1000 | 0.0860 | 0.7378 | 0.2468 |
| gated | margin | 0.2000 | 0.1706 | 0.7620 | 0.2447 |
| gated | margin | 0.3000 | 0.2592 | 0.7875 | 0.2479 |
| gated | margin | 0.5000 | 0.4645 | 0.8464 | 0.2665 |
| gated | oracle (ceiling) | 0.0500 | 0.1743 | 0.7631 | 1.0000 |
| gated | oracle (ceiling) | 0.1000 | 0.3486 | 0.8131 | 1.0000 |
| gated | oracle (ceiling) | 0.2000 | 0.6971 | 0.9131 | 1.0000 |
| gated | oracle (ceiling) | 0.3000 | 1.0000 | 1.0000 | 0.9563 |
| gated | oracle (ceiling) | 0.5000 | 1.0000 | 1.0000 | 0.5738 |
| supervised, LONO | maxprob | 0.0500 | 0.2917 | 0.9072 | 0.7645 |
| supervised, LONO | maxprob | 0.1000 | 0.4861 | 0.9326 | 0.6370 |
| supervised, LONO | maxprob | 0.2000 | 0.7467 | 0.9668 | 0.4894 |
| supervised, LONO | maxprob | 0.3000 | 0.8913 | 0.9858 | 0.3894 |
| supervised, LONO | maxprob | 0.5000 | 0.9690 | 0.9959 | 0.2540 |
| supervised, LONO | margin | 0.0500 | 0.2974 | 0.9079 | 0.7794 |
| supervised, LONO | margin | 0.1000 | 0.4906 | 0.9332 | 0.6429 |
| supervised, LONO | margin | 0.2000 | 0.7712 | 0.9700 | 0.5055 |
| supervised, LONO | margin | 0.3000 | 0.8877 | 0.9853 | 0.3878 |
| supervised, LONO | margin | 0.5000 | 0.9722 | 0.9964 | 0.2549 |
| supervised, LONO | oracle (ceiling) | 0.0500 | 0.3815 | 0.9189 | 1.0000 |
| supervised, LONO | oracle (ceiling) | 0.1000 | 0.7631 | 0.9689 | 1.0000 |
| supervised, LONO | oracle (ceiling) | 0.2000 | 1.0000 | 1.0000 | 0.6554 |
| supervised, LONO | oracle (ceiling) | 0.3000 | 1.0000 | 1.0000 | 0.4369 |
| supervised, LONO | oracle (ceiling) | 0.5000 | 1.0000 | 1.0000 | 0.2622 |

## 3. Who ends up in a 20% review queue? (margin signal)

`enrichment` > 1 means the stratum is over-represented in the queue relative to the corpus — the property you want for a long-tailed task.

| ranker | scope | share_of_corpus | share_of_20%_queue | enrichment | errors_of_this_scope_captured |
| --- | --- | --- | --- | --- | --- |
| gated | all | 1.0000 | 1.0000 | 1.0000 | 0.1874 |
| gated | head | 0.6129 | 0.7154 | 1.1700 | 0.2347 |
| gated | torso | 0.2338 | 0.2220 | 0.9500 | 0.1957 |
| gated | tail | 0.1534 | 0.0627 | 0.4100 | 0.0723 |
| gated | tail_out_of_index | 0.0789 | 0.0107 | 0.1400 | 0.0492 |
| gated | tail_in_index | 0.0744 | 0.0519 | 0.7000 | 0.1044 |
| supervised, LONO | all | 1.0000 | 1.0000 | 1.0000 | 0.7717 |
| supervised, LONO | head | 0.6129 | 0.2975 | 0.4900 | 0.8179 |
| supervised, LONO | torso | 0.2338 | 0.3202 | 1.3700 | 0.7663 |
| supervised, LONO | tail | 0.1534 | 0.3823 | 2.4900 | 0.7532 |
| supervised, LONO | tail_out_of_index | 0.0789 | 0.2300 | 2.9100 | 0.7503 |
| supervised, LONO | tail_in_index | 0.0744 | 0.1523 | 2.0500 | 0.7588 |
