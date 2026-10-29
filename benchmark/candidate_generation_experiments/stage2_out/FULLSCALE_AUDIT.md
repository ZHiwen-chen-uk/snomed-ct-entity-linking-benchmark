# Full-scale audit — claims restated on the canonical eval split

full test mentions 23131 · canonical eval split 18676 (head 11446 / torso 4366 / tail 2864)

Ranker = gated (seen-form -> type, novel-form -> dense+type), identical to stage2_gated.py.

### 1. Error decomposition — is a gated-ranker error a retrieval miss or a ranking miss?

| scope | n | top1_acc | errors | recall_miss | recall_miss_share_of_err | ranking_miss | ranking_miss_share_of_err |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | 18676 | 0.7131 | 5358.0 | 566.0 | 0.1056 | 4792.0 | 0.8944 |
| head | 11446 | 0.7521 | 2838.0 | 71.0 | 0.025 | 2767.0 | 0.975 |
| torso | 4366 | 0.7109 | 1262.0 | 91.0 | 0.0721 | 1171.0 | 0.9279 |
| tail | 2864 | 0.5608 | 1258.0 | 404.0 | 0.3211 | 854.0 | 0.6789 |

### 2. Gold concept present in the training index (the supervised reranker's blind spot)

| scope | n | gold_in_train_index |
| --- | --- | --- |
| TRAIN_all | 52078 | 1.0 |
| TRAIN_head | 35929 | 1.0 |
| TRAIN_torso | 12280 | 1.0 |
| TRAIN_tail | 3869 | 1.0 |
| TEST_all | 18676 | 0.9211 |
| TEST_head | 11446 | 1.0 |
| TEST_torso | 4366 | 1.0 |
| TEST_tail | 2864 | 0.4853 |

### 3. Pool ceiling at k (oracle reachability within the top k of the RRF pool)

| scope | n | ceiling@1 | ceiling@10 | ceiling@20 | ceiling@30 | ceiling@60 |
| --- | --- | --- | --- | --- | --- | --- |
| all | 18676 | 0.551 | 0.8952 | 0.9383 | 0.9536 | 0.9697 |
| head | 11446 | 0.5683 | 0.9457 | 0.9823 | 0.991 | 0.9938 |
| torso | 4366 | 0.5957 | 0.8798 | 0.9343 | 0.9604 | 0.9792 |
| tail | 2864 | 0.4134 | 0.7168 | 0.7689 | 0.7936 | 0.8589 |

### 4. Acronym-shaped mentions — slice size, reachability, accuracy

| scope | n | top1_acc | share_of_eval | reachability |
| --- | --- | --- | --- | --- |
| all | 4363 | 0.7279 | 0.2336 | 0.9702 |
| head | 3205 | 0.7822 | 0.28 | 0.9938 |
| torso | 706 | 0.6657 | 0.1617 | 0.9717 |
| tail | 452 | 0.4403 | 0.1578 | 0.8009 |
