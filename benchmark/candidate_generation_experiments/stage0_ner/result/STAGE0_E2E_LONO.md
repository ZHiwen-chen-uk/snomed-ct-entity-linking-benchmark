# Stage-0 end-to-end, re-run with the learned rerankers

eval notes 55 · gold eval mentions 18676 · IoU thr 0.5 · gate seen-form (0, 0.5, 0) / novel-form (0.5, 0.5, 0)

`gated` reproduces the published end-to-end row and is the harness check. `supervised, standard` is the in-index-blind-spot model. `supervised, LONO` recomputes the four train-derived features leave-one-note-out (experiment 1b).

oracle_span = ranker on GOLD spans (detection = 1); e2e = detected at IoU>=0.5 AND linked correctly; detection_cost = oracle_span - e2e.

## vanilla

**gated**

| scope | n_gold | oracle_span_link_acc | detection_recall@IoU | e2e_recall | linking|detected | detection_cost |
| --- | --- | --- | --- | --- | --- | --- |
| all | 18676 | 0.7131 | 0.7882 | 0.5650 | 0.7168 | 0.1481 |
| head | 11446 | 0.7521 | 0.8360 | 0.6381 | 0.7633 | 0.1140 |
| torso | 4366 | 0.7109 | 0.7467 | 0.5140 | 0.6883 | 0.1969 |
| tail | 2864 | 0.5608 | 0.6606 | 0.3506 | 0.5307 | 0.2102 |

**supervised, standard**

| scope | n_gold | oracle_span_link_acc | detection_recall@IoU | e2e_recall | linking|detected | detection_cost |
| --- | --- | --- | --- | --- | --- | --- |
| all | 18676 | 0.7885 | 0.7882 | 0.6275 | 0.7961 | 0.1610 |
| head | 11446 | 0.9395 | 0.8360 | 0.7846 | 0.9384 | 0.1549 |
| torso | 4366 | 0.7476 | 0.7467 | 0.5330 | 0.7138 | 0.2146 |
| tail | 2864 | 0.2472 | 0.6606 | 0.1442 | 0.2183 | 0.1030 |

**supervised, LONO**

| scope | n_gold | oracle_span_link_acc | detection_recall@IoU | e2e_recall | linking|detected | detection_cost |
| --- | --- | --- | --- | --- | --- | --- |
| all | 18676 | 0.8689 | 0.7882 | 0.6830 | 0.8665 | 0.1859 |
| head | 11446 | 0.9520 | 0.8360 | 0.7956 | 0.9516 | 0.1564 |
| torso | 4366 | 0.8314 | 0.7467 | 0.5934 | 0.7948 | 0.2380 |
| tail | 2864 | 0.5939 | 0.6606 | 0.3698 | 0.5597 | 0.2241 |

## section_aware

**gated**

| scope | n_gold | oracle_span_link_acc | detection_recall@IoU | e2e_recall | linking|detected | detection_cost |
| --- | --- | --- | --- | --- | --- | --- |
| all | 18676 | 0.7131 | 0.8110 | 0.5836 | 0.7196 | 0.1295 |
| head | 11446 | 0.7521 | 0.8607 | 0.6601 | 0.7670 | 0.0920 |
| torso | 4366 | 0.7109 | 0.7762 | 0.5442 | 0.7011 | 0.1667 |
| tail | 2864 | 0.5608 | 0.6655 | 0.3376 | 0.5073 | 0.2232 |

**supervised, standard**

| scope | n_gold | oracle_span_link_acc | detection_recall@IoU | e2e_recall | linking|detected | detection_cost |
| --- | --- | --- | --- | --- | --- | --- |
| all | 18676 | 0.7885 | 0.8110 | 0.6554 | 0.8081 | 0.1331 |
| head | 11446 | 0.9395 | 0.8607 | 0.8151 | 0.9471 | 0.1244 |
| torso | 4366 | 0.7476 | 0.7762 | 0.5696 | 0.7338 | 0.1780 |
| tail | 2864 | 0.2472 | 0.6655 | 0.1477 | 0.2219 | 0.0995 |

**supervised, LONO**

| scope | n_gold | oracle_span_link_acc | detection_recall@IoU | e2e_recall | linking|detected | detection_cost |
| --- | --- | --- | --- | --- | --- | --- |
| all | 18676 | 0.8689 | 0.8110 | 0.7046 | 0.8688 | 0.1643 |
| head | 11446 | 0.9520 | 0.8607 | 0.8232 | 0.9565 | 0.1288 |
| torso | 4366 | 0.8314 | 0.7762 | 0.6216 | 0.8008 | 0.2098 |
| tail | 2864 | 0.5939 | 0.6655 | 0.3572 | 0.5367 | 0.2367 |

