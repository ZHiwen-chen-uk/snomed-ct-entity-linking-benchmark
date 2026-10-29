# Stage-0 end-to-end, re-run with the learned rerankers

eval notes 3 · gold eval mentions 789 · IoU thr 0.5 · gate seen-form (0, 0.5, 0) / novel-form (0.5, 0.5, 0)

`gated` reproduces the published end-to-end row and is the harness check. `supervised, standard` is the in-index-blind-spot model. `supervised, LONO` recomputes the four train-derived features leave-one-note-out (experiment 1b).

oracle_span = ranker on GOLD spans (detection = 1); e2e = detected at IoU>=0.5 AND linked correctly; detection_cost = oracle_span - e2e.

## vanilla

**gated**

| scope | n_gold | oracle_span_link_acc | detection_recall@IoU | e2e_recall | linking|detected | detection_cost |
| --- | --- | --- | --- | --- | --- | --- |
| all | 789 | 0.6071 | 0.6895 | 0.4322 | 0.6268 | 0.1749 |
| head | 433 | 0.6628 | 0.7552 | 0.5104 | 0.6758 | 0.1524 |
| torso | 216 | 0.5833 | 0.6574 | 0.3796 | 0.5775 | 0.2037 |
| tail | 140 | 0.4714 | 0.5357 | 0.2714 | 0.5067 | 0.2000 |

**supervised, standard**

| scope | n_gold | oracle_span_link_acc | detection_recall@IoU | e2e_recall | linking|detected | detection_cost |
| --- | --- | --- | --- | --- | --- | --- |
| all | 789 | 0.7136 | 0.6895 | 0.5108 | 0.7408 | 0.2028 |
| head | 433 | 0.8938 | 0.7552 | 0.6813 | 0.9021 | 0.2125 |
| torso | 216 | 0.6713 | 0.6574 | 0.4398 | 0.6690 | 0.2315 |
| tail | 140 | 0.2214 | 0.5357 | 0.0929 | 0.1733 | 0.1285 |

**supervised, LONO**

| scope | n_gold | oracle_span_link_acc | detection_recall@IoU | e2e_recall | linking|detected | detection_cost |
| --- | --- | --- | --- | --- | --- | --- |
| all | 789 | 0.7934 | 0.6895 | 0.5526 | 0.8015 | 0.2408 |
| head | 433 | 0.9053 | 0.7552 | 0.6767 | 0.8960 | 0.2286 |
| torso | 216 | 0.7315 | 0.6574 | 0.4769 | 0.7254 | 0.2546 |
| tail | 140 | 0.5429 | 0.5357 | 0.2857 | 0.5333 | 0.2572 |

## section_aware

**gated**

| scope | n_gold | oracle_span_link_acc | detection_recall@IoU | e2e_recall | linking|detected | detection_cost |
| --- | --- | --- | --- | --- | --- | --- |
| all | 789 | 0.6071 | 0.7123 | 0.4525 | 0.6352 | 0.1546 |
| head | 433 | 0.6628 | 0.7921 | 0.5358 | 0.6764 | 0.1270 |
| torso | 216 | 0.5833 | 0.6667 | 0.3935 | 0.5903 | 0.1898 |
| tail | 140 | 0.4714 | 0.5357 | 0.2857 | 0.5333 | 0.1857 |

**supervised, standard**

| scope | n_gold | oracle_span_link_acc | detection_recall@IoU | e2e_recall | linking|detected | detection_cost |
| --- | --- | --- | --- | --- | --- | --- |
| all | 789 | 0.7136 | 0.7123 | 0.5425 | 0.7616 | 0.1711 |
| head | 433 | 0.8938 | 0.7921 | 0.7275 | 0.9184 | 0.1663 |
| torso | 216 | 0.6713 | 0.6667 | 0.4444 | 0.6667 | 0.2269 |
| tail | 140 | 0.2214 | 0.5357 | 0.1214 | 0.2267 | 0.1000 |

**supervised, LONO**

| scope | n_gold | oracle_span_link_acc | detection_recall@IoU | e2e_recall | linking|detected | detection_cost |
| --- | --- | --- | --- | --- | --- | --- |
| all | 789 | 0.7934 | 0.7123 | 0.5843 | 0.8203 | 0.2091 |
| head | 433 | 0.9053 | 0.7921 | 0.7275 | 0.9184 | 0.1778 |
| torso | 216 | 0.7315 | 0.6667 | 0.4815 | 0.7222 | 0.2500 |
| tail | 140 | 0.5429 | 0.5357 | 0.3000 | 0.5600 | 0.2429 |

