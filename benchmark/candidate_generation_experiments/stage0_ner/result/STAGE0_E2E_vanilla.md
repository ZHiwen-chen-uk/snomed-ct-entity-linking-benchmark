# Stage-0 end-to-end — vanilla

eval notes 55 · gold eval mentions 18676 · pred mentions 17229 · IoU thr 0.5
gate seen-form (0, 0.5, 0) / novel-form (0.5, 0.5, 0)

oracle_span = gated ranker on GOLD spans (detection=1); e2e = detected@IoU>=0.5 AND linked correctly; detection_cost = oracle_span - e2e

| scope | n_gold | oracle_span_link_acc | detection_recall@IoU | e2e_recall | linking|detected | detection_cost |
| --- | --- | --- | --- | --- | --- | --- |
| all | 18676 | 0.7131 | 0.7882 | 0.5650 | 0.7168 | 0.1481 |
| head | 11446 | 0.7521 | 0.8360 | 0.6381 | 0.7633 | 0.1140 |
| torso | 4366 | 0.7109 | 0.7467 | 0.5140 | 0.6883 | 0.1969 |
| tail | 2864 | 0.5608 | 0.6606 | 0.3506 | 0.5307 | 0.2102 |
