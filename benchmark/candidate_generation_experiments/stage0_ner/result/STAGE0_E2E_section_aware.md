# Stage-0 end-to-end — section_aware

eval notes 55 · gold eval mentions 18676 · pred mentions 18072 · IoU thr 0.5
gate seen-form (0, 0.5, 0) / novel-form (0.5, 0.5, 0)

oracle_span = gated ranker on GOLD spans (detection=1); e2e = detected@IoU>=0.5 AND linked correctly; detection_cost = oracle_span - e2e

| scope | n_gold | oracle_span_link_acc | detection_recall@IoU | e2e_recall | linking|detected | detection_cost |
| --- | --- | --- | --- | --- | --- | --- |
| all | 18676 | 0.7131 | 0.8110 | 0.5836 | 0.7196 | 0.1295 |
| head | 11446 | 0.7521 | 0.8607 | 0.6601 | 0.7670 | 0.0920 |
| torso | 4366 | 0.7109 | 0.7762 | 0.5442 | 0.7011 | 0.1667 |
| tail | 2864 | 0.5608 | 0.6655 | 0.3376 | 0.5073 | 0.2232 |
