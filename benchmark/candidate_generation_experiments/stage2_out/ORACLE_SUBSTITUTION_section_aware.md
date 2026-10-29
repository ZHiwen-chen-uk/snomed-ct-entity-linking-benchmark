# Stage-wise oracle substitution — section_aware

eval notes 55 · gold eval mentions 18676 · pred mentions 18072 · IoU thr 0.5 · gate seen (0, 0.5, 0) / novel (0.5, 0.5, 0)

### 1. End-to-end factorisation (identity_check must equal e2e)

| scope | n_gold | P(detected) | P(reach|det) | P(pick|det,reach) | e2e | identity_check |
| --- | --- | --- | --- | --- | --- | --- |
| all | 18676 | 0.8110 | 0.9742 | 0.7387 | 0.5836 | 0.5836 |
| head | 11446 | 0.8607 | 0.9963 | 0.7698 | 0.6601 | 0.6601 |
| torso | 4366 | 0.7762 | 0.9823 | 0.7137 | 0.5442 | 0.5442 |
| tail | 2864 | 0.6655 | 0.8452 | 0.6002 | 0.3376 | 0.3376 |

`P(detected)` = gold span matched by a predicted span at IoU>=0.5. `P(reach|det)` = gold concept present in the matched span's RRF@60 pool. `P(pick|det,reach)` = gated ranker selects it.

### 2. Marginal value of perfecting ONE stage (others held at measured performance)

| scope | n_gold | current_e2e | perfect_detection (measured) | d_detection | perfect_retrieval (est.) | d_retrieval | perfect_ranking (exact) | d_ranking | perfect_det+retr (est.) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 18676 | 0.5836 | 0.7131 | 0.1295 | 0.5990 | 0.0155 | 0.7901 | 0.2065 | 0.7387 |
| head | 11446 | 0.6601 | 0.7521 | 0.0919 | 0.6626 | 0.0024 | 0.8575 | 0.1974 | 0.7698 |
| torso | 4366 | 0.5442 | 0.7109 | 0.1667 | 0.5540 | 0.0098 | 0.7625 | 0.2183 | 0.7137 |
| tail | 2864 | 0.3376 | 0.5608 | 0.2231 | 0.3995 | 0.0618 | 0.5625 | 0.2249 | 0.6002 |

`perfect_detection` is MEASURED — it is the gold-span pipeline, run directly. `perfect_ranking` is EXACT (P(det)xP(reach|det)). `perfect_retrieval` is an ESTIMATE: it assumes P(pick|det,reach) also holds for mentions whose gold is currently unreachable, which is optimistic because those mentions are harder by construction.

### 3. Gold-span pipeline only — Stage 1 vs Stage 2 priority without detection

| scope | n_gold | pool_reachability | P(pick|reachable) | gold_span_top1 | headroom_retrieval (est.) | headroom_ranking (exact) |
| --- | --- | --- | --- | --- | --- | --- |
| all | 18676 | 0.9697 | 0.7354 | 0.7131 | 0.0223 | 0.2566 |
| head | 11446 | 0.9938 | 0.7567 | 0.7521 | 0.0047 | 0.2417 |
| torso | 4366 | 0.9792 | 0.7261 | 0.7109 | 0.0151 | 0.2682 |
| tail | 2864 | 0.8589 | 0.6528 | 0.5608 | 0.0921 | 0.2982 |

`headroom_retrieval` = what perfect retrieval would add on gold spans (estimate); `headroom_ranking` = what perfect ranking would add (exact).
