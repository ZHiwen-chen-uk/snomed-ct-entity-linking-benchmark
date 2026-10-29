# Stage-wise oracle substitution — vanilla

eval notes 55 · gold eval mentions 18676 · pred mentions 17229 · IoU thr 0.5 · gate seen (0, 0.5, 0) / novel (0.5, 0.5, 0)

### 1. End-to-end factorisation (identity_check must equal e2e)

| scope | n_gold | P(detected) | P(reach|det) | P(pick|det,reach) | e2e | identity_check |
| --- | --- | --- | --- | --- | --- | --- |
| all | 18676 | 0.7882 | 0.9766 | 0.7340 | 0.5650 | 0.5650 |
| head | 11446 | 0.8360 | 0.9975 | 0.7652 | 0.6381 | 0.6381 |
| torso | 4366 | 0.7467 | 0.9828 | 0.7004 | 0.5140 | 0.5140 |
| tail | 2864 | 0.6606 | 0.8605 | 0.6167 | 0.3506 | 0.3506 |

`P(detected)` = gold span matched by a predicted span at IoU>=0.5. `P(reach|det)` = gold concept present in the matched span's RRF@60 pool. `P(pick|det,reach)` = gated ranker selects it.

### 2. Marginal value of perfecting ONE stage (others held at measured performance)

| scope | n_gold | current_e2e | perfect_detection (measured) | d_detection | perfect_retrieval (est.) | d_retrieval | perfect_ranking (exact) | d_ranking | perfect_det+retr (est.) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 18676 | 0.5650 | 0.7131 | 0.1481 | 0.5785 | 0.0135 | 0.7698 | 0.2048 | 0.7340 |
| head | 11446 | 0.6381 | 0.7521 | 0.1139 | 0.6397 | 0.0016 | 0.8339 | 0.1958 | 0.7652 |
| torso | 4366 | 0.5140 | 0.7109 | 0.1970 | 0.5230 | 0.0090 | 0.7339 | 0.2199 | 0.7004 |
| tail | 2864 | 0.3506 | 0.5608 | 0.2102 | 0.4074 | 0.0568 | 0.5684 | 0.2179 | 0.6167 |

`perfect_detection` is MEASURED — it is the gold-span pipeline, run directly. `perfect_ranking` is EXACT (P(det)xP(reach|det)). `perfect_retrieval` is an ESTIMATE: it assumes P(pick|det,reach) also holds for mentions whose gold is currently unreachable, which is optimistic because those mentions are harder by construction.

### 3. Gold-span pipeline only — Stage 1 vs Stage 2 priority without detection

| scope | n_gold | pool_reachability | P(pick|reachable) | gold_span_top1 | headroom_retrieval (est.) | headroom_ranking (exact) |
| --- | --- | --- | --- | --- | --- | --- |
| all | 18676 | 0.9697 | 0.7354 | 0.7131 | 0.0223 | 0.2566 |
| head | 11446 | 0.9938 | 0.7567 | 0.7521 | 0.0047 | 0.2417 |
| torso | 4366 | 0.9792 | 0.7261 | 0.7109 | 0.0151 | 0.2682 |
| tail | 2864 | 0.8589 | 0.6528 | 0.5608 | 0.0921 | 0.2982 |

`headroom_retrieval` = what perfect retrieval would add on gold spans (estimate); `headroom_ranking` = what perfect ranking would add (exact).
