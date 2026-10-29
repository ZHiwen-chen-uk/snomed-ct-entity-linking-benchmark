# Stage-wise oracle substitution — section_aware, gated vs the LONO reranker

eval notes 55 · gold eval mentions 18676 · pred mentions 18072 · IoU thr 0.5 · gate seen (0, 0.5, 0) / novel (0.5, 0.5, 0)

Detection and retrieval are IDENTICAL across the two rows — only the ranking stage changes — so any change in the stage priority is caused solely by improving Stage 2.

### 1. End-to-end factorisation (identity_check must equal e2e)

| ranker | scope | n_gold | P(detected) | P(reach|det) | P(pick|det,reach) | e2e | identity_check |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gated | all | 18676 | 0.8110 | 0.9742 | 0.7387 | 0.5836 | 0.5836 |
| gated | head | 11446 | 0.8607 | 0.9963 | 0.7698 | 0.6601 | 0.6601 |
| gated | torso | 4366 | 0.7762 | 0.9823 | 0.7137 | 0.5442 | 0.5442 |
| gated | tail | 2864 | 0.6655 | 0.8452 | 0.6002 | 0.3376 | 0.3376 |
| supervised, LONO | all | 18676 | 0.8110 | 0.9742 | 0.8918 | 0.7046 | 0.7046 |
| supervised, LONO | head | 11446 | 0.8607 | 0.9963 | 0.9600 | 0.8232 | 0.8232 |
| supervised, LONO | torso | 4366 | 0.7762 | 0.9823 | 0.8153 | 0.6216 | 0.6216 |
| supervised, LONO | tail | 2864 | 0.6655 | 0.8452 | 0.6350 | 0.3572 | 0.3572 |

### 2. Marginal value of perfecting ONE stage

| ranker | scope | n_gold | current_e2e | perfect_detection (measured) | d_detection | perfect_retrieval (est.) | d_retrieval | perfect_ranking (exact) | d_ranking | perfect_det+retr (est.) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gated | all | 18676 | 0.5836 | 0.7131 | 0.1295 | 0.5990 | 0.0155 | 0.7901 | 0.2065 | 0.7387 |
| gated | head | 11446 | 0.6601 | 0.7521 | 0.0919 | 0.6626 | 0.0024 | 0.8575 | 0.1974 | 0.7698 |
| gated | torso | 4366 | 0.5442 | 0.7109 | 0.1667 | 0.5540 | 0.0098 | 0.7625 | 0.2183 | 0.7137 |
| gated | tail | 2864 | 0.3376 | 0.5608 | 0.2231 | 0.3995 | 0.0618 | 0.5625 | 0.2249 | 0.6002 |
| supervised, LONO | all | 18676 | 0.7046 | 0.8689 | 0.1643 | 0.7233 | 0.0187 | 0.7901 | 0.0855 | 0.8918 |
| supervised, LONO | head | 11446 | 0.8232 | 0.9520 | 0.1289 | 0.8262 | 0.0030 | 0.8575 | 0.0343 | 0.9600 |
| supervised, LONO | torso | 4366 | 0.6216 | 0.8314 | 0.2098 | 0.6328 | 0.0112 | 0.7625 | 0.1409 | 0.8153 |
| supervised, LONO | tail | 2864 | 0.3572 | 0.5939 | 0.2367 | 0.4226 | 0.0654 | 0.5625 | 0.2053 | 0.6350 |

`perfect_detection` is MEASURED (the gold-span pipeline). `perfect_ranking` is EXACT. `perfect_retrieval` is an optimistic ESTIMATE: it assumes currently-unreachable golds would be ranked as well as reachable ones, which they would not.

### 3. Gold-span pipeline only — Stage 1 vs Stage 2 without detection

| ranker | scope | n_gold | pool_reachability | P(pick|reachable) | gold_span_top1 | headroom_retrieval (est.) | headroom_ranking (exact) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gated | all | 18676 | 0.9697 | 0.7354 | 0.7131 | 0.0222 | 0.2566 |
| gated | head | 11446 | 0.9938 | 0.7567 | 0.7521 | 0.0047 | 0.2417 |
| gated | torso | 4366 | 0.9792 | 0.7261 | 0.7109 | 0.0151 | 0.2682 |
| gated | tail | 2864 | 0.8593 | 0.6526 | 0.5608 | 0.0918 | 0.2985 |
| supervised, LONO | all | 18676 | 0.9697 | 0.8960 | 0.8689 | 0.0271 | 0.1008 |
| supervised, LONO | head | 11446 | 0.9938 | 0.9580 | 0.9520 | 0.0059 | 0.0418 |
| supervised, LONO | torso | 4366 | 0.9792 | 0.8491 | 0.8314 | 0.0177 | 0.1477 |
| supervised, LONO | tail | 2864 | 0.8593 | 0.6912 | 0.5939 | 0.0973 | 0.2654 |
