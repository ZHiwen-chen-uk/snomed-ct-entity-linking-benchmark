# Stage-wise oracle substitution — vanilla, gated vs the LONO reranker

eval notes 55 · gold eval mentions 18676 · pred mentions 17229 · IoU thr 0.5 · gate seen (0, 0.5, 0) / novel (0.5, 0.5, 0)

Detection and retrieval are IDENTICAL across the two rows — only the ranking stage changes — so any change in the stage priority is caused solely by improving Stage 2.

### 1. End-to-end factorisation (identity_check must equal e2e)

| ranker | scope | n_gold | P(detected) | P(reach|det) | P(pick|det,reach) | e2e | identity_check |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gated | all | 18676 | 0.7882 | 0.9766 | 0.7340 | 0.5650 | 0.5650 |
| gated | head | 11446 | 0.8360 | 0.9975 | 0.7652 | 0.6381 | 0.6381 |
| gated | torso | 4366 | 0.7467 | 0.9828 | 0.7004 | 0.5140 | 0.5140 |
| gated | tail | 2864 | 0.6606 | 0.8605 | 0.6167 | 0.3506 | 0.3506 |
| supervised, LONO | all | 18676 | 0.7882 | 0.9766 | 0.8873 | 0.6830 | 0.6830 |
| supervised, LONO | head | 11446 | 0.8360 | 0.9975 | 0.9540 | 0.7956 | 0.7956 |
| supervised, LONO | torso | 4366 | 0.7467 | 0.9828 | 0.8087 | 0.5934 | 0.5934 |
| supervised, LONO | tail | 2864 | 0.6606 | 0.8605 | 0.6505 | 0.3698 | 0.3698 |

### 2. Marginal value of perfecting ONE stage

| ranker | scope | n_gold | current_e2e | perfect_detection (measured) | d_detection | perfect_retrieval (est.) | d_retrieval | perfect_ranking (exact) | d_ranking | perfect_det+retr (est.) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gated | all | 18676 | 0.5650 | 0.7131 | 0.1481 | 0.5785 | 0.0135 | 0.7698 | 0.2048 | 0.7340 |
| gated | head | 11446 | 0.6381 | 0.7521 | 0.1139 | 0.6397 | 0.0016 | 0.8339 | 0.1958 | 0.7652 |
| gated | torso | 4366 | 0.5140 | 0.7109 | 0.1970 | 0.5230 | 0.0090 | 0.7339 | 0.2199 | 0.7004 |
| gated | tail | 2864 | 0.3506 | 0.5608 | 0.2102 | 0.4074 | 0.0568 | 0.5684 | 0.2179 | 0.6167 |
| supervised, LONO | all | 18676 | 0.6830 | 0.8689 | 0.1859 | 0.6994 | 0.0163 | 0.7698 | 0.0868 | 0.8873 |
| supervised, LONO | head | 11446 | 0.7956 | 0.9520 | 0.1565 | 0.7976 | 0.0020 | 0.8339 | 0.0384 | 0.9540 |
| supervised, LONO | torso | 4366 | 0.5934 | 0.8314 | 0.2380 | 0.6038 | 0.0104 | 0.7339 | 0.1404 | 0.8087 |
| supervised, LONO | tail | 2864 | 0.3698 | 0.5939 | 0.2242 | 0.4297 | 0.0600 | 0.5684 | 0.1987 | 0.6505 |

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
