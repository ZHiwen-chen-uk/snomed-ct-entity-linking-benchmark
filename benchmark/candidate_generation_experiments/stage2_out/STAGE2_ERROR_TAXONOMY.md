# Ranking errors — ontological taxonomy and where the gold sits

eval = canonical split 18,676 mentions · pool = weighted RRF@60 · a RANKING error is a mention whose gold is in the pool but is not ranked first, so retrieval misses are excluded by construction.

Reachability is tested against each ranker's OWN pool. The two pools are built by different scripts and differ for a minority of mentions by tie-breaking at the 60-candidate cap; they give the same oracle ceiling to within one mention, but mixing them would silently drop real errors.

## 1. What kind of mistake is it?

Categories are shares of that scope's ranking errors. `granularity_share` = `too_broad` + `too_specific`, the only slice an is-a distance penalty (the proposal's Group B) could ever address. `max_gain_if_fixed` converts it to a bound on top-1 accuracy: even a PERFECT granularity fixer cannot gain more than that.

| ranker | scope | n_mentions | n_ranking_errors | err_rate | too_broad | too_specific | sibling | same_branch | cross_branch | unrelated | granularity_share | max_gain_if_fixed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gated | all | 18676 | 4792 | 0.2566 | 0.1503 | 0.0780 | 0.0532 | 0.3189 | 0.3516 | 0.0480 | 0.2283 | 0.0586 |
| gated | head | 11446 | 2767 | 0.2417 | 0.1265 | 0.0893 | 0.0434 | 0.3495 | 0.3824 | 0.0090 | 0.2158 | 0.0522 |
| gated | torso | 4366 | 1171 | 0.2682 | 0.1879 | 0.0581 | 0.0649 | 0.2707 | 0.2801 | 0.1383 | 0.2459 | 0.0660 |
| gated | tail | 2864 | 854 | 0.2982 | 0.1756 | 0.0691 | 0.0691 | 0.2857 | 0.3501 | 0.0504 | 0.2447 | 0.0730 |
| gated | tail_out_of_index | 1474 | 401 | 0.2720 | 0.2045 | 0.0898 | 0.1047 | 0.2918 | 0.2993 | 0.0100 | 0.2943 | 0.0801 |
| gated | tail_in_index | 1390 | 453 | 0.3259 | 0.1501 | 0.0508 | 0.0375 | 0.2804 | 0.3951 | 0.0861 | 0.2009 | 0.0655 |
| supervised, LONO | all | 18676 | 1883 | 0.1008 | 0.2034 | 0.0913 | 0.0552 | 0.2639 | 0.3224 | 0.0637 | 0.2947 | 0.0297 |
| supervised, LONO | head | 11446 | 478 | 0.0418 | 0.2301 | 0.1151 | 0.0335 | 0.2218 | 0.3828 | 0.0167 | 0.3452 | 0.0144 |
| supervised, LONO | torso | 4366 | 645 | 0.1477 | 0.2155 | 0.0589 | 0.0512 | 0.2760 | 0.2837 | 0.1147 | 0.2744 | 0.0405 |
| supervised, LONO | tail | 2864 | 760 | 0.2654 | 0.1763 | 0.1039 | 0.0724 | 0.2803 | 0.3171 | 0.0500 | 0.2803 | 0.0744 |
| supervised, LONO | tail_out_of_index | 1474 | 435 | 0.2951 | 0.1977 | 0.1034 | 0.0920 | 0.3126 | 0.2782 | 0.0161 | 0.3011 | 0.0889 |
| supervised, LONO | tail_in_index | 1390 | 325 | 0.2338 | 0.1477 | 0.1046 | 0.0462 | 0.2369 | 0.3692 | 0.0954 | 0.2523 | 0.0590 |

## 2. Is the ranking headroom reachable?

The oracle substitution says perfect ranking is worth a large amount. That is only attainable if the gold is near the top to begin with; if it sits at rank 30+ the headroom is nominal.

| ranker | scope | n_ranking_errors | gold_rank_median | gold_rank_p25 | gold_rank_p75 | gold_in_top3 | gold_in_top10 | gold_beyond_30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gated | all | 4792 | 3 | 2 | 7 | 0.5762 | 0.8214 | 0.0426 |
| gated | head | 2767 | 2 | 2 | 4 | 0.7047 | 0.9180 | 0.0123 |
| gated | torso | 1171 | 4 | 2 | 12 | 0.4381 | 0.7259 | 0.0692 |
| gated | tail | 854 | 6 | 3 | 16 | 0.3489 | 0.6393 | 0.1042 |
| gated | tail_out_of_index | 401 | 6 | 3 | 14 | 0.3516 | 0.6708 | 0.0698 |
| gated | tail_in_index | 453 | 6 | 3 | 20 | 0.3466 | 0.6115 | 0.1347 |
| supervised, LONO | all | 1883 | 3 | 2 | 6 | 0.6123 | 0.8587 | 0.0212 |
| supervised, LONO | head | 478 | 2 | 2 | 3 | 0.8389 | 0.9686 | 0.0000 |
| supervised, LONO | torso | 645 | 3 | 2 | 5 | 0.6357 | 0.9039 | 0.0016 |
| supervised, LONO | tail | 760 | 4 | 2 | 10 | 0.4500 | 0.7513 | 0.0513 |
| supervised, LONO | tail_out_of_index | 435 | 6 | 3 | 14 | 0.3908 | 0.7080 | 0.0667 |
| supervised, LONO | tail_in_index | 325 | 3 | 2 | 8 | 0.5292 | 0.8092 | 0.0308 |
