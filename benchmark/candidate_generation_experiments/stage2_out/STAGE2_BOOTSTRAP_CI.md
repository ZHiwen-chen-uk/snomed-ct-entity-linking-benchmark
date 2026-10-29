# Stage-2 rankers — note-clustered bootstrap confidence intervals

eval = canonical split 18,676 mentions in 55 notes · 10,000 bootstrap replicates, resampling NOTES with replacement · 95% percentile intervals · differences are PAIRED (same resample for both rankers).

Mentions within a discharge summary are correlated, so resampling mentions would understate the variance; the cluster is the note. With only 55 clusters the intervals are necessarily wide — that is the honest precision of this corpus, not a defect of the estimator.

## Top-1 accuracy with 95% CI

| ranker | all | head | torso | tail | tail_out_of_index | tail_in_index |
| --- | --- | --- | --- | --- | --- | --- |
| rrf_rank1 | 0.5508 [0.5345, 0.5664] | 0.5680 [0.5526, 0.5827] | 0.5957 [0.5681, 0.6232] | 0.4138 [0.3813, 0.4452] | 0.3039 [0.2678, 0.3418] | 0.5302 [0.4881, 0.5709] |
| gated | 0.7131 [0.6965, 0.7291] | 0.7521 [0.7379, 0.7654] | 0.7109 [0.6866, 0.7333] | 0.5608 [0.5225, 0.5987] | 0.5041 [0.4571, 0.5547] | 0.6209 [0.5791, 0.6605] |
| supervised, standard | 0.7885 [0.7705, 0.8055] | 0.9395 [0.9304, 0.9475] | 0.7476 [0.7258, 0.7689] | 0.2472 [0.2257, 0.2692] | 0.0305 [0.0203, 0.0425] | 0.4770 [0.4435, 0.5102] |
| supervised, hygiene | 0.7421 [0.7221, 0.7604] | 0.8235 [0.8073, 0.8379] | 0.6606 [0.6342, 0.6852] | 0.5409 [0.5060, 0.5771] | 0.4993 [0.4560, 0.5453] | 0.5849 [0.5465, 0.6219] |
| supervised, LONO | 0.8689 [0.8554, 0.8816] | 0.9520 [0.9455, 0.9582] | 0.8314 [0.8130, 0.8483] | 0.5939 [0.5586, 0.6301] | 0.4810 [0.4356, 0.5295] | 0.7137 [0.6796, 0.7463] |

## Paired differences

30 comparisons are reported, so the per-comparison 95% interval would inflate the family-wise error rate. A Bonferroni-corrected 99.83% interval is given alongside; `survives correction` is the conservative verdict and is the one to quote. Paired differences are far tighter than the marginal intervals above because the same resampled notes are used for both rankers.

| comparison | scope | diff | 95% CI | excludes 0 | Bonferroni 99.83% CI | survives correction |
| --- | --- | --- | --- | --- | --- | --- |
| supervised, LONO - supervised, hygiene | all | 0.1268 | [0.1176, 0.1370] | yes | [0.1123, 0.1435] | yes |
| supervised, LONO - supervised, hygiene | head | 0.1285 | [0.1176, 0.1411] | yes | [0.1123, 0.1490] | yes |
| supervised, LONO - supervised, hygiene | torso | 0.1709 | [0.1548, 0.1864] | yes | [0.1465, 0.1965] | yes |
| supervised, LONO - supervised, hygiene | tail | 0.0531 | [0.0379, 0.0687] | yes | [0.0287, 0.0788] | yes |
| supervised, LONO - supervised, hygiene | tail_out_of_index | -0.0183 | [-0.0315, -0.0060] | yes | [-0.0407, 0.0022] | NO |
| supervised, LONO - supervised, hygiene | tail_in_index | 0.1288 | [0.1073, 0.1507] | yes | [0.0943, 0.1643] | yes |
| supervised, LONO - gated | all | 0.1558 | [0.1475, 0.1643] | yes | [0.1418, 0.1694] | yes |
| supervised, LONO - gated | head | 0.2000 | [0.1896, 0.2108] | yes | [0.1831, 0.2183] | yes |
| supervised, LONO - gated | torso | 0.1205 | [0.1057, 0.1354] | yes | [0.0976, 0.1438] | yes |
| supervised, LONO - gated | tail | 0.0332 | [0.0161, 0.0498] | yes | [0.0053, 0.0590] | yes |
| supervised, LONO - gated | tail_out_of_index | -0.0231 | [-0.0413, -0.0053] | yes | [-0.0527, 0.0053] | NO |
| supervised, LONO - gated | tail_in_index | 0.0928 | [0.0651, 0.1184] | yes | [0.0459, 0.1328] | yes |
| supervised, LONO - supervised, standard | all | 0.0804 | [0.0707, 0.0910] | yes | [0.0649, 0.0973] | yes |
| supervised, LONO - supervised, standard | head | 0.0125 | [0.0080, 0.0177] | yes | [0.0057, 0.0217] | yes |
| supervised, LONO - supervised, standard | torso | 0.0838 | [0.0711, 0.0973] | yes | [0.0639, 0.1052] | yes |
| supervised, LONO - supervised, standard | tail | 0.3467 | [0.3161, 0.3776] | yes | [0.2970, 0.3939] | yes |
| supervised, LONO - supervised, standard | tail_out_of_index | 0.4505 | [0.4047, 0.4997] | yes | [0.3762, 0.5224] | yes |
| supervised, LONO - supervised, standard | tail_in_index | 0.2367 | [0.2070, 0.2670] | yes | [0.1920, 0.2826] | yes |
| supervised, hygiene - supervised, standard | all | -0.0464 | [-0.0617, -0.0311] | yes | [-0.0718, -0.0227] | yes |
| supervised, hygiene - supervised, standard | head | -0.1160 | [-0.1283, -0.1050] | yes | [-0.1387, -0.0988] | yes |
| supervised, hygiene - supervised, standard | torso | -0.0870 | [-0.1101, -0.0632] | yes | [-0.1238, -0.0519] | yes |
| supervised, hygiene - supervised, standard | tail | 0.2936 | [0.2572, 0.3298] | yes | [0.2347, 0.3521] | yes |
| supervised, hygiene - supervised, standard | tail_out_of_index | 0.4688 | [0.4248, 0.5150] | yes | [0.4025, 0.5427] | yes |
| supervised, hygiene - supervised, standard | tail_in_index | 0.1079 | [0.0668, 0.1499] | yes | [0.0434, 0.1742] | yes |
| gated - supervised, standard | all | -0.0754 | [-0.0914, -0.0589] | yes | [-0.1004, -0.0480] | yes |
| gated - supervised, standard | head | -0.1875 | [-0.1995, -0.1758] | yes | [-0.2088, -0.1686] | yes |
| gated - supervised, standard | torso | -0.0366 | [-0.0594, -0.0134] | yes | [-0.0717, 0.0006] | NO |
| gated - supervised, standard | tail | 0.3135 | [0.2761, 0.3504] | yes | [0.2565, 0.3673] | yes |
| gated - supervised, standard | tail_out_of_index | 0.4735 | [0.4259, 0.5243] | yes | [0.3996, 0.5535] | yes |
| gated - supervised, standard | tail_in_index | 0.1439 | [0.1005, 0.1878] | yes | [0.0745, 0.2122] | yes |
