# End-to-end scored with the challenge's own metric

55 evaluation notes · macro character-level IoU averaged over concept ids, per `benchmark/metric.py`. Published leaderboard scores for the three prize-winning systems: 0.4202, 0.4194, 0.3777 (Davidson et al., 2025).

**Comparability caveat.** The leaderboard used the challenge's held-out private test set; this uses the 55 evaluation notes of the public release, with 13 test notes reserved as a tuning split. Treat the comparison as indicative of the operating range, not as a like-for-like ranking.

| variant | ranker | n_pred_spans | n_concepts_scored | macro_char_iou | macro_iou_head | macro_iou_torso | macro_iou_tail |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vanilla | gated | 17229 | 4962 | 0.2703 | 0.5320 | 0.4365 | 0.3049 |
| vanilla | supervised, standard | 17229 | 4202 | 0.2373 | 0.5587 | 0.4096 | 0.1199 |
| vanilla | supervised, LONO | 17229 | 4468 | 0.3405 | 0.6443 | 0.5098 | 0.3239 |
| section_aware | gated | 18072 | 4792 | 0.2829 | 0.5433 | 0.4588 | 0.2935 |
| section_aware | supervised, standard | 18072 | 4283 | 0.2456 | 0.5870 | 0.4333 | 0.1264 |
| section_aware | supervised, LONO | 18072 | 4332 | 0.3544 | 0.6591 | 0.5293 | 0.3134 |

Per-stratum columns restrict the macro average to concepts of that training-frequency stratum, so they are directly comparable with the stratified recall figures reported elsewhere.
