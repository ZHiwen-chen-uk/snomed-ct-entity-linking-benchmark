# Abbreviation Expansion (#2) Results

Acronym slice: 131 of the fixed sample carry an acronym-shaped token. Mined acronyms (train-only): 789. Curated seed: 29.

## All sample

| policy | n | gold_reachability | recall_at_10 | recall_at_30 | new_gold_found | mean_candidates |
| --- | --- | --- | --- | --- | --- | --- |
| synonym45 | 600 | 0.9250 | 0.8817 | 0.9250 | 11 | 11.5983 |
| synonym45+abbrev_precise | 600 | 0.9250 | 0.8817 | 0.9250 | 11 | 11.6100 |
| synonym45+abbrev_fuzzy | 600 | 0.9267 | 0.8817 | 0.9267 | 12 | 16.1500 |
| hybrid_char_first60_k40 | 600 | 0.9600 | 0.8850 | 0.9417 | 32 | 50.3400 |
| hybrid_char_first60_k40+abbrev_precise | 600 | 0.9600 | 0.8850 | 0.9417 | 32 | 50.3450 |

## Tail

| policy | n | gold_reachability | recall_at_10 | recall_at_30 | new_gold_found | mean_candidates |
| --- | --- | --- | --- | --- | --- | --- |
| synonym45 | 86 | 0.5581 | 0.3721 | 0.5581 | 9 | 15.2791 |
| synonym45+abbrev_precise | 86 | 0.5581 | 0.3721 | 0.5581 | 9 | 15.3023 |
| synonym45+abbrev_fuzzy | 86 | 0.5698 | 0.3721 | 0.5698 | 10 | 17.6279 |
| hybrid_char_first60_k40 | 86 | 0.7907 | 0.3953 | 0.6744 | 29 | 54.8256 |
| hybrid_char_first60_k40+abbrev_precise | 86 | 0.7907 | 0.3953 | 0.6744 | 29 | 54.8372 |

## Acronym slice (the targeted subset)

| policy | n | gold_reachability | recall_at_10 | recall_at_30 | new_gold_found | mean_candidates |
| --- | --- | --- | --- | --- | --- | --- |
| synonym45 | 131 | 0.9084 | 0.8931 | 0.9084 | 0 | 6.5725 |
| synonym45+abbrev_precise | 131 | 0.9084 | 0.8931 | 0.9084 | 0 | 6.6260 |
| synonym45+abbrev_fuzzy | 131 | 0.9160 | 0.8931 | 0.9160 | 1 | 27.4198 |
| hybrid_char_first60_k40 | 131 | 0.9313 | 0.8931 | 0.9160 | 3 | 46.4656 |
| hybrid_char_first60_k40+abbrev_precise | 131 | 0.9313 | 0.8931 | 0.9160 | 3 | 46.4885 |

## Gold newly reachable via the acronym layer

_None._
