# Candidate Generation Hybrid Results

Sample file: `.sample600.json`
SNOMED dir: `<project root>\SnomedCT_InternationalRF2_PRODUCTION_20260601T120000Z`

## Overall

| policy | n | gold_reachability | recall_at_30 | recall_at_45 | new_gold_found | mean_candidates |
| --- | --- | --- | --- | --- | --- | --- |
| train | 600 | 0.9067 | 0.9067 | 0.9067 | 0 | 11.3617 |
| synonym45 | 600 | 0.9250 | 0.9250 | 0.9250 | 11 | 11.5983 |
| boosted45_k25 | 600 | 0.9417 | 0.9400 | 0.9417 | 21 | 26.7350 |
| boosted60_k40 | 600 | 0.9483 | 0.9400 | 0.9467 | 25 | 36.0167 |
| grf_cache60_k40 | 600 | 0.9483 | 0.9400 | 0.9467 | 25 | 36.1350 |
| hybrid_char_after60_k40 | 600 | 0.9517 | 0.9400 | 0.9483 | 27 | 50.3400 |
| hybrid_char_first60_k40 | 600 | 0.9600 | 0.9417 | 0.9500 | 32 | 50.3400 |

## Tail

| policy | n | gold_reachability | recall_at_30 | recall_at_45 | new_gold_found | mean_candidates |
| --- | --- | --- | --- | --- | --- | --- |
| train | 86 | 0.4535 | 0.4535 | 0.4535 | 0 | 15.0814 |
| synonym45 | 86 | 0.5581 | 0.5581 | 0.5581 | 9 | 15.2791 |
| boosted45_k25 | 86 | 0.6744 | 0.6628 | 0.6744 | 19 | 33.7674 |
| boosted60_k40 | 86 | 0.7093 | 0.6628 | 0.7093 | 22 | 45.9302 |
| grf_cache60_k40 | 86 | 0.7093 | 0.6628 | 0.7093 | 22 | 46.1395 |
| hybrid_char_after60_k40 | 86 | 0.7326 | 0.6628 | 0.7209 | 24 | 54.8256 |
| hybrid_char_first60_k40 | 86 | 0.7907 | 0.6744 | 0.7326 | 29 | 54.8256 |

## Original Misses

| policy | n | gold_reachability | new_gold_found | mean_candidates |
| --- | --- | --- | --- | --- |
| train | 56 | 0.0000 | 0 | 15.9107 |
| synonym45 | 56 | 0.1964 | 11 | 16.1786 |
| boosted45_k25 | 56 | 0.3750 | 21 | 34.4107 |
| boosted60_k40 | 56 | 0.4464 | 25 | 46.8929 |
| grf_cache60_k40 | 56 | 0.4464 | 25 | 46.8929 |
| hybrid_char_after60_k40 | 56 | 0.4821 | 27 | 55.8571 |
| hybrid_char_first60_k40 | 56 | 0.5714 | 32 | 55.8571 |

## New Gold Examples

| policy | span | gold_term | freq_stratum | candidate_count | gold_rank_1based |
| --- | --- | --- | --- | --- | --- |
| synonym45 | ejection murmur | ejection murmur | torso | 5 | 5 |
| synonym45 | mucopurulent discharge | mucopurulent discharge | tail | 12 | 12 |
| synonym45 | Right ovary | right ovary | tail | 21 | 21 |
| synonym45 | pulmonary toxicity | pulmonary toxicity | tail | 23 | 23 |
| synonym45 | Intramuscular 
hematoma | intramuscular hematoma | tail | 8 | 8 |
| synonym45 | Crohns 
Disease | crohn disease | torso | 21 | 21 |
| synonym45 | Lupus anticoagulant | lupus anticoagulant | tail | 3 | 2 |
| synonym45 | deep hematoma | deep hematoma | tail | 19 | 19 |
| synonym45 | obstructive hydrocephalus | obstructive hydrocephalus | tail | 14 | 14 |
| synonym45 | Sexually active | sexually active | tail | 9 | 9 |
| synonym45 | psoriatic arthritis | psa | tail | 9 | 9 |
| boosted45_k25 | R lateral thigh | structure of lateral surface of right thigh | tail | 45 | 33 |

## Existing Ollama GRF Audit

This row is from the earlier real local-LLM rewrite audit, included here for comparison.

| mode | target | n | audit_limit | topk | ollama_calls | ollama_errors | rewrite_coverage | novel_rewrite_rate | mean_rewrite_count | mean_novel_rewrite_count | grf_adds_candidates_rate | mean_new_candidates_from_grf | train_reachability | lexical_reachability | grf_reachability | gold_found_only_by_grf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ollama | tail | 86 | 0 | 50 | 46 | 0 | 1.0000 | 1.0000 | 4.8953 | 4.6512 | 1.0000 | 31.3023 | 0.4535 | 0.7326 | 0.8256 | 8 |
