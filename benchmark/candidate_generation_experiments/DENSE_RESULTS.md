# Dense Retrieval (#1, SapBERT) Results

Sample: `.sample600.json`  ·  dense-k = 60  ·  encoder = cambridgeltl/SapBERT-from-PubMedBERT-fulltext

## All sample

| policy | n | gold_reachability | recall_at_10 | recall_at_30 | recall_at_60 | new_gold_found | mean_candidates |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_char_first60 | 600 | 0.9600 | 0.8867 | 0.9417 | 0.9600 | 32 | 50.3400 |
| dense60 | 600 | 0.8000 | 0.6900 | 0.7583 | 0.8000 | 46 | 60.0000 |
| hybrid+dense_cap60 | 600 | 0.9617 | 0.8867 | 0.9417 | 0.9617 | 33 | 60.0000 |
| hybrid+dense_cap90 | 600 | 0.9850 | 0.8867 | 0.9417 | 0.9617 | 47 | 88.4350 |

## Tail

| policy | n | gold_reachability | recall_at_10 | recall_at_30 | recall_at_60 | new_gold_found | mean_candidates |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_char_first60 | 86 | 0.7907 | 0.3953 | 0.6744 | 0.7907 | 29 | 54.8256 |
| dense60 | 86 | 0.8605 | 0.8140 | 0.8372 | 0.8605 | 40 | 60.0000 |
| hybrid+dense_cap60 | 86 | 0.8023 | 0.3953 | 0.6744 | 0.8023 | 30 | 60.0000 |
| hybrid+dense_cap90 | 86 | 0.9302 | 0.3953 | 0.6744 | 0.8023 | 41 | 89.3605 |

## Acronym slice

| policy | n | gold_reachability | recall_at_10 | recall_at_30 | recall_at_60 | new_gold_found | mean_candidates |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_char_first60 | 131 | 0.9313 | 0.8931 | 0.9160 | 0.9313 | 3 | 46.4656 |
| dense60 | 131 | 0.7023 | 0.5802 | 0.6489 | 0.7023 | 8 | 60.0000 |
| hybrid+dense_cap60 | 131 | 0.9389 | 0.8931 | 0.9160 | 0.9389 | 4 | 60.0000 |
| hybrid+dense_cap90 | 131 | 0.9695 | 0.8931 | 0.9160 | 0.9389 | 8 | 88.8321 |

## Gold newly reachable via dense (vs hybrid)

| span | gold_term | freq_stratum | gold_rank |
| --- | --- | --- | --- |
| L carotid | structure of left carotid artery | torso | 61 |
| Feeding tube was removed | nasogastric tube removal | tail | 61 |
| Elevated Factor VIII level | hypercoagulability | torso | 72 |
| noncontrast study | ct without contrast | tail | 67 |
| Hemorrhage | arterial bleed | tail | 71 |
| LCX ostial 90% 
stenosis | circumflex branch of left coronary artery stenosis | tail | 68 |
| ventricular | brain ventricle | torso | 63 |
| distal LMCA | structure of distal end of left main coronary artery | tail | 61 |
| AVF | arteriovenous fistula | tail | 44 |
| spinal disease | disorder of spinal region | tail | 61 |
| LAD | structure of branch of left anterior descending artery | tail | 62 |
| painful | pain of skin | tail | 85 |
| no vision changes | emmetropia | tail | 61 |
| lower extremity elevation | elevation of lower limb | tail | 61 |
| red 
blood cell count increased | rbc count raised | tail | 61 |
