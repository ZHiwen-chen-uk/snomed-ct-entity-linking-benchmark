# Stage 2 — unified ranker comparison (same RRF@60 pool, same eval split)

eval n=18676  (head 11446 / torso 4366 / tail 2864)

cross-encoder row = its Colab evaluation on the identical eval split.

| ranker | all | head | torso | tail |
| --- | --- | --- | --- | --- |
| oracle (ceiling) | 0.9697 | 0.9938 | 0.9792 | 0.8589 |
| rrf_rank1 (no ranker) | 0.5510 | 0.5683 | 0.5957 | 0.4134 |
| dense_only | 0.4765 | 0.4541 | 0.5240 | 0.4934 |
| type_only (Group A soft) | 0.6961 | 0.7537 | 0.6995 | 0.4609 |
| gated (type / dense) | 0.7131 | 0.7521 | 0.7109 | 0.5608 |
| cross-encoder (PubMedBERT) | 0.7730 | 0.8700 | 0.6810 | 0.5220 |
