# Stage 2 — Full-23k dense+type reranker (leakage-free)

eval n=18676 (val n=4455) · best weights dense=1.0 type=1.0 lex=0.0

| policy | all | head | torso | tail |
| --- | --- | --- | --- | --- |
| oracle | 0.9699 | 0.9938 | 0.9794 | 0.8600 |
| dense_only | 0.4765 | 0.4541 | 0.5240 | 0.4934 |
| type_only | 0.6961 | 0.7536 | 0.6997 | 0.4609 |
| dense+type+lex(1.0,0.0) | 0.6090 | 0.6283 | 0.6090 | 0.5321 |
