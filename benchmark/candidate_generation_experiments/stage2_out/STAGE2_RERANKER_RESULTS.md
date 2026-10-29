# Stage 2 — Learned Reranker over RRF@60 (fixed 600)

train mentions=6000 · features=13

| ranker | all | head | torso | tail |
| --- | --- | --- | --- | --- |
| oracle | 0.9783 | 0.9973 | 0.9784 | 0.8953 |
| reranker | 0.7933 | 0.9307 | 0.7482 | 0.2674 |
