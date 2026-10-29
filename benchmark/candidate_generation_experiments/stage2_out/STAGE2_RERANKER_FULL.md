# Stage 2 — supervised reranker, FULL SCALE

train = all 52078 train-note mentions ({'head': 35929, 'torso': 12280, 'tail': 3869}) · eval = canonical split 18676 ({'head': 11446, 'tail': 2864, 'torso': 4366}) · features=13 · pool = weighted RRF@60

| ranker | all | head | torso | tail |
| --- | --- | --- | --- | --- |
| oracle (ceiling) | 0.9697 | 0.9938 | 0.9792 | 0.8593 |
| supervised reranker | 0.7885 | 0.9395 | 0.7476 | 0.2472 |
