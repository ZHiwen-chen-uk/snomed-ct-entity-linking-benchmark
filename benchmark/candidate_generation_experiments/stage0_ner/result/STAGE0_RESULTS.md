# Stage-0 span detection — results

model `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` · epochs 4 · max_len 384 · train notes 204 / test notes 68

### vanilla

- gold 23131 · pred 21451
- exact  P 0.783 / R 0.726 / F1 0.754
- IoU>=0.5 P 0.850 / R 0.788 / F1 0.818  · mean best-IoU 0.789

| stratum | recall (IoU>=0.5) | recall (exact) | n_gold |
|---|---|---|---|
| head | 0.836 | 0.805 | 14124 |
| torso | 0.744 | 0.658 | 5277 |
| tail | 0.669 | 0.525 | 3730 |

### section_aware

- gold 23131 · pred 22460
- exact  P 0.778 / R 0.756 / F1 0.767
- IoU>=0.5 P 0.835 / R 0.811 / F1 0.823  · mean best-IoU 0.810

| stratum | recall (IoU>=0.5) | recall (exact) | n_gold |
|---|---|---|---|
| head | 0.862 | 0.839 | 14124 |
| torso | 0.775 | 0.701 | 5277 |
| tail | 0.666 | 0.521 | 3730 |

### Δ section_aware − vanilla (the controlled A/B)

- exact F1: +0.013  ·  IoU F1: +0.005
- tail recall (IoU): -0.003  ·  head recall (IoU): +0.026

