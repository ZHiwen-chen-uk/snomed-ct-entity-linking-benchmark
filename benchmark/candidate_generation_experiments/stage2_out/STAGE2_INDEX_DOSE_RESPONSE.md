# Dose-response: the in-index blind spot as a function of training mismatch

One quantity is varied, the share of training positives whose gold concept is absent from the training-derived index, while the pool, the architecture, the hyperparameters and the evaluation are held fixed. `rho = 0` is the standard model; the real leave-one-note-out matrix is shown on the same axis for reference. The evaluation split's own out-of-index rate is 0.0789 of all mentions.

| condition | train_out_of_index_rate | all | head | torso | tail | tail_out_of_index | tail_in_index |
| --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic rho=0.00 | 0.0000 | 0.7885 | 0.9395 | 0.7476 | 0.2472 | 0.0305 | 0.4770 |
| synthetic rho=0.02 | 0.0216 | 0.8153 | 0.9326 | 0.7261 | 0.4825 | 0.5054 | 0.4583 |
| synthetic rho=0.05 | 0.0518 | 0.8146 | 0.9319 | 0.7254 | 0.4818 | 0.5115 | 0.4504 |
| synthetic rho=0.10 | 0.1000 | 0.8159 | 0.9345 | 0.7270 | 0.4777 | 0.5122 | 0.4410 |
| synthetic rho=0.20 | 0.1991 | 0.8153 | 0.9342 | 0.7224 | 0.4818 | 0.5197 | 0.4417 |
| synthetic rho=0.40 | 0.3971 | 0.8121 | 0.9313 | 0.7151 | 0.4836 | 0.5224 | 0.4424 |
| synthetic rho=0.80 | 0.8007 | 0.7975 | 0.9243 | 0.6855 | 0.4619 | 0.5204 | 0.4000 |
| synthetic rho=1.00 | 0.9994 | 0.0411 | 0.0000 | 0.0000 | 0.2682 | 0.5210 | 0.0000 |
| leave-one-note-out (real) | 0.0732 | 0.8689 | 0.9520 | 0.8314 | 0.5939 | 0.4810 | 0.7137 |

If the blind spot were an idiosyncrasy of this corpus there would be no reason for out-of-index tail accuracy to track this one quantity. If it is the claimed mechanism, the curve should rise steeply from zero and then saturate, while head accuracy erodes only once the training prior has been destroyed.
