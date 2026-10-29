# Stage 3 — Ontology as hard constraint vs soft prior (full 23k test)

n=23131 · base ranker = SapBERT cosine · soft alpha=0.5 · hard filter: cross-branch only, confidence >= 0.9

Hard filter fired on 22690 mentions (98.1%); deleted reachable gold in 541 (2.42% false-exclusion of reachable).

| policy | all | head | torso | tail | cross-branch | sibling | unambiguous |
| --- | --- | --- | --- | --- | --- | --- | --- |
| oracle | 0.9682 | 0.9931 | 0.9780 | 0.8601 | 0.9682 | 1.0000 | None |
| dense_only | 0.4809 | 0.4570 | 0.5283 | 0.5046 | 0.4807 | 1.0000 | None |
| soft_type | 0.6056 | 0.6218 | 0.6039 | 0.5469 | 0.6055 | 1.0000 | None |
| hard_constraint | 0.6135 | 0.6311 | 0.6138 | 0.5466 | 0.6134 | 1.0000 | None |
