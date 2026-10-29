# Abbreviation expansion — FULL SCALE (all test mentions)

n=23131 · pool = weighted RRF@60 · abbrev added as a 6th retriever (weight 1.5, precise mode)

`gold_only_with_abbrev` = gold reachable ONLY when the expander is on; `gold_lost_with_abbrev` = gold displaced out of the pool by the extra list.

| scope | n | reach_base | reach_abbrev | d_reach | r10_base | r10_abbrev | d_r10 | r30_base | r30_abbrev | d_r30 | gold_only_with_abbrev | gold_lost_with_abbrev | mean_cand_base | mean_cand_abbrev |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 23131 | 0.9684 | 0.9685 | 0.0001 | 0.8941 | 0.9050 | 0.0109 | 0.9521 | 0.9527 | 0.0006 | 3 | 0 | 60.0000 | 60.0000 |
| head | 14124 | 0.9931 | 0.9931 | 0.0000 | 0.9456 | 0.9586 | 0.0130 | 0.9905 | 0.9905 | 0.0000 | 0 | 0 | 60.0000 | 60.0000 |
| torso | 5277 | 0.9782 | 0.9784 | 0.0002 | 0.8802 | 0.8935 | 0.0133 | 0.9615 | 0.9632 | 0.0017 | 1 | 0 | 60.0000 | 60.0000 |
| tail | 3730 | 0.8609 | 0.8614 | 0.0005 | 0.7185 | 0.7185 | 0.0000 | 0.7936 | 0.7944 | 0.0008 | 2 | 0 | 60.0000 | 60.0000 |
| acronym_slice | 5338 | 0.9695 | 0.9700 | 0.0005 | 0.9026 | 0.9500 | 0.0474 | 0.9590 | 0.9612 | 0.0022 | 3 | 0 | 60.0000 | 60.0000 |
| acronym_tail | 553 | 0.7993 | 0.8029 | 0.0036 | 0.6564 | 0.6564 | 0.0000 | 0.7324 | 0.7378 | 0.0054 | 2 | 0 | 60.0000 | 60.0000 |
