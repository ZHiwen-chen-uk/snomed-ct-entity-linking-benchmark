"""Stage 3: ontology as a hard CONSTRAINT vs a soft PRIOR (full 23k test).

Motivated by the shift from post-hoc reranking to constraining the candidate
space. Using the same context-driven type signal (transformer-A) three ways,
holding the base ranker (dense cosine) fixed:

  1. dense_only        rank the pool by SapBERT cosine, no type signal
  2. soft_type         score = cosine + alpha * P(type(c) | mention)      (Group A soft)
  3. hard_constraint   on cross-branch pools with a confident prediction,
                       DELETE candidates whose coarse type != predicted type,
                       then rank the survivors by cosine                  (definitional filter)

Reports accuracy by frequency stratum and by ambiguity type (cross-branch /
sibling / unambiguous), plus the hard filter's false-exclusion rate (gold that
was reachable but deleted). All inputs are cached, so this runs in seconds.

    python stage3_ontology_constraint.py
"""
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BM = HERE.parent
sys.path.insert(0, str(BM)); sys.path.insert(0, str(HERE))

from data import CandidateGenerator, build_test_mentions, load, split_notes
from evaluate import freq_stratum
import test_candidate_generation as tcg

POOL_CACHE = HERE / ".stage2_pools_full.pkl"
TPROB_CACHE = HERE / ".stage2_typeprob_cache.pkl"
TYPE4 = {"body structure", "finding", "procedure"}     # transformer-A labels (+ "other")
ALPHA = 0.5                                             # soft-prior weight
CONF = 0.90                                             # hard-filter confidence gate


def coarse4(t):
    return t if t in TYPE4 else "other"


def ambiguity(pool, coarse):
    if len(pool) <= 1:
        return "unambiguous"
    return "cross-branch" if len({coarse[c] for c in pool}) > 1 else "sibling"


def main():
    notes, ann = load()
    train_notes, test_notes = split_notes(ann)
    cg = CandidateGenerator().fit(ann, train_notes)
    mentions = build_test_mentions(notes, ann, test_notes, cg)
    train_freq = Counter(int(c) for c in ann[ann.note_id.isin(train_notes)].concept_id)

    D = pickle.load(open(POOL_CACHE, "rb"))
    tprob = pickle.load(open(TPROB_CACHE, "rb"))
    pools, dcos, coarse_all = D["pools"], D["dcos"], D["coarse"]

    mentions = [m for m in mentions if (m.note_id, m.start) in pools
                and (m.note_id, m.start) in tprob]
    print(f"test mentions with pool + type prob: {len(mentions)}")

    picks = {p: {} for p in ("dense_only", "soft_type", "hard_constraint", "oracle")}
    fired = del_gold = reachable = 0
    amb_of, strat_of = {}, {}

    for m in mentions:
        key = (m.note_id, m.start)
        pool = pools[key]
        dc = dcos[key]
        cz = coarse_all[key]                              # full coarse type per candidate
        tp = tprob[key]
        gold = m.gold_concept
        amb_of[key] = ambiguity(pool, cz)
        strat_of[key] = freq_stratum(gold, train_freq)
        in_pool = gold in set(pool)
        reachable += int(in_pool)

        picks["oracle"][key] = gold if in_pool else None
        # base: dense cosine
        picks["dense_only"][key] = max(pool, key=lambda c: dc.get(c, 0.0)) if pool else None
        # soft: cosine + alpha * P(type)
        picks["soft_type"][key] = max(
            pool, key=lambda c: dc.get(c, 0.0) + ALPHA * tp.get(coarse4(cz[c]), 0.0)) if pool else None
        # hard: cross-branch + confident -> keep predicted-type candidates, then cosine
        pred = max(tp, key=tp.get); conf = tp[pred]
        types4 = {coarse4(cz[c]) for c in pool}
        do_filter = (len({cz[c] for c in pool}) > 1) and conf >= CONF and pred in types4
        if do_filter:
            kept = [c for c in pool if coarse4(cz[c]) == pred]
            fired += 1
            if in_pool and gold not in set(kept):
                del_gold += 1
        else:
            kept = pool
        picks["hard_constraint"][key] = max(kept, key=lambda c: dc.get(c, 0.0)) if kept else None

    allkeys = [(m.note_id, m.start) for m in mentions]
    gold_of = {(m.note_id, m.start): m.gold_concept for m in mentions}

    def acc2(pick, keys):
        if not keys:
            return None
        return round(sum(1 for k in keys if pick.get(k) == gold_of[k]) / len(keys), 4)

    rows = []
    for pol in ("oracle", "dense_only", "soft_type", "hard_constraint"):
        r = {"policy": pol, "all": acc2(picks[pol], allkeys)}
        for s in ("head", "torso", "tail"):
            r[s] = acc2(picks[pol], [k for k in allkeys if strat_of[k] == s])
        for a in ("cross-branch", "sibling", "unambiguous"):
            r[a] = acc2(picks[pol], [k for k in allkeys if amb_of[k] == a])
        rows.append(r)
    df = pd.DataFrame(rows)

    out = HERE / "stage2_out"; out.mkdir(exist_ok=True)
    df.to_csv(out / "stage3_ontology_constraint.csv", index=False)
    n = len(mentions)
    lines = ["# Stage 3 — Ontology as hard constraint vs soft prior (full 23k test)", "",
             f"n={n} · base ranker = SapBERT cosine · soft alpha={ALPHA} · "
             f"hard filter: cross-branch only, confidence >= {CONF}", "",
             f"Hard filter fired on {fired} mentions ({fired/n:.1%}); "
             f"deleted reachable gold in {del_gold} ({del_gold/max(1,reachable):.2%} false-exclusion of reachable).", "",
             tcg.df_to_markdown(df)]
    (out / "STAGE3_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nhard filter fired {fired}/{n} ({fired/n:.1%}); false-exclusion {del_gold}/{reachable} ({del_gold/max(1,reachable):.2%})")
    print(df.to_string(index=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
