"""Item 1: one comparable table, every ranker on the SAME RRF@60 pool and the
SAME eval split (18,676), by frequency stratum. Cheap rankers are recomputed
from cache here; the cross-encoder row is its Colab evaluation on the identical
eval split, included for a like-for-like comparison.

    python stage2_unified.py
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

from candidate_booster import norm
from data import CandidateGenerator, build_test_mentions, load, split_notes
from evaluate import freq_stratum
import test_candidate_generation as tcg

TYPE4 = {"body structure", "finding", "procedure"}
VAL_FRAC = 0.2
SEED = 20260723
POOL_CACHE = HERE / ".stage2_pools_full.pkl"
TPROB_CACHE = HERE / ".stage2_typeprob_cache.pkl"
# cross-encoder eval-split accuracy (from colab_crossencoder.py on the same split)
CROSS_ENCODER = {"all": 0.773, "head": 0.870, "torso": 0.681, "tail": 0.522}


def coarse4(t):
    return t if t in TYPE4 else "other"


def main():
    notes, ann = load()
    tr, te = split_notes(ann)
    cg = CandidateGenerator().fit(ann, tr)
    ms = build_test_mentions(notes, ann, te, cg)
    tf = Counter(int(c) for c in ann[ann.note_id.isin(tr)].concept_id)
    seen = set(cg.form_to_concepts.keys())

    D = pickle.load(open(POOL_CACHE, "rb"))
    tp = pickle.load(open(TPROB_CACHE, "rb"))
    pools, dcos, coarse = D["pools"], D["dcos"], D["coarse"]

    # same eval split as stage2_gated
    tn = sorted(te)
    import random
    random.Random(SEED).shuffle(tn)
    val_notes = set(tn[: max(1, int(len(tn) * VAL_FRAC))])
    ev = [m for m in ms if m.note_id not in val_notes
          and (m.note_id, m.start) in pools and (m.note_id, m.start) in tp]
    print(f"eval mentions: {len(ev)}")

    def score_pick(m, wd, wt, wl):
        key = (m.note_id, m.start); pool = pools[key]
        dc, lx, cz, pr = dcos[key], D["lexsc"][key], coarse[key], tp[key]
        best, bs = None, -1e9
        for c in pool:
            s = wd * dc.get(c, 0.0) + wt * pr.get(coarse4(cz[c]), 0.0) + wl * lx.get(c, 0.0)
            if s > bs:
                best, bs = c, s
        return best

    def gated(m):
        w = (0, 0.5, 0) if norm(m.span) in seen else (0.5, 0.5, 0)
        return score_pick(m, *w)

    picks = {
        "oracle (ceiling)": {(m.note_id, m.start): (m.gold_concept if m.gold_concept in set(pools[(m.note_id, m.start)]) else None) for m in ev},
        "rrf_rank1 (no ranker)": {(m.note_id, m.start): (pools[(m.note_id, m.start)][0] if pools[(m.note_id, m.start)] else None) for m in ev},
        "dense_only": {(m.note_id, m.start): score_pick(m, 1, 0, 0) for m in ev},
        "type_only (Group A soft)": {(m.note_id, m.start): score_pick(m, 0, 1, 0) for m in ev},
        "gated (type / dense)": {(m.note_id, m.start): gated(m) for m in ev},
    }

    gold = {(m.note_id, m.start): m.gold_concept for m in ev}
    strat = {(m.note_id, m.start): freq_stratum(m.gold_concept, tf) for m in ev}
    keys = list(gold)

    def acc(pick, ks):
        return round(sum(1 for k in ks if pick.get(k) == gold[k]) / len(ks), 4) if ks else None

    rows = []
    for name, pick in picks.items():
        r = {"ranker": name, "all": acc(pick, keys)}
        for s in ("head", "torso", "tail"):
            r[s] = acc(pick, [k for k in keys if strat[k] == s])
        rows.append(r)
    rows.append({"ranker": "cross-encoder (PubMedBERT)", **CROSS_ENCODER})
    df = pd.DataFrame(rows)

    out = HERE / "stage2_out"; out.mkdir(exist_ok=True)
    df.to_csv(out / "stage2_unified.csv", index=False)
    n_by = {s: sum(1 for k in keys if strat[k] == s) for s in ("head", "torso", "tail")}
    lines = ["# Stage 2 — unified ranker comparison (same RRF@60 pool, same eval split)", "",
             f"eval n={len(ev)}  (head {n_by['head']} / torso {n_by['torso']} / tail {n_by['tail']})", "",
             "cross-encoder row = its Colab evaluation on the identical eval split.", "",
             tcg.df_to_markdown(df)]
    (out / "STAGE2_UNIFIED.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + df.to_string(index=False))
    print(f"\nsaved -> {out/'STAGE2_UNIFIED.md'}")


if __name__ == "__main__":
    main()
