"""Export tail-eval data so an LLM ranker can be run on Colab (GPU).

Item 3: test whether a strong context LLM beats the leakage-free simple signals
on the tail (target: gated 0.561). The LLM only needs text, the mention with
its context and the candidate terms, so we export two small CSVs and let Colab
do the ranking. Baseline picks (oracle/dense/type/gated) are precomputed here so
the notebook can report the comparison without any of the caches.

    python stage2_export_llm_tail.py            # all tail eval mentions
"""
import pickle
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
BM = HERE.parent
sys.path.insert(0, str(BM)); sys.path.insert(0, str(HERE))

from candidate_booster import SnomedCandidateBooster, norm
from colab_full_candidate_eval import StubOntology
from data import (CandidateGenerator, build_test_mentions, load, split_notes,
                  section_headers, section_at, CONTEXT_WINDOW)
from evaluate import freq_stratum

TYPE4 = {"body structure", "finding", "procedure"}
VAL_FRAC = 0.2; SEED = 20260723; TOPK = 20
POOL_CACHE = HERE / ".stage2_pools_full.pkl"
TPROB_CACHE = HERE / ".stage2_typeprob_cache.pkl"
OUT = HERE / "llm_tail_data"


def coarse4(t):
    return t if t in TYPE4 else "other"


def mark(context, span):
    return context.replace(span, f"<< {span} >>", 1) if span else context


def main():
    OUT.mkdir(exist_ok=True)
    notes, ann = load()
    tr, te = split_notes(ann)
    cg = CandidateGenerator().fit(ann, tr)
    ms = build_test_mentions(notes, ann, te, cg)
    tf = Counter(int(c) for c in ann[ann.note_id.isin(tr)].concept_id)
    seen = set(cg.form_to_concepts.keys())

    D = pickle.load(open(POOL_CACHE, "rb")); tp = pickle.load(open(TPROB_CACHE, "rb"))
    pools, dcos, coarse, lexsc = D["pools"], D["dcos"], D["coarse"], D["lexsc"]

    tn = sorted(te); import random; random.Random(SEED).shuffle(tn)
    val_notes = set(tn[: max(1, int(len(tn) * VAL_FRAC))])
    tail = [m for m in ms if m.note_id not in val_notes
            and freq_stratum(m.gold_concept, tf) == "tail"
            and (m.note_id, m.start) in pools and (m.note_id, m.start) in tp]
    print(f"tail eval mentions: {len(tail)}")

    booster = SnomedCandidateBooster(cg, StubOntology(), release_dir=".", mode="synonym")

    def score_pick(m, wd, wt, wl):
        key = (m.note_id, m.start); pool = pools[key][:TOPK]
        dc, lx, cz, pr = dcos[key], lexsc[key], coarse[key], tp[key]
        return max(pool, key=lambda c: wd*dc.get(c,0.0)+wt*pr.get(coarse4(cz[c]),0.0)+wl*lx.get(c,0.0)) if pool else -1
    def gated(m):
        w = (0,0.5,0) if norm(m.span) in seen else (0.5,0.5,0); return score_pick(m,*w)

    m_rows, c_rows = [], []
    for m in tail:
        key = f"{m.note_id}|{m.start}"
        pool = pools[(m.note_id, m.start)][:TOPK]
        m_rows.append({
            "key": key,
            "mention_text": f"Section: {m.section or 'NA'} | Mention: {m.span} | Context: {mark(m.context, m.span)}",
            "gold": m.gold_concept,
            "oracle_ok": int(m.gold_concept in set(pool)),
            "dense_pick": score_pick(m, 1, 0, 0),
            "type_pick": score_pick(m, 0, 1, 0),
            "gated_pick": gated(m),
        })
        for rank, c in enumerate(pool):
            c_rows.append({"key": key, "rank": rank, "cand_id": int(c),
                           "cand_term": booster.label(c)})

    pd.DataFrame(m_rows).to_csv(OUT / "llm_tail_mentions.csv", index=False)
    pd.DataFrame(c_rows).to_csv(OUT / "llm_tail_candidates.csv", index=False)
    # baseline accuracy for reference
    import numpy as np
    mdf = pd.DataFrame(m_rows)
    for col in ("dense_pick", "type_pick", "gated_pick"):
        acc = (mdf[col] == mdf["gold"]).mean()
        print(f"  baseline {col:12} tail acc = {acc:.4f}")
    print(f"  oracle (gold in top-{TOPK}) = {mdf['oracle_ok'].mean():.4f}")
    print(f"exported {len(m_rows)} mentions, {len(c_rows)} candidate rows -> {OUT}")


if __name__ == "__main__":
    main()
