"""Full-scale restatement of claims that were originally measured on subsamples.

Writing rule for the thesis: every reported number must come from the full test
set (23,131 mentions) or the canonical eval split (18,676), never from the fixed
600 / 150-mention pilots. This script recomputes, on the canonical split, the
four claims that were previously only available at pilot scale:

  1. recall-miss vs ranking-miss decomposition of the gated ranker's errors
     (replaces "9 of 14 tail errors were recall misses", n=150)
  2. gold-in-training-index rate, train mentions vs test mentions, by stratum
     (the mechanism behind the supervised reranker's tail collapse)
  3. pool ceilings at k = 1/10/20/30/60 by stratum
     (replaces the "top-20 tail ceiling 0.753" figure taken from an ~800 subset)
  4. acronym-slice size and reachability
     (replaces "131 of 600, ~93% reachable")

All of it runs off the caches built by stage2_gated.py, so it is seconds, not
hours. Output: stage2_out/FULLSCALE_AUDIT.md + .csv

    python stage_audit_fullscale.py
"""
import os
import pickle
import random
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
BM = HERE.parent
sys.path.insert(0, str(BM)); sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BM / "group_a_transformer"))

from abbrev_expander import acronym_tokens
from candidate_booster import norm
from data import CandidateGenerator, build_test_mentions, load, split_notes
from evaluate import freq_stratum
from stage2_gated import build_or_load_pools, pick, TFMR_DIR, VAL_FRAC, SEED
from stage2_reranker import TypeProba

STRATA = ("head", "torso", "tail")


def by_stratum(mentions, train_freq):
    out = {s: [] for s in STRATA}
    for m in mentions:
        out[freq_stratum(m.gold_concept, train_freq)].append(m)
    return out


def main():
    t0 = time.time()
    notes, ann = load()
    train_notes, test_notes = split_notes(ann)
    cg = CandidateGenerator().fit(ann, train_notes)
    mentions = build_test_mentions(notes, ann, test_notes, cg)
    train_freq = Counter(int(c) for c in ann[ann.note_id.isin(train_notes)].concept_id)
    seen_forms = set(cg.form_to_concepts.keys())
    print(f"test mentions (full): {len(mentions)}  ({time.time()-t0:.0f}s)", flush=True)

    # canonical eval split, identical construction to stage2_gated / stage2_unified
    tn = sorted(test_notes)
    random.Random(SEED).shuffle(tn)
    val_notes = set(tn[:max(1, int(len(tn) * VAL_FRAC))])
    ev = [m for m in mentions if m.note_id not in val_notes]
    print(f"canonical eval split: {len(ev)}", flush=True)

    D = build_or_load_pools(mentions, cg)
    for m in mentions:
        m.candidates = D["pools"][(m.note_id, m.start)]
    print(f"pools ready ({time.time()-t0:.0f}s)", flush=True)

    tp = TypeProba(TFMR_DIR)
    tprob = tp.proba(mentions)
    print(f"type probs ready ({time.time()-t0:.0f}s)", flush=True)

    # gate weights as locked in stage2_gated.py: seen-form -> type, novel -> dense+type
    W_SEEN, W_NOVEL = (0, 0.5, 0), (0.5, 0.5, 0)

    def gated(m):
        w = W_SEEN if norm(m.span) in seen_forms else W_NOVEL
        return pick(m, D, tprob, *w)

    rows = []
    ev_strat = by_stratum(ev, train_freq)

    # ---- 1. error decomposition: recall miss vs ranking miss -----------------
    print("[1] error decomposition", flush=True)
    for scope, subset in [("all", ev)] + [(s, ev_strat[s]) for s in STRATA]:
        n = len(subset)
        correct = reach_miss = rank_miss = 0
        for m in subset:
            pool = set(m.candidates)
            p = gated(m)
            if p == m.gold_concept:
                correct += 1
            elif m.gold_concept not in pool:
                reach_miss += 1
            else:
                rank_miss += 1
        err = n - correct
        rows.append({
            "block": "1_error_decomposition", "scope": scope, "n": n,
            "top1_acc": round(correct / n, 4),
            "errors": err,
            "recall_miss": reach_miss,
            "recall_miss_share_of_err": round(reach_miss / err, 4) if err else None,
            "ranking_miss": rank_miss,
            "ranking_miss_share_of_err": round(rank_miss / err, 4) if err else None,
        })

    # ---- 2. gold-in-training-index asymmetry --------------------------------
    # The index a supervised reranker learns from is the set of concepts the
    # training annotations attach to any surface form. For a TRAIN mention the
    # gold is in it by construction; for a TEST tail mention it often is not.
    print("[2] gold-in-training-index asymmetry", flush=True)
    train_index = set()
    for _form, concepts in cg.form_to_concepts.items():
        train_index.update(int(c) for c in concepts)

    tr_ann = ann[ann.note_id.isin(train_notes)]
    tr_pairs = [(int(c), freq_stratum(int(c), train_freq)) for c in tr_ann.concept_id]
    for scope in ("all",) + STRATA:
        sub = tr_pairs if scope == "all" else [p for p in tr_pairs if p[1] == scope]
        if not sub:
            continue
        rows.append({"block": "2_gold_in_train_index", "scope": f"TRAIN_{scope}",
                     "n": len(sub),
                     "gold_in_train_index": round(
                         sum(1 for c, _ in sub if c in train_index) / len(sub), 4)})
    for scope, subset in [("all", ev)] + [(s, ev_strat[s]) for s in STRATA]:
        rows.append({"block": "2_gold_in_train_index", "scope": f"TEST_{scope}",
                     "n": len(subset),
                     "gold_in_train_index": round(
                         sum(1 for m in subset if m.gold_concept in train_index) / len(subset), 4)})

    # ---- 3. pool ceilings at k ----------------------------------------------
    print("[3] pool ceilings", flush=True)
    for scope, subset in [("all", ev)] + [(s, ev_strat[s]) for s in STRATA]:
        r = {"block": "3_pool_ceiling", "scope": scope, "n": len(subset)}
        for k in (1, 10, 20, 30, 60):
            r[f"ceiling@{k}"] = round(sum(
                1 for m in subset if m.gold_concept in set(m.candidates[:k])) / len(subset), 4)
        rows.append(r)

    # ---- 4. acronym slice at full scale -------------------------------------
    print("[4] acronym slice", flush=True)
    acro = [m for m in ev if acronym_tokens(m.span)]
    acro_strat = by_stratum(acro, train_freq)
    for scope, subset in [("all", acro)] + [(s, acro_strat[s]) for s in STRATA]:
        if not subset:
            continue
        rows.append({
            "block": "4_acronym_slice", "scope": scope, "n": len(subset),
            "share_of_eval": round(len(subset) / len(ev if scope == "all" else ev_strat[scope]), 4),
            "reachability": round(sum(
                1 for m in subset if m.gold_concept in set(m.candidates)) / len(subset), 4),
            "top1_acc": round(sum(1 for m in subset if gated(m) == m.gold_concept) / len(subset), 4),
        })

    df = pd.DataFrame(rows)
    out = HERE / "stage2_out"; out.mkdir(exist_ok=True)
    df.to_csv(out / "fullscale_audit.csv", index=False)

    lines = ["# Full-scale audit — claims restated on the canonical eval split", "",
             f"full test mentions {len(mentions)} · canonical eval split {len(ev)} "
             f"(head {len(ev_strat['head'])} / torso {len(ev_strat['torso'])} / tail {len(ev_strat['tail'])})", "",
             "Ranker = gated (seen-form -> type, novel-form -> dense+type), identical to stage2_gated.py.", ""]
    for block, title in [
        ("1_error_decomposition", "1. Error decomposition — is a gated-ranker error a retrieval miss or a ranking miss?"),
        ("2_gold_in_train_index", "2. Gold concept present in the training index (the supervised reranker's blind spot)"),
        ("3_pool_ceiling", "3. Pool ceiling at k (oracle reachability within the top k of the RRF pool)"),
        ("4_acronym_slice", "4. Acronym-shaped mentions — slice size, reachability, accuracy"),
    ]:
        sub = df[df.block == block].dropna(axis=1, how="all").drop(columns=["block"])
        cols = list(sub.columns)
        lines += [f"### {title}", "",
                  "| " + " | ".join(cols) + " |",
                  "| " + " | ".join("---" for _ in cols) + " |"]
        for _, r in sub.iterrows():
            lines.append("| " + " | ".join(
                "" if pd.isna(r[c]) else str(r[c]) for c in cols) + " |")
        lines.append("")
    (out / "FULLSCALE_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(df.to_string(index=False))
    print(f"\nwrote {out/'FULLSCALE_AUDIT.md'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
