"""Pull out real failure examples, one stage at a time, split by frequency.

The results elsewhere are all numbers, so there is no way to see what a detection
miss, a retrieval miss or a ranking error actually looks like on the page. This
script reads them out of caches that already exist, which means nothing has to be
retrained:

    .stage2_picks_eval.pkl        per-mention top-1 pick for all five rankers
    .stage2_pools_full.pkl        the weighted RRF@60 pool on gold spans
    stage0_ner/result/stage0_pred_spans_vanilla.jsonl   detected spans

It writes five groups, one per stage plus two that isolate the study's central
mechanism:

    detection_miss    no predicted span reaches IoU >= 0.5 with the gold span
    detection_partial detected at 0.5 <= IoU < 1.0, so the boundary is wrong
    retrieval_miss    gold concept absent from the 60-candidate pool
    ranking_error     gold in the pool, another candidate ranked first
    blind_spot        out-of-index tail mention the standard reranker gets wrong
                      and the LONO reranker gets right, with the training
                      frequency of what the standard model preferred instead

Nothing is sampled. Candidates are sorted by an explicit key and the first few
are taken, so the same examples come back on a re-run.

    KMP_DUPLICATE_LIB_OK=TRUE python stage_error_examples.py
"""
import json
import pickle
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
BM = HERE.parent
sys.path.insert(0, str(BM)); sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BM / "group_a_transformer"))

from data import CandidateGenerator, build_test_mentions, load, split_notes
from evaluate import freq_stratum
from ontology import SnomedOntology
from stage0_chain_e2e import IOU_THR, PRED_DIR, iou
from stage2_error_taxonomy import classify
from stage2_gated import SEED, SNOMED_DIR, VAL_FRAC
import test_candidate_generation as tcg

PICKS = HERE / ".stage2_picks_eval.pkl"
GOLD_POOLS = HERE / ".stage2_pools_full.pkl"
N_PER_CELL = 3
STRATA = ("head", "torso", "tail")


def stratum_of(m, train_freq):
    s = freq_stratum(m.gold_concept, train_freq)
    if s == "tail":
        return "tail (out-of-index)" if train_freq.get(m.gold_concept, 0) == 0 \
            else "tail (in-index)"
    return s


def context_of(text, start, end, width=34):
    """A one-line window around the span, with the span itself in brackets."""
    lo, hi = max(0, start - width), min(len(text), end + width)
    left = re.sub(r"\s+", " ", text[lo:start]).strip()
    mid = re.sub(r"\s+", " ", text[start:end]).strip()
    right = re.sub(r"\s+", " ", text[end:hi]).strip()
    return f"...{left} [{mid}] {right}..."


def main():
    t0 = time.time()
    notes, ann = load()
    train_notes, test_notes = split_notes(ann)
    cg = CandidateGenerator().fit(ann, train_notes)
    train_freq = Counter(int(c) for c in
                         ann[ann.note_id.isin(train_notes)].concept_id)
    import random as _r
    tn = sorted(test_notes); _r.Random(SEED).shuffle(tn)
    val_notes = set(tn[:max(1, int(len(tn) * VAL_FRAC))])
    eval_notes = set(tn) - val_notes
    gold = build_test_mentions(notes, ann, test_notes, cg)
    ev = [m for m in gold if m.note_id in eval_notes]
    text_by_id = dict(zip(notes.note_id, notes.text))
    print(f"eval mentions {len(ev):,}  ({time.time()-t0:.0f}s)", flush=True)

    ont = SnomedOntology(SNOMED_DIR)
    print(f"ontology ready  ({time.time()-t0:.0f}s)", flush=True)

    def term(cid):
        return ont.preferred_term(int(cid)) or f"[{cid}]"

    D = pickle.load(open(GOLD_POOLS, "rb"))
    picks = pickle.load(open(PICKS, "rb"))["picks"]
    lono, std = picks["supervised, LONO"], picks["supervised, standard"]

    # ---- Stage 0: detection ------------------------------------------------
    pred_recs = [json.loads(l) for l in open(
        PRED_DIR / "stage0_pred_spans_vanilla.jsonl", encoding="utf-8")]
    pred_by_note = {}
    for rec in pred_recs:
        if rec["note_id"] in eval_notes:
            pred_by_note.setdefault(rec["note_id"], []).extend(
                (int(s), int(e)) for s, e in rec["pred_spans"])

    rows = []
    for m in ev:
        k = (m.note_id, m.start)
        st = stratum_of(m, train_freq)
        text = text_by_id[m.note_id]
        best = max((iou((m.start, m.end), p)
                    for p in pred_by_note.get(m.note_id, ())), default=0.0)
        base = {"stratum": st, "note_id": m.note_id, "start": m.start,
                "mention": m.span, "n_words": len(m.span.split()),
                "gold": m.gold_concept, "gold_term": term(m.gold_concept),
                "train_freq": train_freq.get(m.gold_concept, 0),
                "best_iou": round(best, 3),
                "context": context_of(text, m.start, m.end)}
        # Detection is judged on predicted spans; retrieval and ranking are
        # judged on GOLD spans, as everywhere else in the study, so a mention
        # that detection missed can still supply a ranking example.
        if best < IOU_THR:
            rows.append({**base, "group": "detection_miss", "predicted": None,
                         "predicted_term": None, "relation": None})
        elif best < 1.0:
            rows.append({**base, "group": "detection_partial", "predicted": None,
                         "predicted_term": None, "relation": None})
        pool = D["pools"].get(k)
        if pool is None:
            continue
        if m.gold_concept not in pool:
            rows.append({**base, "group": "retrieval_miss", "predicted": None,
                         "predicted_term": None, "relation": None})
            continue
        p_lono, p_std = lono.get(k), std.get(k)
        gold_pos = pool.index(m.gold_concept) + 1
        if p_lono is not None and p_lono != m.gold_concept:
            rows.append({**base, "group": "ranking_error", "predicted": int(p_lono),
                         "predicted_term": term(p_lono),
                         "relation": classify(ont, p_lono, m.gold_concept),
                         "pred_train_freq": train_freq.get(int(p_lono), 0),
                         "gold_pool_pos": gold_pos})
        if (st == "tail (out-of-index)" and p_std is not None
                and p_std != m.gold_concept and p_lono == m.gold_concept):
            rows.append({**base, "group": "blind_spot", "predicted": int(p_std),
                         "predicted_term": term(p_std),
                         "relation": classify(ont, p_std, m.gold_concept),
                         "pred_train_freq": train_freq.get(int(p_std), 0),
                         "gold_pool_pos": gold_pos})

    df = pd.DataFrame(rows)
    out = HERE / "stage2_out"; out.mkdir(exist_ok=True)
    df.to_csv(out / "stage_error_examples_all.csv", index=False)

    # ---- where do the out-of-index tail picks go? ---------------------------
    # The examples suggest the standard model funnels these mentions onto a few
    # very frequent concepts. Quantify it, so the qualitative table is backed by
    # a number: the training frequency of what each ranker picks, and how
    # concentrated those picks are. Ties are broken by concept id, never by
    # dictionary order.
    ooi = [m for m in ev if stratum_of(m, train_freq) == "tail (out-of-index)"]
    funnel = []
    for rname in ["supervised, standard", "supervised, LONO", "gated"]:
        pk = picks[rname]
        chosen, demoted, n_err = [], 0, 0
        for m in ooi:
            k = (m.note_id, m.start)
            if k not in pk:
                continue
            c = int(pk[k])
            chosen.append(c)
            pool = D["pools"].get(k) or []
            # A ranking error whose gold was ALREADY first in the unranked pool
            # is one the ranker actively demoted, which separates a self-inflicted
            # error from a genuinely hard one.
            if c != m.gold_concept and m.gold_concept in pool:
                n_err += 1
                demoted += pool.index(m.gold_concept) == 0
        freqs = pd.Series([train_freq.get(c, 0) for c in chosen])
        cnt = Counter(chosen)
        top = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
        funnel.append({
            "ranker": rname, "n_picks": len(chosen),
            "pick_train_freq_median": int(freqs.median()),
            "picks_freq_zero_share": round(float((freqs == 0).mean()), 4),
            "distinct_concepts": len(cnt),
            "top3_share": round(sum(c for _, c in top) / len(chosen), 4),
            "top_pick": f"{top[0][0]} {term(top[0][0])}",
            "top_pick_share": round(top[0][1] / len(chosen), 4),
            "top_pick_train_freq": train_freq.get(top[0][0], 0),
            "n_ranking_errors": n_err,
            "errors_demoted_from_pool_top": round(demoted / n_err, 4) if n_err else None})
    dff = pd.DataFrame(funnel)
    dff.to_csv(out / "stage_error_ooi_funnel.csv", index=False)
    print("\nout-of-index tail: where the picks go")
    print(dff.to_string(index=False))
    print(f"\ncounts by group x stratum ({time.time()-t0:.0f}s)")
    print(pd.crosstab(df.group, df.stratum).to_string())

    # ---- deterministic shortlist -------------------------------------------
    # Preference within a cell: for detection, the longest mention (the
    # compositional-span story); for retrieval and ranking, the mention whose
    # gold is rarest, then longest surface form; for the blind spot, the largest
    # training frequency of the concept wrongly preferred.
    keys = {"detection_miss": lambda r: (-r.n_words, r.mention),
            "detection_partial": lambda r: (-r.n_words, r.mention),
            "retrieval_miss": lambda r: (r.train_freq, -r.n_words, r.mention),
            "ranking_error": lambda r: (r.train_freq, -r.n_words, r.mention),
            "blind_spot": lambda r: (-r.get("pred_train_freq", 0), r.mention)}
    # Concepts outside the International release carry no fully specified name
    # (the corpus annotates a few of these), and an unnamed concept cannot be
    # shown to a reader, so those rows are kept in the full CSV and excluded
    # here. Repeated surface forms are also collapsed: the same mention text
    # recurs across notes and adds nothing as a second example.
    named = df[~df.gold_term.str.startswith("concept:")
               & (df.predicted_term.isna()
                  | ~df.predicted_term.fillna("").str.startswith("concept:"))]
    picked = []
    for g in ["detection_miss", "detection_partial", "retrieval_miss",
              "ranking_error", "blind_spot"]:
        sub = named[named.group == g]
        for st in ["head", "torso", "tail (in-index)", "tail (out-of-index)"]:
            cell = sub[sub.stratum == st]
            if cell.empty:
                continue
            recs = sorted(cell.to_dict("records"),
                          key=lambda d: keys[g](pd.Series(d)))
            seen_forms, take = set(), []
            for r in recs:
                f = re.sub(r"\s+", " ", str(r["mention"])).strip().lower()
                if f in seen_forms:
                    continue
                seen_forms.add(f); take.append(r)
                if len(take) == N_PER_CELL:
                    break
            picked.extend(take)
    short = pd.DataFrame(picked)
    short.to_csv(out / "stage_error_examples_shortlist.csv", index=False)

    lines = ["# Qualitative failure examples", "",
             f"eval split {len(ev):,} mentions - detected spans from the vanilla "
             "detector - pool = weighted RRF@60 on gold spans - rankers read "
             "from the cached per-mention picks.", "",
             "Counts by group and stratum:", "",
             tcg.df_to_markdown(pd.crosstab(df.group, df.stratum).reset_index()),
             ""]
    for g in short.group.unique():
        lines += [f"## {g}", ""]
        for st in short[short.group == g].stratum.unique():
            lines.append(f"**{st}**")
            lines.append("")
            for r in short[(short.group == g) & (short.stratum == st)].itertuples():
                lines.append(f"- `{r.mention}` (gold {r.gold} *{r.gold_term}*, "
                             f"train freq {r.train_freq}, best IoU {r.best_iou})")
                if r.predicted_term:
                    extra = ""
                    if g == "blind_spot":
                        extra = (f", train freq "
                                 f"{int(getattr(r, 'pred_train_freq', 0) or 0)}")
                    lines.append(f"  - picked {int(r.predicted)} "
                                 f"*{r.predicted_term}* ({r.relation}{extra})")
                lines.append(f"  - {r.context}")
            lines.append("")
    (out / "STAGE_ERROR_EXAMPLES.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {out/'STAGE_ERROR_EXAMPLES.md'} "
          f"and two CSVs ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
