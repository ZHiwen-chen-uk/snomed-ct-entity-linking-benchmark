"""Selective prediction: accuracy-coverage curves and a human-review triage budget.

Everything else in this thesis measures where the pipeline fails. This asks the
deployment question that follows from it: given that the linker is wrong on
roughly a third of mentions, can it tell you WHICH third, so a human reviews
those and trusts the rest?

For each eval mention the ranker produces a confidence:

    maxprob  the winning candidate's score
    margin   winner minus runner-up  (usually the better signal for selection)

Sort by confidence, auto-accept the most confident fraction, route the rest to
review. Reported three ways:

  1. risk-coverage, accuracy on the auto-accepted portion at each coverage
  2. error capture, what share of ALL errors lands in the review queue
  3. post-review accuracy, corpus accuracy if a reviewer corrects the queue,
     which is the number a deployment would actually be judged on

An ORACLE ordering (all errors first) is included as the ceiling, so the gap
shows how much of the achievable triage the confidence signal really delivers.

The review queue's composition is also reported, because the interesting
question for a long-tailed task is whether triage preferentially surfaces the
rare concepts, the ones a clinician most needs flagged.

    KMP_DUPLICATE_LIB_OK=TRUE python stage2_abstention.py
"""
import pickle
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BM = HERE.parent
sys.path.insert(0, str(BM)); sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BM / "group_a_transformer"))

from candidate_booster import norm
from data import CandidateGenerator, build_test_mentions, load, split_notes
from evaluate import freq_stratum
from stage0_chain_e2e import GOLD_POOLS, GOLD_TPROB, tune_gate
from stage2_gated import SEED, TYPE4, VAL_FRAC
from stage2_feature_hygiene import FEAT_EVAL, FEAT_TRAIN, load_feats
from stage2_lono_training import LONO_TRAIN
import test_candidate_generation as tcg

BUDGETS = [0.05, 0.10, 0.20, 0.30, 0.50]        # fraction sent to human review
COVERAGES = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
SCOPES = ["all", "head", "torso", "tail", "tail_out_of_index", "tail_in_index"]


def scopes_of(m, train_freq):
    s = freq_stratum(m.gold_concept, train_freq)
    out = ["all", s]
    if s == "tail":
        out.append("tail_out_of_index" if train_freq.get(m.gold_concept, 0) == 0
                   else "tail_in_index")
    return out


def top2(scores):
    """(winner index, maxprob, margin) for one mention's candidate scores."""
    if len(scores) == 1:
        return 0, float(scores[0]), float(scores[0])
    o = np.argpartition(-scores, 1)[:2]
    a, b = (o[0], o[1]) if scores[o[0]] >= scores[o[1]] else (o[1], o[0])
    return int(a), float(scores[a]), float(scores[a] - scores[b])


def curves(ok, conf, name, signal):
    """risk-coverage, error capture and post-review accuracy for one signal."""
    n = len(ok)
    order = np.argsort(-conf, kind="stable")     # most confident first
    ok_sorted = ok[order]
    cum_ok = np.cumsum(ok_sorted)
    total_err = int((~ok).sum())

    rows = []
    for cov in COVERAGES:
        k = max(1, int(round(cov * n)))
        acc = cum_ok[k - 1] / k
        rows.append({"ranker": name, "signal": signal, "coverage": cov,
                     "n_accepted": k, "accuracy_on_accepted": round(float(acc), 4)})
    cov_df = pd.DataFrame(rows)

    rows = []
    for bud in BUDGETS:
        k = max(1, int(round((1 - bud) * n)))    # auto-accepted
        err_accepted = k - int(cum_ok[k - 1])
        captured = (total_err - err_accepted) / total_err if total_err else None
        post = (int(cum_ok[k - 1]) + (n - k)) / n     # reviewer fixes the queue
        rows.append({"ranker": name, "signal": signal, "review_budget": bud,
                     "errors_captured": round(float(captured), 4) if captured is not None else None,
                     "post_review_accuracy": round(float(post), 4),
                     "precision_of_queue": round(float((total_err - err_accepted) / (n - k)), 4)
                     if n - k else None})
    bud_df = pd.DataFrame(rows)
    return cov_df, bud_df


def main():
    t0 = time.time()
    notes, ann = load()
    train_notes, test_notes = split_notes(ann)
    cg = CandidateGenerator().fit(ann, train_notes)
    train_freq = Counter(int(c) for c in ann[ann.note_id.isin(train_notes)].concept_id)
    seen_forms = set(cg.form_to_concepts.keys())
    tn = sorted(test_notes); random.Random(SEED).shuffle(tn)
    val_notes = set(tn[:max(1, int(len(tn) * VAL_FRAC))])
    eval_notes = set(tn) - val_notes
    gold = build_test_mentions(notes, ann, test_notes, cg)
    ev = [m for m in gold if m.note_id in eval_notes]
    gold_of = {(m.note_id, m.start): m.gold_concept for m in ev}

    Xev, _, meta_ev = load_feats(FEAT_EVAL, "eval")
    keys, pools, seen = [], [], set()
    for (nid, st, c) in meta_ev:
        k = (nid, int(st))
        if k not in seen:
            seen.add(k); keys.append(k); pools.append([])
        pools[-1].append(int(c))

    from sklearn.experimental import enable_hist_gradient_boosting  # noqa
    from sklearn.ensemble import HistGradientBoostingClassifier
    Xtr, ytr, _ = load_feats(FEAT_TRAIN, "train")
    Xlono = np.load(LONO_TRAIN, allow_pickle=True)["X"]
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                         max_depth=6, l2_regularization=1.0,
                                         random_state=SEED)
    clf.fit(Xlono, ytr)
    del Xtr, Xlono
    proba = clf.predict_proba(Xev)[:, 1]
    del Xev
    lono = {}
    off = 0
    for k, p in zip(keys, pools):
        i, mx, mg = top2(proba[off:off + len(p)])
        lono[k] = (int(p[i]), mx, mg)
        off += len(p)
    print(f"  LONO confidences ready ({time.time()-t0:.0f}s)", flush=True)

    gold_D = pickle.load(open(GOLD_POOLS, "rb"))
    gold_tprob = pickle.load(open(GOLD_TPROB, "rb"))
    for m in gold:
        k = (m.note_id, m.start)
        if k in gold_D["pools"]:
            m.candidates = gold_D["pools"][k]
    w_seen, w_novel = tune_gate(gold_D, gold_tprob,
                                [m for m in gold if m.note_id in val_notes],
                                train_freq, seen_forms)
    gated = {}
    for m in ev:
        k = (m.note_id, m.start)
        wd, wt, wl = w_seen if norm(m.span) in seen_forms else w_novel
        tp = gold_tprob.get(k, {})
        dcos, lexsc, coarse = gold_D["dcos"][k], gold_D["lexsc"][k], gold_D["coarse"][k]
        pool = list(gold_D["pools"][k])
        sc = np.array([wd * dcos.get(c, 0.0) + wt * tp.get(ct4, 0.0)
                       + wl * lexsc.get(c, 0.0)
                       for c, ct4 in ((c, coarse[c] if coarse[c] in TYPE4 else "other")
                                      for c in pool)])
        i, mx, mg = top2(sc)
        gated[k] = (int(pool[i]), mx, mg)
    del gold_D
    print(f"  gated confidences ready ({time.time()-t0:.0f}s)", flush=True)

    order_keys = [(m.note_id, m.start) for m in ev]
    scope_mask = {s: np.array([s in scopes_of(m, train_freq) for m in ev])
                  for s in SCOPES}

    cov_all, bud_all, comp_rows = [], [], []
    for name, pk in (("gated", gated), ("supervised, LONO", lono)):
        ok = np.array([pk[k][0] == gold_of[k] for k in order_keys])
        for signal in ("maxprob", "margin"):
            conf = np.array([pk[k][1 if signal == "maxprob" else 2]
                             for k in order_keys])
            c, b = curves(ok, conf, name, signal)
            cov_all.append(c); bud_all.append(b)
        # oracle ordering: every error first
        c, b = curves(ok, ok.astype(float), name, "oracle (ceiling)")
        cov_all.append(c); bud_all.append(b)

        # composition of a 20% review queue under the better signal (margin)
        conf = np.array([pk[k][2] for k in order_keys])
        q = np.argsort(conf, kind="stable")[:int(round(0.20 * len(ok)))]
        qm = np.zeros(len(ok), dtype=bool); qm[q] = True
        for s in SCOPES:
            share_corpus = scope_mask[s].mean()
            share_queue = scope_mask[s][qm].mean()
            comp_rows.append({
                "ranker": name, "scope": s,
                "share_of_corpus": round(float(share_corpus), 4),
                "share_of_20%_queue": round(float(share_queue), 4),
                "enrichment": round(float(share_queue / share_corpus), 2)
                if share_corpus else None,
                "errors_of_this_scope_captured": round(
                    float((~ok & scope_mask[s] & qm).sum()
                          / max((~ok & scope_mask[s]).sum(), 1)), 4)})

    cov = pd.concat(cov_all, ignore_index=True)
    bud = pd.concat(bud_all, ignore_index=True)
    comp = pd.DataFrame(comp_rows)

    out = HERE / "stage2_out"; out.mkdir(exist_ok=True)
    cov.to_csv(out / "stage2_abstention_coverage.csv", index=False)
    bud.to_csv(out / "stage2_abstention_budget.csv", index=False)
    comp.to_csv(out / "stage2_abstention_queue.csv", index=False)
    lines = [
        "# Selective prediction — accuracy-coverage and human-review triage", "",
        f"eval = canonical split {len(ev):,} mentions, gold spans, weighted RRF@60 "
        "pool. `oracle (ceiling)` orders every error ahead of every correct "
        "prediction and bounds what any confidence signal could achieve.", "",
        "## 1. Accuracy on the auto-accepted portion", "",
        tcg.df_to_markdown(cov), "",
        "## 2. Review budget: what does reviewing X% buy?", "",
        "`errors_captured` = share of all errors that land in the queue. "
        "`post_review_accuracy` = corpus accuracy if a reviewer corrects every "
        "queued mention. `precision_of_queue` = share of the queue that really "
        "was wrong, i.e. how much reviewer time is not wasted.", "",
        tcg.df_to_markdown(bud), "",
        "## 3. Who ends up in a 20% review queue? (margin signal)", "",
        "`enrichment` > 1 means the stratum is over-represented in the queue "
        "relative to the corpus — the property you want for a long-tailed task.", "",
        tcg.df_to_markdown(comp), "",
    ]
    (out / "STAGE2_ABSTENTION.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + bud.to_string(index=False))
    print("\n" + comp.to_string(index=False))
    print(f"\nwrote {out/'STAGE2_ABSTENTION.md'} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
