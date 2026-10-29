"""Metrics and stratified analysis for the four groups.

Headline metrics (proposal sec. 'Evaluation and Analysis'):
  * entity-level accuracy  : top-1 concept correct
  * recall@k               : gold concept present in candidate set (a ceiling
                              on every group; isolates candidate-generation)
  * macro char-IoU         : official challenge metric, computed with spans
                              fixed to gold so it reflects *linking* quality

Stratifications:
  * concept frequency  : head / torso / tail (training frequency terciles)
  * ontology depth     : shallow / deep
  * ambiguity type     : cross-branch (candidates span >1 semantic type)
                         vs. sibling (candidates share one type) vs. unambiguous
"""
from collections import Counter
from typing import Dict, List

import pandas as pd

from metric import macro_iou
from pipeline import Strategies
from ontology import Ontology


def recall_at_k(mentions, ks=(1, 3, 5, 10, 30)) -> Dict[int, float]:
    out = {}
    for k in ks:
        hit = sum(1 for m in mentions if m.gold_concept in m.candidates[:k])
        out[k] = hit / len(mentions) if mentions else 0.0
    return out


def char_iou_for_predictions(mentions, preds: List[int]) -> float:
    """Spans fixed to gold; only the predicted concept_id varies per group."""
    pred_df = pd.DataFrame([
        {"note_id": m.note_id, "start": m.start, "end": m.end,
         "concept_id": p if p is not None else -1}
        for m, p in zip(mentions, preds)
    ])
    gold_df = pd.DataFrame([
        {"note_id": m.note_id, "start": m.start, "end": m.end,
         "concept_id": m.gold_concept}
        for m in mentions
    ])
    return macro_iou(pred_df, gold_df)


def ambiguity_type(mention, ontology: Ontology) -> str:
    types = {ontology.semantic_type(c) for c in mention.candidates}
    if len(mention.candidates) <= 1:
        return "unambiguous"
    return "cross-branch" if len(types) > 1 else "sibling"


def freq_stratum(concept, train_freq: Counter) -> str:
    f = train_freq.get(concept, 0)
    if f == 0:
        return "tail"
    if f >= 20:
        return "head"
    return "torso" if f >= 3 else "tail"


def depth_stratum(concept, ontology: Ontology) -> str:
    return "deep" if ontology.depth(concept) >= 4 else "shallow"


def evaluate_all(mentions, strat: Strategies, ontology: Ontology,
                 train_freq: Counter, groups: Dict[str, str]):
    """Run every group over every mention; return (summary_df, strata dict)."""
    # precompute per-mention strata
    amb = [ambiguity_type(m, ontology) for m in mentions]
    frq = [freq_stratum(m.gold_concept, train_freq) for m in mentions]
    dep = [depth_stratum(m.gold_concept, ontology) for m in mentions]

    rows = []
    by_amb, by_freq, by_depth = {}, {}, {}
    for label, gkey in groups.items():
        preds = [strat.predict(m, gkey) for m in mentions]
        correct = [int(p == m.gold_concept) for p, m in zip(preds, mentions)]

        acc = sum(correct) / len(mentions)
        iou = char_iou_for_predictions(mentions, preds)
        rows.append({"group": label, "accuracy": acc, "char_iou": iou})

        def stratified(keys):
            agg = {}
            for k, c in zip(keys, correct):
                agg.setdefault(k, []).append(c)
            return {k: sum(v) / len(v) for k, v in agg.items()}

        by_amb[label] = stratified(amb)
        by_freq[label] = stratified(frq)
        by_depth[label] = stratified(dep)

    summary = pd.DataFrame(rows)
    strata = {"ambiguity": by_amb, "frequency": by_freq, "depth": by_depth}
    return summary, strata
