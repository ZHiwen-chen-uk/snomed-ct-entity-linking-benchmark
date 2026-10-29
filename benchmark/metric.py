"""Official-style scoring metric for the SNOMED CT Entity Linking Challenge.

Macro-averaged character-level IoU (intersection-over-union), reproduced from
the challenge's reference implementation:

    For every concept_id appearing in the gold OR predicted annotations:
        - lay each annotation's concept_id onto a [n_notes x n_chars] matrix
          (overlapping spans: the later write wins, matching the official code)
        - intersection = chars where gold == c AND pred == c   (over ALL notes)
        - union        = chars where gold == c OR  pred == c
        - IoU(c)       = intersection / union
    score = mean over all concept_ids   <-- macro: every concept weighted equally

A concept with a non-empty gold/pred extent but zero overlap contributes 0,
which is why precise spans on rare concepts matter so much.
"""
import numpy as np
import scipy.sparse as sp


def iou_per_class(pred, gold):
    """pred / gold: DataFrames with columns note_id, start, end, concept_id.
    Returns dict {concept_id: iou}."""
    docs = np.unique(np.concatenate([pred.note_id.values, gold.note_id.values]))
    doc_index = {d: i for i, d in enumerate(docs)}
    cats = np.unique(np.concatenate([pred.concept_id.values, gold.concept_id.values]))
    max_end = int(np.max(np.concatenate([pred.end.values, gold.end.values])))

    def char_matrix(df):
        mtx = sp.lil_matrix((len(docs), max_end), dtype=np.int64)
        for row in df.itertuples():
            mtx[doc_index[row.note_id], int(row.start):int(row.end)] = int(row.concept_id)
        return mtx.tocsr()

    gt_mtx = char_matrix(gold)
    pred_mtx = char_matrix(pred)

    ious = {}
    for cat in cats:
        gt_cat = gt_mtx == cat
        pred_cat = pred_mtx == cat
        intersection = gt_cat.multiply(pred_cat)
        union = gt_cat + pred_cat
        u = union.sum()
        ious[int(cat)] = (intersection.sum() / u) if u else 0.0
    return ious


def macro_iou(pred, gold):
    """The single headline number: macro-averaged character IoU."""
    ious = iou_per_class(pred, gold)
    return float(np.mean(list(ious.values()))) if ious else 0.0
