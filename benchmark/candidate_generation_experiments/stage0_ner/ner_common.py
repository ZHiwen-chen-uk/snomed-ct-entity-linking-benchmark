"""Shared Stage-0 NER harness: windowing, BIO alignment, SECTION injection, and
char-level detection metrics. Used by both the local smoke run and the Colab run
(run_ner.py) so the two variants differ in exactly ONE thing: whether each window
is prefixed with its governing section header.

Label scheme: O=0, B-ENT=1, I-ENT=2 (single entity class -- Stage 0 only decides
IS-a-mention; the concept is decided downstream by Stage 1/2).

Section-aware variant (the controlled A/B):
  vanilla       : [CLS] <window tokens> [SEP]
  section_aware : [CLS] <section-header tokens> [SEP] <window tokens> [SEP]
Only the section prefix changes; window tokens, labels and offsets are identical,
so any metric delta is attributable to the section signal (KIRI keyed its winning
dictionary on (section, mention); Davidson et al. 2025 recommend section context).
"""
import bisect
import json
from collections import Counter, defaultdict

O, B, I = 0, 1, 2
LABELS = ["O", "B-ENT", "I-ENT"]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def section_at(sections, pos):
    """Nearest section header at or before char offset `pos`. sections=[[off,name]]."""
    if not sections:
        return ""
    offs = [s[0] for s in sections]
    i = bisect.bisect_right(offs, pos) - 1
    return sections[i][1] if i >= 0 else ""


def _char_labels_for_tokens(offsets, entities):
    """BIO label per token from gold entity char-spans (subword granularity: every
    token overlapping an entity is labelled; first overlapping token = B, rest I)."""
    labels = [O] * len(offsets)
    starts = [o[0] for o in offsets]
    for e in entities:
        s, t = e["start"], e["end"]
        # first token whose [start,end) overlaps [s,t)
        lo = bisect.bisect_right(starts, s) - 1
        if lo < 0:
            lo = 0
        first = True
        for k in range(lo, len(offsets)):
            a, b = offsets[k]
            if a >= t:
                break
            if b <= s:
                continue
            labels[k] = B if first else I
            first = False
    return labels


def encode_note(tokenizer, rec, variant, max_len=384, stride=None, content_tokens=None):
    """Yield window dicts for one note: input_ids, attention_mask, labels, offsets
    (char spans in NOTE coordinates; (-1,-1) for special/section-prefix tokens)."""
    text, sections, ents = rec["text"], rec["sections"], rec["entities"]
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                    truncation=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    tok_labels = _char_labels_for_tokens(offs, ents)
    cls, sep = tokenizer.cls_token_id, tokenizer.sep_token_id
    pad = tokenizer.pad_token_id or 0

    # room for content tokens after specials + optional section prefix
    if content_tokens is None:
        content_tokens = max_len - 32
    if stride is None:
        stride = content_tokens - 48                 # ~48-token overlap between windows

    windows = []
    n = len(ids)
    a = 0
    while a < n:
        b = min(a + content_tokens, n)
        c_ids, c_off, c_lab = ids[a:b], offs[a:b], tok_labels[a:b]
        prefix_ids, prefix_off = [cls], [(-1, -1)]
        if variant == "section_aware":
            sec = section_at(sections, c_off[0][0])
            if sec:
                sec_ids = tokenizer(sec, add_special_tokens=False)["input_ids"][:24]
                prefix_ids = [cls] + sec_ids + [sep]
                prefix_off = [(-1, -1)] * len(prefix_ids)
        input_ids = prefix_ids + c_ids + [sep]
        offsets = prefix_off + c_off + [(-1, -1)]
        labels = [-100] * len(prefix_ids) + c_lab + [-100]
        input_ids = input_ids[:max_len]
        offsets = offsets[:max_len]
        labels = labels[:max_len]
        attn = [1] * len(input_ids)
        while len(input_ids) < max_len:
            input_ids.append(pad); attn.append(0); labels.append(-100); offsets.append((-1, -1))
        windows.append({"input_ids": input_ids, "attention_mask": attn,
                        "labels": labels, "offsets": offsets})
        if b >= n:
            break
        a += stride
    return windows


def build_dataset(tokenizer, records, variant, max_len=384):
    feats, win_meta = [], []
    for ri, rec in enumerate(records):
        for w in encode_note(tokenizer, rec, variant, max_len=max_len):
            feats.append({k: w[k] for k in ("input_ids", "attention_mask", "labels")})
            win_meta.append({"note_idx": ri, "offsets": w["offsets"]})
    return feats, win_meta


# ---- decode token predictions -> char spans, then note-level dedup ----------
def decode_window(pred_labels, offsets):
    """Contiguous B/I runs -> [(start,end)] char spans in note coordinates."""
    spans, cur = [], None
    for lab, (a, bb) in zip(pred_labels, offsets):
        if a < 0:                                    # special / section prefix
            if cur:
                spans.append(cur); cur = None
            continue
        if lab == B:
            if cur:
                spans.append(cur)
            cur = [a, bb]
        elif lab == I and cur:
            cur[1] = bb
        else:                                        # O (or I with no open span)
            if cur:
                spans.append(cur); cur = None
    if cur:
        spans.append(cur)
    return [(s, e) for s, e in spans if e > s]


def merge_spans(spans):
    """Union overlapping/adjacent predicted char intervals within a note."""
    if not spans:
        return []
    spans = sorted(spans)
    out = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def iou(a, b):
    s = max(a[0], b[0]); e = min(a[1], b[1])
    inter = max(0, e - s)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union else 0.0


def evaluate_detection(records, pred_spans_by_note, iou_thr=0.5):
    """records: gold (with strata). pred_spans_by_note: {note_id:[(s,e)]}.
    Returns overall exact/IoU P-R-F1, recall by stratum, mean best-IoU."""
    strata = ("head", "torso", "tail")
    gold_n = Counter(); det_iou = Counter(); det_exact = Counter()
    tot_gold = tot_pred = exact_tp = iou_tp = pred_matched = 0
    best_iou_sum = 0.0

    for rec in records:
        golds = [(e["start"], e["end"], e["stratum"]) for e in rec["entities"]]
        preds = merge_spans(pred_spans_by_note.get(rec["note_id"], []))
        gset = {(s, e) for s, e, _ in golds}
        tot_gold += len(golds); tot_pred += len(preds)

        pred_used = [False] * len(preds)
        for (gs, ge, st) in golds:
            gold_n[st] += 1
            g = (gs, ge)
            if g in gset and any(p == g for p in preds):
                exact_tp += 1; det_exact[st] += 1
            best, bj = 0.0, -1
            for j, p in enumerate(preds):
                v = iou(g, p)
                if v > best:
                    best, bj = v, j
            best_iou_sum += best
            if best >= iou_thr:
                iou_tp += 1; det_iou[st] += 1
                if bj >= 0:
                    pred_used[bj] = True
        pred_matched += sum(pred_used)

    def prf(tp, npred, ngold):
        p = tp / npred if npred else 0.0
        r = tp / ngold if ngold else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f

    ep, er, ef = prf(exact_tp, tot_pred, tot_gold)
    ip, ir, iff = prf(pred_matched, tot_pred, tot_gold)   # IoU precision uses matched preds
    return {
        "n_gold": tot_gold, "n_pred": tot_pred,
        "exact_P": ep, "exact_R": er, "exact_F1": ef,
        "iou_P": ip, "iou_R": ir, "iou_F1": iff,
        "mean_best_iou": best_iou_sum / tot_gold if tot_gold else 0.0,
        "recall_iou_by_stratum": {s: (det_iou[s] / gold_n[s] if gold_n[s] else 0.0)
                                  for s in strata},
        "recall_exact_by_stratum": {s: (det_exact[s] / gold_n[s] if gold_n[s] else 0.0)
                                    for s in strata},
        "gold_by_stratum": {s: gold_n[s] for s in strata},
    }


def format_report(name, m):
    L = [f"### {name}", "",
         f"- gold {m['n_gold']} · pred {m['n_pred']}",
         f"- exact  P {m['exact_P']:.3f} / R {m['exact_R']:.3f} / F1 {m['exact_F1']:.3f}",
         f"- IoU>=0.5 P {m['iou_P']:.3f} / R {m['iou_R']:.3f} / F1 {m['iou_F1']:.3f}"
         f"  · mean best-IoU {m['mean_best_iou']:.3f}", "",
         "| stratum | recall (IoU>=0.5) | recall (exact) | n_gold |",
         "|---|---|---|---|"]
    for s in ("head", "torso", "tail"):
        L.append(f"| {s} | {m['recall_iou_by_stratum'][s]:.3f} | "
                 f"{m['recall_exact_by_stratum'][s]:.3f} | {m['gold_by_stratum'][s]} |")
    return "\n".join(L) + "\n"
