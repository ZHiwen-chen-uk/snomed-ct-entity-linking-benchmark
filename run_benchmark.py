"""Rough end-to-end benchmark for the SNOMED CT Entity Linking Challenge.

Pipeline:
  1. Split the 272 annotated notes into train / test using `annotation_type`
     (notes with 'test' annotations -> test set; the rest -> train set).
  2. Build a baseline: a dictionary that maps each surface form seen in the
     training annotations to its most frequent SNOMED concept_id.
  3. Predict on the test notes by greedy longest-phrase matching.
  4. Score with the official macro character-IoU metric.

This is intentionally simple -- it is the "does the harness work + what does a
trivial baseline get" reference point, NOT a competitive system. Your LLM +
ontology rerank pipeline is meant to beat this.
"""
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metric import macro_iou, iou_per_class  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / \
    "snomed-ct-entity-linking-challenge-1.2.1" / \
    "snomed-ct-entity-linking-challenge-1.2.1"

TOKEN_RE = re.compile(r"\S+")
MAX_PHRASE_WORDS = 8


def load():
    notes = pd.read_csv(DATA / "train_notes.csv")
    ann = pd.read_csv(DATA / "train_annotations.csv")
    ann = ann.dropna(subset=["start", "end", "concept_id"])
    return notes, ann


def split_notes(ann):
    test_notes = set(ann.loc[ann.annotation_type == "test", "note_id"].unique())
    train_notes = set(ann.note_id.unique()) - test_notes
    return train_notes, test_notes


def build_dictionary(ann, train_notes):
    """surface form (lowercased) -> most frequent concept_id."""
    sub = ann[ann.note_id.isin(train_notes)]
    by_form = defaultdict(Counter)
    for row in sub.itertuples():
        form = str(row.span).strip().lower()
        if form:
            by_form[form][int(row.concept_id)] += 1
    return {form: c.most_common(1)[0][0] for form, c in by_form.items()}


def predict_note(note_id, text, phrase_dict, max_words):
    """Greedy, non-overlapping longest-match over whitespace tokens."""
    tokens = [(m.group(0).lower(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]
    preds = []
    i = 0
    while i < len(tokens):
        matched = False
        for n in range(min(max_words, len(tokens) - i), 0, -1):
            window = tokens[i:i + n]
            phrase = " ".join(w for w, _, _ in window)
            # strip trailing punctuation that breaks exact match
            phrase_clean = phrase.strip(".,;:()")
            for cand in (phrase, phrase_clean):
                if cand in phrase_dict:
                    preds.append({
                        "note_id": note_id,
                        "start": window[0][1],
                        "end": window[-1][2],
                        "concept_id": phrase_dict[cand],
                    })
                    i += n
                    matched = True
                    break
            if matched:
                break
        if not matched:
            i += 1
    return preds


def main():
    notes, ann = load()
    train_notes, test_notes = split_notes(ann)
    phrase_dict = build_dictionary(ann, train_notes)

    print(f"notes: {len(notes)} (train {len(train_notes)} / test {len(test_notes)})")
    print(f"dictionary entries (surface forms): {len(phrase_dict)}")

    text_by_id = dict(zip(notes.note_id, notes.text))
    preds = []
    for nid in test_notes:
        preds.extend(predict_note(nid, str(text_by_id[nid]), phrase_dict, MAX_PHRASE_WORDS))
    pred_df = pd.DataFrame(preds)

    gold = ann[ann.note_id.isin(test_notes)][["note_id", "start", "end", "concept_id"]].copy()

    print(f"gold annotations (test): {len(gold)} | predicted: {len(pred_df)}")

    score = macro_iou(pred_df, gold)
    print("\n=== Macro character IoU (headline) ===")
    print(f"  baseline = {score:.4f}")
    print("  (official challenge benchmark blog reported ~0.1794;")
    print("   winning solutions ~0.4+)")

    # quick head/tail breakdown by concept frequency in gold
    per_class = iou_per_class(pred_df, gold)
    freq = Counter(int(c) for c in gold.concept_id)
    ranked = sorted(per_class, key=lambda c: freq.get(c, 0), reverse=True)
    head = ranked[:len(ranked) // 3] or ranked
    tail = ranked[-len(ranked) // 3:] or ranked
    mean = lambda cs: sum(per_class[c] for c in cs) / len(cs)
    print("\n=== IoU by concept frequency (illustrative) ===")
    print(f"  head (frequent) concepts: {mean(head):.4f}")
    print(f"  tail (rare)     concepts: {mean(tail):.4f}")
    print("  -> the gap here is exactly what an ontology-aware reranker can close.")


if __name__ == "__main__":
    main()
