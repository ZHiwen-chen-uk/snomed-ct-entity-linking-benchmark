"""Build datasets for the Group A transformer semantic-type classifier.

Outputs CSV files with the same label definition as the existing sklearn Group A:

    input = section + marked mention + surrounding context
    label = ontology.coarse_type(gold_concept)

The inner train/validation split is note-level to avoid span leakage.
Validation rows also include train-only candidate sets so `evaluate_a_gate.py`
can measure false exclusions for filtering policies.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set

import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = THIS_DIR.parent
PROJECT_ROOT = BENCHMARK_DIR.parent
sys.path.insert(0, str(BENCHMARK_DIR))

from data import (  # noqa: E402
    CONTEXT_WINDOW,
    CandidateGenerator,
    load,
    section_at,
    section_headers,
    semtype_feature_text,
    split_notes,
)
from ontology import SnomedOntology  # noqa: E402


SEED = 20260720


def mark_context(context: str, span: str) -> str:
    if not span:
        return context
    return context.replace(span, f"[MENTION] {span} [/MENTION]", 1)


def transformer_text(section: str, span: str, context: str) -> str:
    marked = mark_context(context, span)
    return (
        f"Section: {section or 'UNKNOWN'}\n"
        f"Mention: {span}\n"
        f"Context: {marked}"
    )


def candidate_payload(cg: CandidateGenerator, ont: SnomedOntology, span: str) -> Dict[str, str]:
    candidates = cg.generate(span)
    return {
        "candidate_ids": json.dumps([int(c) for c in candidates]),
        "candidate_coarse_types": json.dumps([ont.coarse_type(int(c)) for c in candidates]),
    }


def build_rows(
    notes: pd.DataFrame,
    ann: pd.DataFrame,
    note_ids: Set[str],
    ont: SnomedOntology,
    split_name: str,
    candidate_generator: CandidateGenerator | None = None,
) -> List[Dict[str, object]]:
    text_by_id = dict(zip(notes.note_id, notes.text))
    hdr_cache = {}
    rows: List[Dict[str, object]] = []
    sub = ann[ann.note_id.isin(note_ids)].copy()
    for row in sub.itertuples():
        text = str(text_by_id.get(row.note_id, ""))
        if row.note_id not in hdr_cache:
            hdr_cache[row.note_id] = section_headers(text)
        lo = max(0, int(row.start) - CONTEXT_WINDOW)
        hi = min(len(text), int(row.end) + CONTEXT_WINDOW)
        section = section_at(hdr_cache[row.note_id], int(row.start))
        span = str(row.span)
        context = text[lo:hi]
        gold = int(row.concept_id)
        item = {
            "split": split_name,
            "note_id": row.note_id,
            "start": int(row.start),
            "end": int(row.end),
            "span": span,
            "section": section,
            "context": context,
            "feature_text": semtype_feature_text(section, span, context),
            "transformer_text": transformer_text(section, span, context),
            "gold_concept": gold,
            "label": ont.coarse_type(gold),
        }
        if candidate_generator is not None:
            item.update(candidate_payload(candidate_generator, ont, span))
        rows.append(item)
    return rows


def split_inner_train(train_notes: Iterable[str], val_frac: float, seed: int):
    note_list = sorted(train_notes)
    random.Random(seed).shuffle(note_list)
    n_val = max(1, int(len(note_list) * val_frac))
    val_ids = set(note_list[:n_val])
    inner_train_ids = set(note_list[n_val:])
    return inner_train_ids, val_ids


def write_label_map(out_dir: Path, labels: Sequence[str]):
    labels = sorted(set(labels))
    label2id = {label: i for i, label in enumerate(labels)}
    (out_dir / "label_map.json").write_text(
        json.dumps({"labels": labels, "label2id": label2id, "id2label": {i: l for l, i in label2id.items()}}, indent=2),
        encoding="utf-8",
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(THIS_DIR / "data"))
    p.add_argument("--snomed-dir", default=str(PROJECT_ROOT / "SnomedCT_InternationalRF2_PRODUCTION_20260601T120000Z"))
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    notes, ann = load()
    official_train_notes, official_test_notes = split_notes(ann)
    inner_train_notes, val_notes = split_inner_train(official_train_notes, args.val_frac, args.seed)
    ont = SnomedOntology(args.snomed_dir)

    # Validation candidates must not see validation annotations.
    cg_inner = CandidateGenerator().fit(ann, inner_train_notes)
    # Test candidates match the main benchmark setting: all official train notes.
    cg_official = CandidateGenerator().fit(ann, official_train_notes)

    train_rows = build_rows(notes, ann, inner_train_notes, ont, "train")
    val_rows = build_rows(notes, ann, val_notes, ont, "val", candidate_generator=cg_inner)
    test_rows = build_rows(notes, ann, official_test_notes, ont, "test", candidate_generator=cg_official)

    pd.DataFrame(train_rows).to_csv(out_dir / "semtype_train.csv", index=False)
    pd.DataFrame(val_rows).to_csv(out_dir / "semtype_val.csv", index=False)
    pd.DataFrame(test_rows).to_csv(out_dir / "semtype_test.csv", index=False)
    write_label_map(out_dir, [r["label"] for r in train_rows + val_rows])

    manifest = {
        "seed": args.seed,
        "val_frac": args.val_frac,
        "n_train_notes": len(inner_train_notes),
        "n_val_notes": len(val_notes),
        "n_test_notes": len(official_test_notes),
        "n_train_examples": len(train_rows),
        "n_val_examples": len(val_rows),
        "n_test_examples": len(test_rows),
        "train_label_counts": Counter(r["label"] for r in train_rows),
        "val_label_counts": Counter(r["label"] for r in val_rows),
        "test_label_counts": Counter(r["label"] for r in test_rows),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"wrote {out_dir}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
