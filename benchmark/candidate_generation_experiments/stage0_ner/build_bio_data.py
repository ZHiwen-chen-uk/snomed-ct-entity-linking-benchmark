"""Stage-0 span detection: build per-note NER data (model-agnostic, local, fast).

Emits ONE record per note (not per window) so the tokenizer/windowing/section
injection can be decided later in the trainer (ner_common.py). Each record keeps
the raw text, the gold entity char-spans, and the section-header offsets.

Split follows the rest of the pipeline (data.split_notes): notes with any 'test'
annotation -> test set. Detection ground truth on the test notes is the exact
'test' annotation set (the 23,131 mentions the downstream linker is scored on),
so Stage-0 recall is directly comparable to Stage-1/2. Training BIO comes from
the 'train'(+accepted) annotations in the 204 train notes.

Frequency strata (head >=20 / torso 3-19 / tail <3) are taken from evaluate.py
so head/torso/tail mean the same thing at every stage. Strata are attached to
GOLD spans only (used for detection *recall* by stratum -- the crux question:
does the NER miss tail spans more than head spans?).

    python build_bio_data.py        # -> ner_train.jsonl, ner_test.jsonl, BIO_STATS.md
"""
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))          # benchmark/  (data.py, evaluate.py)

from data import load, split_notes, section_headers   # noqa: E402
from evaluate import freq_stratum                      # noqa: E402

TRAIN_ANN_TYPES = {"train", "proposed_ACCEPTED"}       # supervision on train notes
TEST_ANN_TYPES = {"test"}                              # detection GT on test notes


def note_record(note_id, text, rows, train_freq):
    """One JSONL record: raw text + gold entity char-spans + section offsets."""
    ents = []
    for r in rows:
        s, e = int(r.start), int(r.end)
        if e <= s or e > len(text):
            continue                                   # guard against bad offsets
        ents.append({
            "start": s, "end": e,
            "span": text[s:e],
            "concept_id": int(r.concept_id),
            "stratum": freq_stratum(int(r.concept_id), train_freq),
        })
    ents.sort(key=lambda d: (d["start"], d["end"]))
    return {
        "note_id": note_id,
        "text": text,
        "sections": section_headers(text),             # [[offset, name], ...]
        "entities": ents,
    }


def dump(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    notes, ann = load()
    train_notes, test_notes = split_notes(ann)
    text_by_id = dict(zip(notes.note_id, notes.text))
    train_freq = Counter(int(c) for c in ann[ann.note_id.isin(train_notes)].concept_id)

    def build(note_ids, ann_types):
        recs = []
        sub = ann[ann.note_id.isin(note_ids) & ann.annotation_type.isin(ann_types)]
        for nid, grp in sub.groupby("note_id"):
            text = str(text_by_id.get(nid, ""))
            if not text:
                continue
            recs.append(note_record(nid, text, list(grp.itertuples()), train_freq))
        return recs

    train_recs = build(train_notes, TRAIN_ANN_TYPES)
    test_recs = build(test_notes, TEST_ANN_TYPES)
    dump(train_recs, HERE / "ner_train.jsonl")
    dump(test_recs, HERE / "ner_test.jsonl")

    # ---- stats -------------------------------------------------------------
    def strat_counts(recs):
        c = Counter(e["stratum"] for r in recs for e in r["entities"])
        return c

    tr_c, te_c = strat_counts(train_recs), strat_counts(test_recs)
    n_train_ent = sum(tr_c.values())
    n_test_ent = sum(te_c.values())
    sec_cov = sum(1 for r in test_recs for e in r["entities"]) or 1
    lens = sorted(len(r["text"]) for r in test_recs)

    lines = [
        "# Stage-0 NER data — BIO source statistics", "",
        f"- train notes: {len(train_recs)}  ·  gold spans: {n_train_ent}",
        f"- test notes:  {len(test_recs)}  ·  gold spans (detection GT): {n_test_ent}",
        f"- test note length chars: min {lens[0]} / median {lens[len(lens)//2]} / max {lens[-1]}",
        "",
        "## Gold-span frequency strata (test = detection recall denominators)", "",
        "| stratum | train spans | test spans | test % |",
        "|---|---|---|---|",
    ]
    for s in ("head", "torso", "tail"):
        lines.append(f"| {s} | {tr_c.get(s,0)} | {te_c.get(s,0)} | "
                     f"{100*te_c.get(s,0)/n_test_ent:.1f}% |")
    lines += [
        "", "## Section headers",
        f"- distinct section headers (test notes): "
        f"{len({name for r in test_recs for _, name in r['sections']})}",
        f"- test notes with >=1 detected header: "
        f"{sum(1 for r in test_recs if r['sections'])}/{len(test_recs)}",
        "",
        "Note: strata attach to GOLD spans; detection precision (false positives) "
        "cannot be stratified by gold frequency, so Stage-0 reports RECALL by "
        "stratum (does the detector miss tail spans more?) and overall P/F1/IoU.",
    ]
    (HERE / "BIO_STATS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote ner_train.jsonl ({len(train_recs)}) + ner_test.jsonl "
          f"({len(test_recs)}) -> {HERE}")


if __name__ == "__main__":
    main()
