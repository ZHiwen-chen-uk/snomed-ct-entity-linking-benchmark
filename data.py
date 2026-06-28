"""Data loading, train/test split, and candidate generation.

Shared front-end for every experimental group (Baseline / A / B / C). The key
design rule from the proposal: *candidate generation is identical across groups*
so that any performance difference is attributable to the ontology strategy,
not to a different initial candidate set (Davidson et al., 2025; Shen et al.,
2015).

Mention-level framing
---------------------
We evaluate the **candidate-ranking** sub-problem the proposal actually studies:
gold spans are given, and each group must pick the correct SNOMED concept from a
shared, high-recall candidate set. Span detection is intentionally held fixed to
the gold spans so the comparison isolates the *linking* step.
"""
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / \
    "snomed-ct-entity-linking-challenge-1.2.1" / \
    "snomed-ct-entity-linking-challenge-1.2.1"

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
MAX_CANDIDATES = 30          # cap per mention (recall vs. ranking cost trade-off)
CONTEXT_WINDOW = 120         # chars of context on each side of a mention


@dataclass
class Mention:
    """One gold annotation to be linked."""
    note_id: str
    start: int
    end: int
    span: str                # surface form as it appears in text
    gold_concept: int        # gold SNOMED concept_id
    context: str = ""        # surrounding clinical text (for the LLM ranker)
    candidates: List[int] = field(default_factory=list)  # shared candidate set


def _norm(form: str) -> str:
    return str(form).strip().lower()


def load():
    notes = pd.read_csv(DATA_DIR / "train_notes.csv")
    ann = pd.read_csv(DATA_DIR / "train_annotations.csv")
    ann = ann.dropna(subset=["start", "end", "concept_id"]).copy()
    ann["start"] = ann["start"].astype(int)
    ann["end"] = ann["end"].astype(int)
    ann["concept_id"] = ann["concept_id"].astype(int)
    return notes, ann


def split_notes(ann: pd.DataFrame):
    """Official-style split: notes with any 'test' annotation -> test set."""
    test_notes = set(ann.loc[ann.annotation_type == "test", "note_id"].unique())
    train_notes = set(ann.note_id.unique()) - test_notes
    return train_notes, test_notes


class CandidateGenerator:
    """High-recall candidate generation via surface-form + token lookup.

    Built ONLY from training annotations (no test leakage). Two recall sources:
      1. exact normalised surface-form match  -> concepts seen for that form
      2. token back-off                        -> concepts frequently linked to
         any training form sharing a token (handles unseen / morphological forms)

    NOTE: This is a lexical stand-in for the proposal's "string matching and
    synonym lookup against SNOMED CT". Swap in a real SNOMED synonym index for
    the dissertation; the interface (`generate(form) -> List[int]`) is stable.
    """

    def __init__(self, max_candidates: int = MAX_CANDIDATES):
        self.max_candidates = max_candidates
        self.form_to_concepts: Dict[str, Counter] = defaultdict(Counter)
        self.token_to_concepts: Dict[str, Counter] = defaultdict(Counter)
        self.global_freq: Counter = Counter()
        self.concept_forms: Dict[int, Counter] = defaultdict(Counter)  # concept -> spans

    def fit(self, ann: pd.DataFrame, train_notes: Set[str]):
        sub = ann[ann.note_id.isin(train_notes)]
        for row in sub.itertuples():
            form = _norm(row.span)
            cid = int(row.concept_id)
            if not form:
                continue
            self.form_to_concepts[form][cid] += 1
            self.global_freq[cid] += 1
            self.concept_forms[cid][str(row.span).strip()] += 1
            for tok in set(TOKEN_RE.findall(form)):
                self.token_to_concepts[tok][cid] += 1
        return self

    def concept_term(self, concept: int) -> str:
        """A human-readable label for a concept_id: its most frequent training
        span. Stand-in for the SNOMED preferred term (swap in the real FSN when
        SNOMED CT is available)."""
        c = self.concept_forms.get(concept)
        return c.most_common(1)[0][0] if c else f"concept:{concept}"

    def generate(self, form: str) -> List[int]:
        form = _norm(form)
        scores: Counter = Counter()
        # 1. exact form match (strongest signal)
        for cid, c in self.form_to_concepts.get(form, Counter()).items():
            scores[cid] += 5 * c
        # 2. token back-off (recall for unseen forms)
        for tok in set(TOKEN_RE.findall(form)):
            for cid, c in self.token_to_concepts.get(tok, Counter()).most_common(20):
                scores[cid] += c
        return [cid for cid, _ in scores.most_common(self.max_candidates)]

    def prior(self, form: str, concept: int) -> float:
        """P(concept | form) from training counts; used by the baseline ranker."""
        c = self.form_to_concepts.get(_norm(form))
        if c and sum(c.values()):
            return c.get(concept, 0) / sum(c.values())
        return 0.0


def build_test_mentions(notes, ann, test_notes, cg: CandidateGenerator) -> List[Mention]:
    """Gold mentions for the test notes, each with context + shared candidates."""
    text_by_id = dict(zip(notes.note_id, notes.text))
    sub = ann[(ann.note_id.isin(test_notes)) & (ann.annotation_type == "test")]
    mentions = []
    for row in sub.itertuples():
        text = str(text_by_id.get(row.note_id, ""))
        lo = max(0, row.start - CONTEXT_WINDOW)
        hi = min(len(text), row.end + CONTEXT_WINDOW)
        m = Mention(
            note_id=row.note_id, start=row.start, end=row.end,
            span=str(row.span), gold_concept=int(row.concept_id),
            context=text[lo:hi],
            candidates=cg.generate(str(row.span)),
        )
        mentions.append(m)
    return mentions
