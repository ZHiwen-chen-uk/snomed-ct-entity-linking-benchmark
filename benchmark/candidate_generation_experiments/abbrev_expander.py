"""Case-sensitive abbreviation / acronym expansion for candidate generation.

Motivation
----------
Lexical and char-n-gram retrieval both lowercase their input (`norm`), which
destroys the one signal that distinguishes a clinical acronym from an ordinary
word: case. "UA" (urinalysis), "MS" (multiple sclerosis / mitral stenosis /
morphine sulfate), "LMCA" (left main coronary artery) never share enough
character overlap with their SNOMED preferred terms to be retrieved, so their
gold concept is simply unreachable. This module adds a *protected*,
case-sensitive expansion layer that runs before the normal retrievers.

Two dictionary sources, both keyed on the EXACT-CASE token:
  1. data-mined  : training annotations whose span carries an acronym-shaped
                   token -> the gold concepts it linked to (train-only, no test
                   leakage). This is dataset-grounded and reproducible.
  2. curated seed: a small, transparent map of well-known clinical acronyms ->
                   expansion phrases, for acronyms absent from training.

`expand(span)` returns (direct_concepts, expansion_terms):
  * direct_concepts  : concept ids from the data-mined map (high precision,
                       case-sensitive dictionary hit).
  * expansion_terms  : surface strings (curated phrases + labels of the mined
                       concepts) to route through the existing lexical/char
                       retrievers so SNOMED synonyms of the expansion are found.

Only mentions that actually contain an acronym-shaped token are touched, so the
distractor cost on ordinary mentions is zero.
"""

import re
from collections import Counter, defaultdict

# a token that could be an acronym; case is preserved (unlike candidate_booster.tokens)
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")

# all-caps English/function words that are not clinical acronyms
_NOT_ACRONYM = {
    "AND", "OR", "NOT", "THE", "WITH", "FOR", "PER", "PRN", "BID", "TID",
    "QID", "PO", "IV", "IM",  # dosing routes/freqs: linked to qualifiers, skip
    "NO", "YES", "AM", "PM",
}


def is_acronym_token(tok):
    """Case-sensitive acronym shape test.

    True for UA, MS, LMCA, LAD, MI, CT, MRI, COPD, pRBC (>=2 uppercase letters,
    length 2-6). False for ordinary Capitalised words (one leading uppercase) and
    for common all-caps function words.
    """
    if not (2 <= len(tok) <= 6):
        return False
    if tok.upper() in _NOT_ACRONYM:
        return False
    letters = [c for c in tok if c.isalpha()]
    if not letters:
        return False
    upper = sum(1 for c in letters if c.isupper())
    # need at least two uppercase letters (rules out "Ua", "Ct", ordinary words)
    return upper >= 2


def acronym_tokens(span):
    """Ordered-unique acronym-shaped tokens in a span, exact case preserved."""
    seen, out = set(), []
    for tok in _TOKEN_RE.findall(str(span)):
        if is_acronym_token(tok) and tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


# ---------------------------------------------------------------------------
# curated seed: well-known clinical acronyms -> expansion phrases.
# Deliberately small and transparent. Ambiguous acronyms list every sense so the
# retriever can surface each branch's gold concept (the disambiguation itself is
# left to the downstream ranker / Group A type filter).
# ---------------------------------------------------------------------------
CURATED_SEED = {
    "UA": ["urinalysis", "urine analysis"],
    "MS": ["multiple sclerosis", "mitral stenosis", "morphine sulfate"],
    "MI": ["myocardial infarction"],
    "RA": ["rheumatoid arthritis", "right atrium", "room air"],
    "CT": ["computed tomography"],
    "MRI": ["magnetic resonance imaging"],
    "CXR": ["chest x-ray", "chest radiograph"],
    "LAD": ["left anterior descending coronary artery"],
    "LMCA": ["left main coronary artery"],
    "LM": ["left main coronary artery"],
    "RCA": ["right coronary artery"],
    "COPD": ["chronic obstructive pulmonary disease"],
    "CHF": ["congestive heart failure"],
    "CAD": ["coronary artery disease"],
    "CKD": ["chronic kidney disease"],
    "AKI": ["acute kidney injury"],
    "DVT": ["deep vein thrombosis"],
    "PE": ["pulmonary embolism"],
    "UTI": ["urinary tract infection"],
    "AF": ["atrial fibrillation"],
    "AFib": ["atrial fibrillation"],
    "HTN": ["hypertension"],
    "DM": ["diabetes mellitus"],
    "GI": ["gastrointestinal"],
    "CABG": ["coronary artery bypass graft"],
    "ESRD": ["end stage renal disease"],
    "SOB": ["shortness of breath"],
    "pRBC": ["packed red blood cells"],
    "PRBC": ["packed red blood cells"],
}


class AbbrevExpander:
    """Case-sensitive acronym -> (concepts, expansion terms) expander."""

    def __init__(self, mined=None, curated=None, max_direct=5):
        self.mined = mined or {}
        self.curated = curated if curated is not None else CURATED_SEED
        self.max_direct = int(max_direct)

    # ---- construction -----------------------------------------------------
    @classmethod
    def from_training(cls, ann, train_notes, max_direct=5, min_count=1):
        """Mine (exact-case acronym token -> gold concept counts) from training
        annotations only. `min_count` drops one-off noise."""
        mined = defaultdict(Counter)
        sub = ann[ann.note_id.isin(train_notes)]
        for row in sub.itertuples():
            cid = int(row.concept_id)
            for tok in acronym_tokens(row.span):
                mined[tok][cid] += 1
        mined = {
            tok: counts
            for tok, counts in mined.items()
            if sum(counts.values()) >= min_count
        }
        return cls(mined=mined, max_direct=max_direct)

    # ---- expansion --------------------------------------------------------
    def has_acronym(self, span):
        return bool(acronym_tokens(span))

    def expand(self, span, labeler=None):
        """Return (direct_concepts, expansion_terms) for a mention span.

        `labeler(concept) -> str` (e.g. booster.label) turns mined concepts into
        expansion strings so their SNOMED synonyms are also retrievable.
        """
        direct, terms = [], []
        for tok in acronym_tokens(span):
            counts = self.mined.get(tok)
            if counts:
                for cid, _ in counts.most_common(self.max_direct):
                    direct.append(int(cid))
                    if labeler is not None:
                        lab = labeler(cid)
                        if lab and not str(lab).startswith("concept:"):
                            terms.append(str(lab))
            for phrase in self.curated.get(tok, ()):
                terms.append(phrase)
        return _dedupe_int(direct), _dedupe_text(terms)


def _dedupe_int(items):
    seen, out = set(), []
    for x in items:
        x = int(x)
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _dedupe_text(items):
    seen, out = set(), []
    for x in items:
        key = re.sub(r"\s+", " ", str(x).strip().lower())
        if key and key not in seen:
            seen.add(key)
            out.append(str(x).strip())
    return out
