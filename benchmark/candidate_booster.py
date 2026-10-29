"""SNOMED-backed candidate generation booster.

The original `CandidateGenerator` is deliberately train-only. That makes group
comparisons clean, but it under-recovers long-tail concepts whose surface forms
never appear in training annotations. This module adds an optional layer before
A/B/C:

* `synonym`: exact lookup over active SNOMED descriptions.
* `boosted`: synonym + lightweight token/fuzzy lexical retrieval.
* `grf`: boosted + local-LLM query rewrites before retrieval.

Downstream code still sees one shared `Mention.candidates` list.
"""

import json
import math
import pickle
import re
import urllib.request as urlreq
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from pipeline import OLLAMA_MODEL, OLLAMA_URL


HERE = Path(__file__).resolve().parent
SYN_CACHE = HERE / ".snomed_synindex.pkl"
TOKEN_CACHE = HERE / ".snomed_tokenindex.pkl"
LABEL_CACHE = HERE / ".snomed_concept_labels.pkl"
GRF_CACHE = HERE / ".grf_cache.json"

TOKEN_RE = re.compile(r"[a-z0-9]+")
TAG_RE = re.compile(r"\s*\([^()]+\)\s*$")
STOPWORDS = {
    "and", "are", "after", "before", "chronic", "clinical", "current",
    "finding", "for", "from", "history", "left", "right", "lower", "upper",
    "mild", "moderate", "normal", "not", "of", "on", "or", "other",
    "procedure", "severe", "status", "structure", "the", "to", "with",
    "without",
}

GRF_SYSTEM = (
    "You are a clinical terminologist. Given a clinical mention and its context, "
    "output up to 5 alternative surface forms that SNOMED CT might use for the "
    "same concept in context. Expand abbreviations and use standard clinical "
    "terms. Terms only, no explanations."
)


def norm(text):
    return re.sub(r"\s+", " ", str(text).strip().lower())


def tokens(text):
    out = []
    for tok in TOKEN_RE.findall(norm(text)):
        if len(tok) < 3 or tok in STOPWORDS:
            continue
        out.append(tok)
    return out


def ordered_unique(items, limit=None):
    seen = set()
    out = []
    for item in items:
        item = int(item)
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
        if limit is not None and len(out) >= limit:
            break
    return out


def ordered_unique_text(items, limit=None):
    seen = set()
    out = []
    for item in items:
        text = str(item).strip()
        key = norm(text)
        if not text or key in seen:
            continue
        seen.add(key)
        out.append(text)
        if limit is not None and len(out) >= limit:
            break
    return out


def load_json(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _terminology_dir(release_dir):
    release_dir = Path(release_dir)
    hits = list(release_dir.glob("**/Snapshot/Terminology/sct2_Concept_Snapshot_*.txt"))
    if not hits:
        raise FileNotFoundError(f"No RF2 Snapshot/Terminology under {release_dir}")
    return hits[0].parent


def build_synindex(release_dir, cache_file=SYN_CACHE):
    """Build form -> concept ids from all active SNOMED descriptions."""
    cache_file = Path(cache_file)
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    term_dir = _terminology_dir(release_dir)
    concept_f = next(term_dir.glob("sct2_Concept_Snapshot_*.txt"))
    desc_f = next(term_dir.glob("sct2_Description_Snapshot-en_*.txt"))

    active = set()
    with open(concept_f, encoding="utf-8") as f:
        next(f)
        for line in f:
            cols = line.split("\t")
            if cols[2] == "1":
                active.add(cols[0])

    form_to_concepts = {}
    with open(desc_f, encoding="utf-8") as f:
        next(f)
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9 or cols[2] != "1" or cols[4] not in active:
                continue
            term = TAG_RE.sub("", cols[7]).strip()
            if not term:
                continue
            form_to_concepts.setdefault(norm(term), set()).add(int(cols[4]))

    compact = {form: tuple(sorted(concepts)) for form, concepts in form_to_concepts.items()}
    with open(cache_file, "wb") as f:
        pickle.dump(compact, f, protocol=pickle.HIGHEST_PROTOCOL)
    return compact


class SnomedCandidateBooster:
    """Optional SNOMED candidate-generation layer for long-tail recall."""

    def __init__(
        self,
        base_generator,
        ontology,
        release_dir,
        mode="boosted",
        max_candidates=45,
        expand_k=25,
        grf_mode="cache",
        max_postings=12000,
    ):
        self.base = base_generator
        self.ontology = ontology
        self.release_dir = release_dir
        self.mode = mode
        self.max_candidates = int(max_candidates)
        self.expand_k = int(expand_k)
        self.grf_mode = grf_mode
        self.max_postings = max_postings
        self.global_freq = base_generator.global_freq

        self.syn_index = build_synindex(release_dir)
        self.concept_labels = self._load_or_build_concept_labels()
        self.token_index = self._load_or_build_token_index()
        self.grf_cache = load_json(GRF_CACHE)
        self.grf_calls = 0
        self.grf_errors = 0

    def concept_term(self, concept):
        if int(concept) in self.base.concept_forms:
            return self.base.concept_term(int(concept))
        return self.label(concept)

    def prior(self, form, concept):
        return self.base.prior(form, concept)

    def generate(self, form):
        return self.base.generate(form)

    def label(self, concept):
        pref = self.ontology.preferred_term(concept)
        if pref and not str(pref).startswith("concept:"):
            return pref
        return self.concept_labels.get(int(concept), f"concept:{concept}")

    def boost_mentions(self, mentions):
        stats = Counter()
        for mention in mentions:
            before = list(mention.candidates)
            after = self.boost(mention)
            mention.candidates = after
            stats["mentions"] += 1
            stats["before_total"] += len(before)
            stats["after_total"] += len(after)
            stats["expanded"] += int(len(after) > len(before))
            stats["gold_before"] += int(mention.gold_concept in set(before))
            stats["gold_after"] += int(mention.gold_concept in set(after))
            stats["gold_new"] += int(
                mention.gold_concept not in set(before)
                and mention.gold_concept in set(after)
            )
        return stats

    def boost(self, mention):
        base = list(mention.candidates)
        exact_hits = [int(c) for c in self.syn_index.get(norm(mention.span), ())]

        if self.mode == "synonym":
            return ordered_unique(base + exact_hits, self.max_candidates)

        lexical_hits, _ = self.retrieve([mention.span], self.expand_k)
        expanded = base + exact_hits + lexical_hits

        if self.mode == "grf":
            terms = ordered_unique_text([mention.span] + self.grf_rewrite(mention), limit=6)
            grf_hits, _ = self.retrieve(terms, self.expand_k)
            expanded.extend(grf_hits)

        return ordered_unique(expanded, self.max_candidates)

    def retrieve(self, terms, limit):
        clean_terms = [norm(t) for t in terms if norm(t)]
        query_tokens = set()
        raw = Counter()

        for i, term in enumerate(clean_terms):
            term_weight = max(0.25, 1.0 - 0.15 * i)
            for cid in self.syn_index.get(term, ()):
                raw[int(cid)] += 8.0 * term_weight
            toks = set(tokens(term))
            query_tokens.update(toks)
            for tok in toks:
                posting = self.token_index.get(tok, ())
                if not posting:
                    continue
                idf_like = 1.0 / max(1.0, math.sqrt(len(posting)))
                tok_weight = term_weight * (1.0 + min(len(tok), 12) / 12.0) * idf_like
                for cid in posting:
                    raw[int(cid)] += tok_weight

        if not raw:
            return [], {}

        shortlist = [cid for cid, _ in raw.most_common(max(limit * 20, 200))]
        rescored = []
        for cid in shortlist:
            score = raw[cid] + 3.0 * self._text_similarity(clean_terms, query_tokens, cid)
            rescored.append((cid, score))
        rescored.sort(key=lambda kv: kv[1], reverse=True)

        top = rescored[:limit]
        max_score = max((score for _, score in top), default=1.0)
        scores = {cid: (score / max_score if max_score else 0.0) for cid, score in top}
        return [cid for cid, _ in top], scores

    def grf_rewrite(self, mention):
        key = f"{mention.note_id}|{mention.start}"
        if key in self.grf_cache and self.grf_mode != "ollama_refresh":
            vals = self.grf_cache.get(key) or []
            return [str(v) for v in vals if str(v).strip()]
        if self.grf_mode not in ("ollama", "ollama_refresh"):
            return []

        marked = mention.context.replace(mention.span, f"<<< {mention.span} >>>", 1)
        user = (
            f"Context (mention between <<< >>>):\n...{marked}...\n\n"
            f"Mention: \"{mention.span}\"\n\n"
            "Return SNOMED-style alternative terms for the same concept."
        )
        schema = {
            "type": "object",
            "properties": {"terms": {"type": "array", "items": {"type": "string"}}},
            "required": ["terms"],
        }
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [{"role": "system", "content": GRF_SYSTEM}, {"role": "user", "content": user}],
            "stream": False,
            "format": schema,
            "options": {"temperature": 0, "num_predict": 120},
        }
        try:
            req = urlreq.Request(
                OLLAMA_URL,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urlreq.urlopen(req, timeout=120) as r:
                resp = json.loads(r.read())
            terms = json.loads(resp["message"]["content"]).get("terms", [])
            self.grf_calls += 1
        except Exception:
            terms = []
            self.grf_errors += 1
        self.grf_cache[key] = terms
        GRF_CACHE.write_text(json.dumps(self.grf_cache, indent=2))
        return [str(t) for t in terms if str(t).strip()]

    def _text_similarity(self, query_norms, query_tokens, concept):
        label = norm(self.label(concept))
        label_tokens = set(tokens(label))
        if not label:
            return 0.0
        seq = max((SequenceMatcher(None, q, label).ratio() for q in query_norms), default=0.0)
        dice = 0.0
        if label_tokens and query_tokens:
            dice = 2 * len(label_tokens & query_tokens) / (len(label_tokens) + len(query_tokens))
        exact = 1.0 if label in query_norms else 0.0
        return 1.2 * exact + 0.65 * dice + 0.35 * seq

    def _load_or_build_concept_labels(self):
        if LABEL_CACHE.exists():
            with open(LABEL_CACHE, "rb") as f:
                return pickle.load(f)

        labels = {}
        for form, concepts in self.syn_index.items():
            clean = TAG_RE.sub("", str(form)).strip()
            if not clean:
                continue
            rank = (len(tokens(clean)) or 99, len(clean), clean)
            for cid in concepts:
                cid = int(cid)
                prev = labels.get(cid)
                if prev is None or rank < prev[0]:
                    labels[cid] = (rank, clean)

        compact = {cid: label for cid, (_, label) in labels.items()}
        with open(LABEL_CACHE, "wb") as f:
            pickle.dump(compact, f, protocol=pickle.HIGHEST_PROTOCOL)
        return compact

    def _load_or_build_token_index(self):
        if TOKEN_CACHE.exists():
            with open(TOKEN_CACHE, "rb") as f:
                return pickle.load(f)

        index = {}
        too_common = set()
        for form, concepts in self.syn_index.items():
            form_tokens = set(tokens(form))
            if not form_tokens:
                continue
            for tok in form_tokens:
                if tok in too_common:
                    continue
                bucket = index.setdefault(tok, set())
                bucket.update(concepts)
                if len(bucket) > self.max_postings:
                    too_common.add(tok)
                    del index[tok]

        compact = {tok: tuple(sorted(concepts)) for tok, concepts in index.items()}
        with open(TOKEN_CACHE, "wb") as f:
            pickle.dump(compact, f, protocol=pickle.HIGHEST_PROTOCOL)
        return compact
