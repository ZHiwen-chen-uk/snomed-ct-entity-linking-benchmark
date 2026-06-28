"""The four experimental groups from the proposal.

    Baseline : LLM picks best candidate, no ontology signal.
    Group A  : ontology PRE-FILTER (predicted semantic type removes
               incompatible candidates) -> then baseline ranks.
    Group B  : baseline ranks -> ontology POST-HOC RERANK (hierarchy
               distance penalty, weight beta).
    Group C  : A then B (filter, rank, rerank).

Every group shares the same candidate set (from data.CandidateGenerator) and the
same LLM ranker, so differences are attributable to the ontology strategy alone.
"""
from collections import Counter
from typing import Dict, List

from data import CandidateGenerator, Mention
from ontology import Ontology, MOCK_SEM_TYPES

# ---- tunable hyper-parameters (proposal: tuned on a validation split) --------
BETA = 0.04                  # Group B: hierarchy-distance penalty weight
FILTER_CONFIDENCE = 0.55     # Group A: keep-all fallback below this confidence
SEMTYPE_ACCURACY = 0.85      # MOCK predictor accuracy (real model replaces this)


# =============================================================================
# LLM ranker  (BASELINE engine, shared by all groups)
# =============================================================================
class LLMRanker:
    """Scores each candidate for a mention. Higher = better.

    Default `PriorRanker` is an offline stand-in: P(concept | surface form) from
    training counts, backed off to global concept frequency — a strong, fully
    reproducible proxy for an LLM that resolves unambiguous mentions by prior.

    To use a REAL LLM, subclass and override `score()` to prompt the model with
    `mention.context` + the candidate preferred terms and parse its choice.
    (Use the latest Claude model, e.g. claude-opus-4-8, with structured output.)
    """

    def score(self, mention: Mention, candidates: List[int]) -> Dict[int, float]:
        raise NotImplementedError


class PriorRanker(LLMRanker):
    def __init__(self, cg: CandidateGenerator):
        self.cg = cg

    def score(self, mention: Mention, candidates: List[int]) -> Dict[int, float]:
        gmax = max(self.cg.global_freq.values()) if self.cg.global_freq else 1
        out = {}
        for cid in candidates:
            prior = self.cg.prior(mention.span, cid)            # form-conditioned
            glob = self.cg.global_freq.get(cid, 0) / gmax       # back-off
            out[cid] = prior + 0.05 * glob
        return out


# ---- real LLM ranker (Stage-1 baseline engine) -----------------------------
import json as _json
from pathlib import Path as _Path

LLM_MODEL = "claude-opus-4-8"


def _load_cache(path):
    if path and _Path(path).exists():
        try:
            return _json.loads(_Path(path).read_text())
        except Exception:
            return {}
    return {}


def _save_cache(path, cache):
    if path:
        _Path(path).write_text(_json.dumps(cache))

SELECT_TOOL = {
    "name": "select_concept",
    "description": "Select the single SNOMED CT candidate that best matches the "
                   "clinical mention, given its surrounding context.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "concept_id": {"type": "integer",
                           "description": "The chosen candidate's concept_id."},
        },
        "required": ["concept_id"],
        "additionalProperties": False,
    },
}

LLM_SYSTEM = (
    "You are a clinical terminologist performing SNOMED CT entity linking. "
    "You are shown a mention from a discharge note, its surrounding context, and "
    "a fixed list of candidate SNOMED CT concepts (each an id and a representative "
    "term). Choose the single candidate that best captures the mention's meaning "
    "in context. You must pick exactly one candidate from the list via the "
    "select_concept tool — do not invent concept ids."
)


class ClaudeRanker(LLMRanker):
    """Stage-1 LLM baseline using the real Claude API (claude-opus-4-8).

    Presents mention + context + candidate list; the model picks one candidate
    via a strict tool call (the enum-free integer is validated against the
    candidate set after the call). Scores are cached per (mention, candidate-set)
    so Groups Baseline/B (same set) and A/C (filtered set) reuse calls.

    The chosen candidate gets score 1.0; the rest fall back to the lexical prior
    so reranking still has a sensible tie-break order.
    """

    def __init__(self, cg: CandidateGenerator, client=None, model: str = LLM_MODEL,
                 cache_file=None):
        import anthropic
        self.cg = cg
        # oauth beta header is required for /v1/messages when authenticating with
        # a Bearer token from `ant auth login`; harmless with an API key.
        self.client = client or anthropic.Anthropic(
            default_headers={"anthropic-beta": "oauth-2025-04-20"})
        self.model = model
        self.cache_file = cache_file
        self._cache: Dict = _load_cache(cache_file)
        self.calls = 0
        self.errors = 0

    def _ask(self, mention: Mention, candidates: List[int]) -> int:
        lines = [f"  - concept_id {c}: {self.cg.concept_term(c)}" for c in candidates]
        prompt = (
            f"Clinical context (mention shown between <<< >>>):\n"
            f"...{mention.context.replace(mention.span, f'<<< {mention.span} >>>', 1)}...\n\n"
            f"Mention to link: \"{mention.span}\"\n\n"
            f"Candidate concepts:\n" + "\n".join(lines) + "\n\n"
            f"Select the best candidate."
        )
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=LLM_SYSTEM,
            tools=[SELECT_TOOL],
            tool_choice={"type": "tool", "name": "select_concept"},
            messages=[{"role": "user", "content": prompt}],
        )
        self.calls += 1
        for block in resp.content:
            if block.type == "tool_use":
                cid = int(block.input.get("concept_id", -1))
                return cid if cid in candidates else candidates[0]
        return candidates[0]

    def score(self, mention: Mention, candidates: List[int]) -> Dict[int, float]:
        if not candidates:
            return {}
        key = f"{self.model}|{mention.note_id}|{mention.start}|" \
              + ",".join(map(str, sorted(candidates)))
        if key not in self._cache:
            try:
                self._cache[key] = self._ask(mention, candidates)
            except Exception:
                self.errors += 1
                self._cache[key] = candidates[0]   # graceful fallback
            _save_cache(self.cache_file, self._cache)   # persist (resumable)
        chosen = self._cache[key]
        gmax = max(self.cg.global_freq.values()) if self.cg.global_freq else 1
        out = {c: 0.05 * (self.cg.global_freq.get(c, 0) / gmax) for c in candidates}
        out[chosen] = 1.0
        return out


# ---- local LLM ranker via Ollama (free, on-device — no data leaves the host) -
import json as _json
import urllib.request as _urlreq

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:3b-instruct"


class OllamaRanker(LLMRanker):
    """Stage-1 LLM baseline using a LOCAL model served by Ollama.

    Identical contract to ClaudeRanker, but calls the on-device Ollama server —
    so PhysioNet clinical text never leaves the machine (DUA-friendly) and it's
    free. Uses Ollama structured output with an enum-constrained schema, so the
    model is forced to return one of the candidate concept_ids (the local
    equivalent of strict tool use).

    Swap OLLAMA_MODEL for a larger model (e.g. qwen2.5:7b-instruct) for quality.
    """

    def __init__(self, cg: CandidateGenerator, model: str = OLLAMA_MODEL,
                 url: str = OLLAMA_URL, cache_file=None):
        self.cg = cg
        self.model = model
        self.url = url
        self.cache_file = cache_file
        self._cache: Dict = _load_cache(cache_file)
        self.calls = 0
        self.errors = 0

    def _ask(self, mention: Mention, candidates: List[int]) -> int:
        lines = [f"  - concept_id {c}: {self.cg.concept_term(c)}" for c in candidates]
        marked = mention.context.replace(mention.span, f"<<< {mention.span} >>>", 1)
        user = (
            f"Clinical context (mention between <<< >>>):\n...{marked}...\n\n"
            f"Mention to link: \"{mention.span}\"\n\n"
            f"Candidate SNOMED CT concepts:\n" + "\n".join(lines) + "\n\n"
            f"Return the concept_id of the single best-matching candidate."
        )
        schema = {
            "type": "object",
            "properties": {"concept_id": {"type": "integer", "enum": candidates}},
            "required": ["concept_id"],
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": LLM_SYSTEM},
                         {"role": "user", "content": user}],
            "stream": False,
            "format": schema,                       # enum-constrained output
            "options": {"temperature": 0, "num_predict": 40},
        }
        req = _urlreq.Request(
            self.url, data=_json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with _urlreq.urlopen(req, timeout=120) as r:
            resp = _json.loads(r.read())
        self.calls += 1
        cid = int(_json.loads(resp["message"]["content"])["concept_id"])
        return cid if cid in candidates else candidates[0]

    score = ClaudeRanker.score   # identical caching + scoring logic


# =============================================================================
# Semantic-type predictor  (GROUP A engine)
# =============================================================================
class SemTypePredictor:
    """Predicts a mention's SNOMED semantic type for pre-filtering.

    REAL plan (proposal): fine-tune a language model on (mention + context) ->
    semantic tag of the gold concept. Here it is SIMULATED: it returns the gold
    concept's (mock) type with probability `accuracy`, else a wrong type, plus a
    confidence — enough to exercise the confidence-fallback logic. Returns
    (predicted_type, confidence).

    Replace `predict()` with real model inference; the rest of Group A is unchanged.
    """

    def __init__(self, ontology: Ontology, accuracy: float = SEMTYPE_ACCURACY):
        self.ontology = ontology
        self.accuracy = accuracy

    def predict(self, mention: Mention):
        gold_type = self.ontology.semantic_type(mention.gold_concept)
        # deterministic pseudo-random draw from mention identity (reproducible)
        h = (hash((mention.note_id, mention.start)) % 1000) / 1000.0
        if h < self.accuracy:
            return gold_type, 0.5 + 0.5 * (1 - h)          # confident & correct
        wrong = MOCK_SEM_TYPES[(hash(mention.span) // 7) % len(MOCK_SEM_TYPES)]
        if wrong == gold_type:
            wrong = MOCK_SEM_TYPES[(MOCK_SEM_TYPES.index(wrong) + 1) % len(MOCK_SEM_TYPES)]
        return wrong, 0.4 + 0.3 * h                         # low-confidence wrong


# =============================================================================
# Strategies
# =============================================================================
def _argmax(scores: Dict[int, float]):
    return max(scores, key=scores.get) if scores else None


def _apply_filter(mention, candidates, predictor, ontology):
    """Group A: drop candidates whose semantic type != predicted type.
    Confidence fallback keeps all candidates to avoid false exclusion."""
    pred_type, conf = predictor.predict(mention)
    if conf < FILTER_CONFIDENCE:
        return candidates                                   # fallback: keep all
    kept = [c for c in candidates if ontology.semantic_type(c) == pred_type]
    return kept or candidates                               # never empty


def _cluster_medoid(candidates, ontology):
    """Reference point = structural centre of the candidate set (the candidate
    with minimum total is-a distance to the others). The proposal's 'main
    candidate cluster' — NOT the top-scored candidate, so an isolated winner can
    itself be penalised."""
    if len(candidates) <= 1:
        return candidates[0] if candidates else None
    best, best_tot = None, None
    for a in candidates:
        tot = sum(ontology.distance(a, b) for b in candidates if b != a)
        if best_tot is None or tot < best_tot:
            best, best_tot = a, tot
    return best


def _apply_rerank(scores, ontology, beta=BETA):
    """Group B: penalise candidates structurally distant from the cluster medoid.
    score' = score - beta * distance(medoid, candidate). Demotes isolated /
    granularity-mismatched predictions; leaves tightly-clustered sets untouched."""
    if not scores:
        return scores
    ref = _cluster_medoid(list(scores), ontology)
    return {c: s - beta * ontology.distance(ref, c) for c, s in scores.items()}


class Strategies:
    """Holds the shared engines and exposes one ranking fn per group."""

    def __init__(self, ranker: LLMRanker, predictor: SemTypePredictor,
                 ontology: Ontology):
        self.ranker = ranker
        self.predictor = predictor
        self.ontology = ontology

    def ranked(self, mention: Mention, group: str) -> List[int]:
        """Return candidates ranked best-first for the given group."""
        cands = list(mention.candidates)
        if group in ("A", "C"):
            cands = _apply_filter(mention, cands, self.predictor, self.ontology)
        scores = self.ranker.score(mention, cands)
        if group in ("B", "C"):
            scores = _apply_rerank(scores, self.ontology)
        return [c for c, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]

    def predict(self, mention: Mention, group: str):
        ranked = self.ranked(mention, group)
        return ranked[0] if ranked else None


GROUPS = {
    "Baseline": "baseline",
    "A: Pre-filter": "A",
    "B: Rerank": "B",
    "C: Filter+Rerank": "C",
}
