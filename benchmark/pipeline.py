"""The four experimental groups from the proposal.

    Baseline : LLM picks best candidate, no ontology signal.
    Group A  : ontology PRE-FILTER (predicted semantic type removes
               incompatible candidates) -> then the LLM ranks.
    Group B  : LLM ranks -> ontology POST-HOC RERANK (hierarchy distance
               penalty, weight beta) targets granularity mismatch.
    Group C  : A then B (filter, rank, rerank).

Every group shares the same candidate set (data.CandidateGenerator) and the same
LLM ranker, so any difference is attributable to the ontology strategy alone. All
ontology structure comes from the REAL SnomedOntology (ontology.py).
"""
import json as _json
import pickle
import urllib.request as _urlreq
from pathlib import Path as _Path
from typing import Dict, List

from data import CandidateGenerator, Mention
from ontology import Ontology, COARSE_LABELS

# ---- tunable hyper-parameters (proposal: tuned on a validation split) --------
BETA = 0.04                  # Group B: hierarchy-distance penalty weight
FILTER_CONFIDENCE = 0.55     # Group A: keep-all fallback below this confidence


# =============================================================================
# shared cache helpers
# =============================================================================
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


# =============================================================================
# LLM ranker  (BASELINE engine, shared by all groups)
# =============================================================================
LLM_SYSTEM = (
    "You are a clinical terminologist performing SNOMED CT entity linking. "
    "You are shown a mention from a discharge note, its surrounding context, and "
    "a fixed list of candidate SNOMED CT concepts (each an id and a representative "
    "term). Choose the single candidate that best captures the mention's meaning "
    "in context. You must pick exactly one candidate from the list. Do not invent "
    "concept ids."
)

LLM_MODEL = "claude-opus-4-8"


class LLMRanker:
    """Scores each candidate for a mention. Higher = better. Subclass and
    override `score()` to call a real model (ClaudeRanker / OllamaRanker)."""

    def score(self, mention: Mention, candidates: List[int]) -> Dict[int, float]:
        raise NotImplementedError


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


def _rank_prompt(mention: Mention, candidates: List[int], cg: CandidateGenerator) -> str:
    lines = [f"  - concept_id {c}: {cg.concept_term(c)}" for c in candidates]
    marked = mention.context.replace(mention.span, f"<<< {mention.span} >>>", 1)
    return (
        f"Clinical context (mention between <<< >>>):\n...{marked}...\n\n"
        f"Mention to link: \"{mention.span}\"\n\n"
        f"Candidate SNOMED CT concepts:\n" + "\n".join(lines) + "\n\n"
        f"Select the best-matching candidate."
    )


class ClaudeRanker(LLMRanker):
    """Stage-1 LLM baseline using the real Claude API (claude-opus-4-8)."""

    def __init__(self, cg: CandidateGenerator, client=None, model: str = LLM_MODEL,
                 cache_file=None):
        import anthropic
        self.cg = cg
        self.client = client or anthropic.Anthropic(
            default_headers={"anthropic-beta": "oauth-2025-04-20"})
        self.model = model
        self.cache_file = cache_file
        self._cache: Dict = _load_cache(cache_file)
        self.calls = 0
        self.errors = 0

    def _ask(self, mention: Mention, candidates: List[int]) -> int:
        resp = self.client.messages.create(
            model=self.model, max_tokens=1024, system=LLM_SYSTEM,
            tools=[SELECT_TOOL],
            tool_choice={"type": "tool", "name": "select_concept"},
            messages=[{"role": "user",
                       "content": _rank_prompt(mention, candidates, self.cg)}],
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
                self._cache[key] = candidates[0]        # graceful fallback
            _save_cache(self.cache_file, self._cache)
        chosen = self._cache[key]
        gmax = max(self.cg.global_freq.values()) if self.cg.global_freq else 1
        out = {c: 0.05 * (self.cg.global_freq.get(c, 0) / gmax) for c in candidates}
        out[chosen] = 1.0
        return out


OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:3b-instruct"


def _ollama_json(url, model, system, user, schema, num_predict=40):
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "stream": False,
        "format": schema,
        "options": {"temperature": 0, "num_predict": num_predict},
    }
    req = _urlreq.Request(url, data=_json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
    with _urlreq.urlopen(req, timeout=120) as r:
        resp = _json.loads(r.read())
    return _json.loads(resp["message"]["content"])


class OllamaRanker(LLMRanker):
    """Stage-1 LLM baseline using a LOCAL model served by Ollama.

    Same contract as ClaudeRanker, but on-device, PhysioNet clinical text never
    leaves the machine (DUA-friendly) and it is free. Uses Ollama structured
    output with an enum-constrained schema, so the model must return one of the
    candidate concept_ids.
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
        schema = {"type": "object",
                  "properties": {"concept_id": {"type": "integer",
                                                "enum": candidates}},
                  "required": ["concept_id"]}
        out = _ollama_json(self.url, self.model, LLM_SYSTEM,
                           _rank_prompt(mention, candidates, self.cg), schema)
        self.calls += 1
        cid = int(out["concept_id"])
        return cid if cid in candidates else candidates[0]

    score = ClaudeRanker.score   # identical caching + scoring logic


# =============================================================================
# Semantic-type predictor  (GROUP A engine)
# =============================================================================
class SemTypePredictor:
    """Predicts a mention's coarse SNOMED semantic type for pre-filtering.

    Interface: ``predict(mention) -> (coarse_type, confidence)`` where
    coarse_type is one of ontology.COARSE_LABELS. Subclasses implement a real
    model; the confidence feeds Group A's keep-all fallback.
    """

    def predict(self, mention: Mention):
        raise NotImplementedError


SEMTYPE_SYSTEM = (
    "You are a clinical terminologist. Given a mention from a discharge note and "
    "its surrounding context, classify which SNOMED CT top-level semantic "
    "category the mention refers to IN THIS CONTEXT. Report a calibrated "
    "confidence in [0,1]. Categories: "
    "finding (disorders/symptoms/findings), procedure (interventions/therapies), "
    "substance (chemicals/drugs as substances), product (medicinal products), "
    "body structure (anatomy), organism, observable entity (measurable "
    "properties), situation (context/absent/family history), event, specimen, "
    "physical object, environment, person, qualifier value, other."
)


class LLMSemTypePredictor(SemTypePredictor):
    """Zero-shot semantic-type predictor using the local Ollama model (or Claude).

    Reads only the mention + context (never the gold concept). Enum-constrained
    structured output over the coarse label space. Cached per (note, offset).

    NOTE: this is the low-risk first predictor. The proposal specifies a
    fine-tuned language model; that swaps in behind this same interface without
    touching Group A's filter logic.
    """

    def __init__(self, backend="ollama", model=OLLAMA_MODEL, url=OLLAMA_URL,
                 client=None, cache_file=None):
        self.backend = backend
        self.model = model
        self.url = url
        self.client = client
        self.cache_file = cache_file
        self._cache: Dict = _load_cache(cache_file)
        self.calls = 0
        self.errors = 0

    def _prompt(self, mention: Mention) -> str:
        marked = mention.context.replace(mention.span, f"<<< {mention.span} >>>", 1)
        return (f"Clinical context (mention between <<< >>>):\n...{marked}...\n\n"
                f"Mention: \"{mention.span}\"\n\n"
                f"Which SNOMED CT semantic category does it refer to here?")

    def _ask(self, mention: Mention):
        schema = {"type": "object",
                  "properties": {"semantic_type": {"type": "string",
                                                   "enum": COARSE_LABELS},
                                 "confidence": {"type": "number"}},
                  "required": ["semantic_type", "confidence"]}
        if self.backend == "claude":
            return self._ask_claude(mention, schema)
        out = _ollama_json(self.url, self.model, SEMTYPE_SYSTEM,
                           self._prompt(mention), schema)
        self.calls += 1
        return out["semantic_type"], float(out.get("confidence", 1.0))

    def _ask_claude(self, mention, schema):
        tool = {"name": "classify_type", "strict": True,
                "input_schema": {"type": "object",
                                 "properties": {
                                     "semantic_type": {"type": "string",
                                                       "enum": COARSE_LABELS},
                                     "confidence": {"type": "number"}},
                                 "required": ["semantic_type", "confidence"],
                                 "additionalProperties": False}}
        resp = self.client.messages.create(
            model=self.model, max_tokens=256, system=SEMTYPE_SYSTEM,
            tools=[tool], tool_choice={"type": "tool", "name": "classify_type"},
            messages=[{"role": "user", "content": self._prompt(mention)}])
        self.calls += 1
        for block in resp.content:
            if block.type == "tool_use":
                return (block.input["semantic_type"],
                        float(block.input.get("confidence", 1.0)))
        return "other", 0.0

    def predict(self, mention: Mention):
        key = f"{self.model}|{mention.note_id}|{mention.start}"
        if key not in self._cache:
            try:
                self._cache[key] = list(self._ask(mention))
            except Exception:
                self.errors += 1
                self._cache[key] = ["other", 0.0]       # low-conf -> keep all
            _save_cache(self.cache_file, self._cache)
        t, c = self._cache[key]
        return t, float(c)


class SklearnSemTypePredictor(SemTypePredictor):
    """Trained coarse-type classifier (train_semtype.py). Its predict_proba gives a
    CALIBRATED confidence, so Group A's confidence fallback actually works, unlike
    the overconfident zero-shot LLM. Uses the mention's section header + span +
    context (same feature string as training)."""

    def __init__(self, model_path):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        self.classes = list(self.model.classes_)
        self.calls = 0

    def predict(self, mention: Mention):
        from data import semtype_feature_text
        x = semtype_feature_text(mention.section, mention.span, mention.context)
        proba = self.model.predict_proba([x])[0]
        self.calls += 1
        i = int(proba.argmax())
        return self.classes[i], float(proba[i])


class AgreementPredictor(SemTypePredictor):
    """Combine two predictors: filter only when both AGREE and the calibrated
    (primary) confidence clears the bar; otherwise emit low confidence so Group A
    keeps all candidates. Agreement between two independent predictors is a far more
    reliable 'safe to filter' signal than either model's self-reported confidence,     it directly suppresses the high-confidence-but-wrong deletions that made the
    single zero-shot predictor a coin-flip."""

    def __init__(self, primary: SemTypePredictor, secondary: SemTypePredictor,
                 min_prob=FILTER_CONFIDENCE):
        self.primary = primary       # calibrated (sklearn)
        self.secondary = secondary   # independent second opinion (LLM)
        self.min_prob = min_prob
        self.calls = 0

    def predict(self, mention: Mention):
        pt, pp = self.primary.predict(mention)
        st, _ = self.secondary.predict(mention)
        self.calls += 1
        if pt == st and pp >= self.min_prob:
            return pt, pp                # both agree & confident -> filter
        return pt, 0.0                   # disagree / unsure -> keep all


# =============================================================================
# Strategies
# =============================================================================
def _apply_filter(mention, candidates, predictor, ontology):
    """Group A: keep only candidates whose coarse semantic type matches the
    predicted type. Definitional exclusion, not reordering. Two safeguards from
    the proposal: (1) confidence fallback keeps all candidates when the predictor
    is unsure; (2) never return empty (a predictor error must not delete the gold
    candidate outright)."""
    pred_type, conf = predictor.predict(mention)
    if conf < FILTER_CONFIDENCE:
        return candidates
    kept = [c for c in candidates if ontology.coarse_type(c) == pred_type]
    return kept or candidates


def _cluster_medoid(candidates, ontology):
    """Reference point = structural centre of the candidate set (the candidate
    with minimum total is-a distance to the others). NOT the top-scored
    candidate, so an isolated / granularity-mismatched winner can be penalised."""
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
    score' = score - beta * distance(medoid, candidate). Targets granularity
    mismatch (over-broad / over-specific outliers); leaves tightly-clustered
    sibling sets essentially untouched (they are all near the medoid)."""
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
