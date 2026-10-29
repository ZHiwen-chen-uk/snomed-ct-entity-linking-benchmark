"""Ontology provider, the single place SNOMED CT structure enters the pipeline.

The rest of the code only ever talks to the `Ontology` interface, so the real
terminology is the one and only source of structural signal:

    semantic_type(concept) -> str     # raw FSN tag, e.g. "disorder", "procedure"
    coarse_type(concept)   -> str     # ~11 filter buckets (see COARSE_TYPE)
    depth(concept)         -> int     # shortest is-a path length to root
    distance(c1, c2)       -> int     # is-a path distance between two concepts

`SnomedOntology` loads the SNOMED CT International RF2 release (semantic tags from
each concept's Fully Specified Name; the is-a graph from the relationship file).
"""
import pickle
import re
from pathlib import Path


# --- coarse semantic-type buckets -------------------------------------------
# SNOMED has ~150 FSN semantic tags; that is too fine for a mention-level type
# predictor (and disorder/finding, product/drug are routinely conflated). Group A
# predicts one of these coarse buckets and keeps only candidates whose tag maps
# to it. The buckets preserve the cross-branch distinctions that actually matter
# for disambiguation (substance vs finding vs procedure, the "aspirin" case).
COARSE_TYPE = {
    "disorder": "finding", "finding": "finding",
    "procedure": "procedure", "regime/therapy": "procedure",
    "body structure": "body structure", "cell structure": "body structure",
    "cell": "body structure", "morphologic abnormality": "body structure",
    "substance": "substance",
    "product": "product", "clinical drug": "product",
    "medicinal product": "product", "medicinal product form": "product",
    "organism": "organism",
    "qualifier value": "qualifier value",
    "observable entity": "observable entity",
    "situation": "situation",
    "event": "event",
    "specimen": "specimen",
    "physical object": "physical object",
    "environment": "environment", "environment / location": "environment",
    "occupation": "person", "person": "person",
}
# the label space the semantic-type predictor chooses from
COARSE_LABELS = sorted(set(COARSE_TYPE.values())) + ["other"]


def coarsen(tag: str) -> str:
    return COARSE_TYPE.get(tag, "other")


class Ontology:
    """Interface every group ranks against."""

    def semantic_type(self, concept: int) -> str:
        raise NotImplementedError

    def coarse_type(self, concept: int) -> str:
        return coarsen(self.semantic_type(concept))

    def depth(self, concept: int) -> int:
        raise NotImplementedError

    def distance(self, c1: int, c2: int) -> int:
        raise NotImplementedError


class SnomedOntology(Ontology):
    """REAL backend, SNOMED CT International RF2 release.

    Parses the Snapshot/Terminology files once and caches a compact pickle so
    later runs load in ~1s instead of re-parsing ~600MB:

      * sct2_Concept       -> set of active concept ids
      * sct2_Description    -> active FSN (typeId 900000000000003001); the
                               trailing "(disorder)" tag is the semantic type
      * sct2_Relationship   -> active is-a edges (typeId 116680003); child ->
                               parents adjacency, from which depth() (BFS from
                               root) and distance() (via lowest common ancestor)
                               are derived.

    Semantic distance model: SNOMED is a multi-parent DAG, so distance(a, b) is
    the LCA path length using each concept's shortest depth to root
    ``depth(a) + depth(b) - 2 * depth(lca)``, the standard is-a taxonomic
    distance. Unknown/retired concepts degrade gracefully (type "unknown",
    depth 0, a large sentinel distance) rather than raising.
    """

    ROOT = 138875005                 # | SNOMED CT Concept | (hierarchy root)
    IS_A = "116680003"               # is-a relationship typeId
    FSN = "900000000000003001"       # Fully Specified Name description typeId
    _TAG_RE = re.compile(r"\(([^()]+)\)\s*$")
    FAR = 40                         # sentinel distance when no path / unknown

    def __init__(self, release_dir, cache_file=None, rebuild=False):
        self.release_dir = Path(release_dir)
        self.cache_file = Path(cache_file) if cache_file else (
            self.release_dir / ".snomed_ontology_cache.pkl")
        self._anc_cache = {}         # concept -> {ancestor: depth} (memoised)
        self.pref = {}               # conceptId -> preferred term (FSN minus tag)
        if self.cache_file.exists() and not rebuild:
            self._load_cache()
        else:
            self._build()
            self._save_cache()

    # ---- construction --------------------------------------------------------
    def _terminology_dir(self):
        hits = list(self.release_dir.glob(
            "**/Snapshot/Terminology/sct2_Concept_Snapshot_*.txt"))
        if not hits:
            raise FileNotFoundError(
                f"No RF2 Snapshot/Terminology under {self.release_dir}")
        return hits[0].parent

    def _build(self):
        term = self._terminology_dir()
        concept_f = next(term.glob("sct2_Concept_Snapshot_*.txt"))
        desc_f = next(term.glob("sct2_Description_Snapshot-en_*.txt"))
        rel_f = next(term.glob("sct2_Relationship_Snapshot_*.txt"))

        active = set()
        with open(concept_f, encoding="utf-8") as f:
            next(f)
            for line in f:
                c = line.split("\t")
                if c[2] == "1":
                    active.add(int(c[0]))

        self.sem = {}                # conceptId -> semantic tag
        with open(desc_f, encoding="utf-8") as f:
            next(f)
            for line in f:
                c = line.rstrip("\n").split("\t")
                if len(c) < 9 or c[2] != "1" or c[6] != self.FSN:
                    continue
                cid = int(c[4]); term_txt = c[7]
                m = self._TAG_RE.search(term_txt)
                self.sem[cid] = m.group(1) if m else "unknown"
                # preferred term = FSN with the trailing "(tag)" stripped
                self.pref[cid] = (term_txt[:m.start()].strip() if m else term_txt)

        self.parents = {}            # child -> [parents] (active is-a only)
        with open(rel_f, encoding="utf-8") as f:
            next(f)
            for line in f:
                c = line.split("\t")
                if c[2] != "1" or c[7] != self.IS_A:
                    continue
                child, parent = int(c[4]), int(c[5])
                if child in active and parent in active:
                    self.parents.setdefault(child, []).append(parent)

        self._compute_depths(active)

    def _compute_depths(self, active):
        """Shortest depth from ROOT via a BFS over parent->children edges."""
        children = {}
        for child, ps in self.parents.items():
            for p in ps:
                children.setdefault(p, []).append(child)
        self.depths = {self.ROOT: 0}
        frontier = [self.ROOT]
        d = 0
        while frontier:
            d += 1
            nxt = []
            for node in frontier:
                for ch in children.get(node, ()):
                    if ch not in self.depths:
                        self.depths[ch] = d
                        nxt.append(ch)
            frontier = nxt

    # ---- persistence ---------------------------------------------------------
    def _save_cache(self):
        with open(self.cache_file, "wb") as f:
            pickle.dump({"sem": self.sem, "pref": self.pref,
                         "parents": self.parents, "depths": self.depths},
                        f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_cache(self):
        with open(self.cache_file, "rb") as f:
            d = pickle.load(f)
        self.sem, self.parents, self.depths = d["sem"], d["parents"], d["depths"]
        self.pref = d.get("pref", {})

    # ---- Ontology interface --------------------------------------------------
    def semantic_type(self, concept: int) -> str:
        return self.sem.get(int(concept), "unknown")

    def preferred_term(self, concept: int) -> str:
        return self.pref.get(int(concept), f"concept:{concept}")

    def depth(self, concept: int) -> int:
        return self.depths.get(int(concept), 0)

    def _ancestors(self, concept: int):
        """{ancestor_id: depth} including the concept itself; memoised."""
        concept = int(concept)
        if concept in self._anc_cache:
            return self._anc_cache[concept]
        seen = {concept: self.depths.get(concept, 0)}
        stack = list(self.parents.get(concept, ()))
        while stack:
            a = stack.pop()
            if a in seen:
                continue
            seen[a] = self.depths.get(a, 0)
            stack.extend(self.parents.get(a, ()))
        self._anc_cache[concept] = seen
        return seen

    def is_descendant(self, concept: int, ancestor: int) -> bool:
        """True if `concept` is `ancestor` or lies in its is-a subtree."""
        return int(ancestor) in self._ancestors(concept)

    def distance(self, c1: int, c2: int) -> int:
        c1, c2 = int(c1), int(c2)
        if c1 == c2:
            return 0
        a1, a2 = self._ancestors(c1), self._ancestors(c2)
        common = a1.keys() & a2.keys()
        if not common:
            return self.FAR
        lca_depth = max(a1[x] for x in common)
        return (self.depths.get(c1, 0) - lca_depth) + (self.depths.get(c2, 0) - lca_depth)
