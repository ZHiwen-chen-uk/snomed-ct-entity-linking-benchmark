"""Ontology provider — the single place SNOMED CT structure enters the pipeline.

The rest of the code only ever talks to the `Ontology` interface, so swapping
the placeholder for the real terminology is a one-class change.

    semantic_type(concept) -> str     # e.g. "disorder", "procedure", "substance"
    depth(concept)         -> int     # shortest is-a path length to root
    distance(c1, c2)       -> int     # is-a path distance between two concepts

Two backends
------------
* ``SnomedOntology``  — REAL backend. Loads semantic tags from each concept's
  Fully Specified Name and the is-a graph from the SNOMED CT RF2 relationship
  file. Stubbed: implement once you have credentialed SNOMED CT access.

* ``MockOntology``    — SIMULATED backend so the pipeline runs end-to-end today.
  ===================================================================
  ||  WARNING: synthetic types & hierarchy. Numbers produced with   ||
  ||  MockOntology validate the *plumbing*, not the science.         ||
  ||  Replace with SnomedOntology for any reportable result.         ||
  ===================================================================
"""
from typing import Iterable

# Synthetic semantic-type labels, loosely mirroring real SNOMED top-level tags.
MOCK_SEM_TYPES = [
    "disorder", "finding", "procedure", "substance",
    "body structure", "organism", "qualifier value", "observable entity",
]


class Ontology:
    """Interface every group ranks against."""

    def semantic_type(self, concept: int) -> str:
        raise NotImplementedError

    def depth(self, concept: int) -> int:
        raise NotImplementedError

    def distance(self, c1: int, c2: int) -> int:
        raise NotImplementedError


class MockOntology(Ontology):
    """Deterministic synthetic ontology derived from concept_id alone.

    Stable across runs (pure function of the id), so experiments are
    reproducible, but it carries NO real clinical meaning. Its only job is to
    give Groups A/B/C something structurally consistent to operate on while the
    real terminology is unavailable.
    """

    def __init__(self, n_types: int = len(MOCK_SEM_TYPES), branching: int = 6,
                 max_depth: int = 7):
        self.n_types = n_types
        self.branching = branching
        self.max_depth = max_depth

    def semantic_type(self, concept: int) -> str:
        return MOCK_SEM_TYPES[concept % self.n_types]

    def _path(self, concept: int):
        """A synthetic root->concept path of node ids (for distance/depth)."""
        d = 1 + (concept // 7) % self.max_depth      # depth 1..max_depth
        path, node = [], concept
        for _ in range(d):
            node = node // self.branching
            path.append(node)
        return list(reversed(path))                  # root-most first

    def depth(self, concept: int) -> int:
        return len(self._path(concept))

    def distance(self, c1: int, c2: int) -> int:
        if c1 == c2:
            return 0
        p1, p2 = self._path(c1), self._path(c2)
        # length of shared prefix = depth of lowest common ancestor
        lca = 0
        for a, b in zip(p1, p2):
            if a == b:
                lca += 1
            else:
                break
        # cross-type concepts live in different sub-trees -> add a modest gap
        cross = 0 if self.semantic_type(c1) == self.semantic_type(c2) else 2
        return (len(p1) - lca) + (len(p2) - lca) + cross


class SnomedOntology(Ontology):
    """REAL backend — fill in with credentialed SNOMED CT release files.

    Expected inputs (RF2):
      * sct2_Description / FSN  -> parse trailing "(disorder)" etc. as sem_type
      * sct2_Relationship       -> is-a (typeId 116680003) edges -> BFS for
                                    depth and distance (cache with lru_cache)
    """

    def __init__(self, release_dir):
        raise NotImplementedError(
            "Plug in SNOMED CT RF2 files here. Parse semantic tags from FSNs and "
            "build the is-a graph for depth()/distance(). Until then use "
            "MockOntology(); see README 'Swapping in real SNOMED CT'."
        )
