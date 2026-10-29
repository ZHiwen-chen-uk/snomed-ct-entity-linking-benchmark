"""Reciprocal Rank Fusion (RRF) over the candidate-generation retrievers.

Each retriever (train, synonym-exact, lexical, char n-gram, dense) produces its
own ranked list for a mention. RRF merges them by rank, not by score, so the
incompatible score scales (lexical token overlap vs SapBERT cosine) never have
to be normalised:

    RRF(c) = sum_r  1 / (k + rank_r(c))          # rank_r 1-based; absent -> 0

A concept ranked highly by any single retriever, or moderately by several, rises
to the top. This is what turns the dense recall-ceiling gain (tail reachability
0.79 -> 0.93) into usable early ranking: dense-only golds that a naive
concatenation buries at rank 60+ get surfaced into recall@30.

Shared by the local eval (test_rrf_fusion.py) and the Colab full-dataset runner.
"""
from candidate_booster import norm


def exact_hits(booster, span):
    return [int(c) for c in booster.syn_index.get(norm(span), ())]


def per_retriever_lists(mention, booster, char_retriever, dense_list, k):
    """Ranked list from each retriever for one mention. dense_list is precomputed
    (dense retrieval is batched for speed)."""
    lexical, _ = booster.retrieve([mention.span], k)
    char, _ = char_retriever.retrieve(mention.span, k)
    return {
        "train": list(mention.candidates),        # cg.generate order
        "synonym": exact_hits(booster, mention.span),
        "lexical": lexical,
        "char": char,
        "dense": list(dense_list),
    }


def rrf_fuse(ranked_lists, k=60, cap=None, weights=None):
    """Fuse ranked lists (dict name->list or list of lists) by RRF.

    weights: optional dict name->float (only when ranked_lists is a dict) to
    up/down-weight a retriever's contribution. Default all 1.0.
    """
    if isinstance(ranked_lists, dict):
        items = ranked_lists.items()
    else:
        items = [(str(i), lst) for i, lst in enumerate(ranked_lists)]
    scores = {}
    for name, lst in items:
        w = 1.0 if weights is None else float(weights.get(name, 1.0))
        if w == 0.0:
            continue
        for rank, cid in enumerate(lst):
            cid = int(cid)
            scores[cid] = scores.get(cid, 0.0) + w / (k + rank + 1)
    fused = sorted(scores, key=lambda c: scores[c], reverse=True)
    return fused[:cap] if cap else fused
