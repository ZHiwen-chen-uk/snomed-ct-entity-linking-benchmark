"""Does the in-index blind spot show up on a second corpus? Test it on BC5CDR.

On the SNOMED CT Entity Linking Challenge corpus, a reranker trained on
in-domain annotations fails on concepts that are absent from the training-derived
index, and recomputing the index-derived features out-of-fold repairs it. The
dose-response sweep pins the failure to one quantity. All of that is still one
corpus, one ontology and one kind of note, so this script runs the same
comparison somewhere with none of those in common.

    corpus     BioCreative V CDR, 1,500 PubMed abstracts split 500 train,
               500 dev, 500 test, with chemical and disease mentions
               normalised to MeSH
    ontology   the CTD disease and chemical vocabularies (MeSH names plus
               synonyms). These give the external concept index, which is what
               lets a gold absent from the training annotations still be
               reachable
    task       given a gold span, rank the candidates and pick a concept

The mechanism has nothing to do with SNOMED. All it needs is that the
annotations training the model are also what fill its retrieval index. BC5CDR
works that way too, so the same collapse should appear and the same repair
should remove it.

This pipeline is weaker than the main one on purpose. There is no dense
retrieval, no learned type predictor and no cross-encoder, because what is being
tested is the supervision signal and not the quality of the retriever. The
reachability the weaker pool reaches is reported so a reader can judge that
choice.

    KMP_DUPLICATE_LIB_OK=TRUE python replicate_blind_spot.py
"""
import gzip
import pickle
import re
import sys
import time
from collections import Counter, defaultdict
from hashlib import md5
from heapq import nlargest
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CDR = DATA / "CDR_Data" / "CDR.Corpus.v010516"
OUT = HERE / "out"
CACHE = HERE / ".cache"

POOL = 60
RRF_K = 60
SEED = 20260723
RRF_WEIGHTS = {"train": 2.0, "exact": 3.0, "lexical": 1.0, "char": 1.0}
FNAMES = ["exact", "lexical", "lex_rr", "char", "char_rr",
          "train_rr", "prior", "log_freq", "in_train"]
LEAKY = ["train_rr", "prior", "log_freq", "in_train"]
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def norm(s):
    return str(s).strip().lower()


# --------------------------------------------------------------- corpus
def read_pubtator(path):
    """-> list of dicts (pmid, start, end, text, type, concept)."""
    rows = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 6:
            continue
        pmid, start, end, text, typ, cid = parts[:6]
        cid = cid.strip()
        if cid in ("-1", "", "MESH:-1"):
            continue                      # unnormalisable, excluded
        cid = cid.split("|")[0].replace("MESH:", "")   # composite -> first
        if not cid:
            continue
        rows.append({"note_id": pmid, "start": int(start), "end": int(end),
                     "span": text, "type": typ, "mesh": cid})
    return rows


def load_corpus():
    splits = {}
    for name, fn in (("train", "CDR_TrainingSet.PubTator.txt"),
                     ("dev", "CDR_DevelopmentSet.PubTator.txt"),
                     ("test", "CDR_TestSet.PubTator.txt")):
        splits[name] = read_pubtator(CDR / fn)
    return splits


# --------------------------------------------------------------- ontology
def load_ctd():
    """MeSH id -> list of surface names, from the CTD disease + chemical files."""
    names = defaultdict(list)
    for fn, id_col, name_col, syn_col in (
            ("CTD_diseases.tsv.gz", 1, 0, 7),
            ("CTD_chemicals.tsv.gz", 1, 0, 7)):
        with gzip.open(DATA / fn, "rt", encoding="utf-8") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                p = line.rstrip("\n").split("\t")
                if len(p) <= max(id_col, name_col, syn_col):
                    continue
                cid = p[id_col].replace("MESH:", "").strip()
                if not cid.startswith(("D", "C")):
                    continue
                vals = [p[name_col]] + (p[syn_col].split("|") if p[syn_col] else [])
                for v in vals:
                    v = v.strip()
                    if v:
                        names[cid].append(v)
    return {k: sorted(set(v)) for k, v in names.items()}


_TIE = {}


def tie_key(cid):
    """Neutral deterministic tie-break, memoised.

    Sorting ties on the raw concept id is NOT neutral for MeSH: `C` ids are
    supplementary records, which skew rare, and `D` ids are main descriptors,
    which skew common, so ascending order would systematically favour exactly
    the rare concepts this experiment measures. A fixed digest of the id is
    uncorrelated with rarity and still fully reproducible. It is memoised
    because it sits inside sort keys on every mention.
    """
    v = _TIE.get(cid)
    if v is None:
        v = _TIE[cid] = int(md5(cid.encode()).hexdigest()[:12], 16)
    return v


class NameIndex:
    """Exact / token / character-trigram retrieval over the CTD name index.

    Scores are length-normalised (Jaccard for trigrams, IDF-weighted coverage
    for tokens) rather than raw counts. Raw counts produce large numbers of
    exact ties, and that hands the outcome to whatever breaks them. The pool
    would then be decided by an arbitrary convention instead of by the retrieval
    signal.
    """

    def __init__(self, names):
        self.exact = defaultdict(set)
        self.tok = defaultdict(set)
        self.tri = defaultdict(set)
        self.tri_n = Counter()          # trigram-set size of each concept
        self.n_names = 0
        for cid, forms in names.items():
            cid_tris = set()
            for f in forms:
                nf = norm(f)
                if not nf:
                    continue
                self.n_names += 1
                self.exact[nf].add(cid)
                for t in set(TOKEN_RE.findall(nf)):
                    self.tok[t].add(cid)
                s = f" {nf} "
                cid_tris.update(s[i:i + 3] for i in range(len(s) - 2))
            for g in cid_tris:
                self.tri[g].add(cid)
            self.tri_n[cid] = len(cid_tris)
        self.tok = {k: v for k, v in self.tok.items() if len(v) <= 20000}
        self.tri = {k: v for k, v in self.tri.items() if len(v) <= 40000}

    def retrieve(self, form, k):
        nf = norm(form)
        ex = sorted(self.exact.get(nf, ()), key=tie_key)
        # Tokens are iterated in SORTED order, not set order. Each concept
        # receives exactly one += per matching token, so fixing the token order
        # fixes the order of the float additions. Without this the sums differ
        # in their last bits between runs, because floating-point addition is
        # not associative and set iteration order follows the per-process string
        # hash seed. Those tiny differences then reorder the pool.
        toks = sorted(set(TOKEN_RE.findall(nf)))
        lex = Counter()
        for t in toks:
            post = self.tok.get(t)
            if post:
                w = 1.0 / float(np.log1p(len(post) + 1))
                for c in post:
                    lex[c] += w
        norm_l = sum(1.0 / float(np.log1p(len(self.tok.get(t, ())) + 1))
                     for t in toks) or 1.0
        s = f" {nf} "
        tris = sorted({s[i:i + 3] for i in range(len(s) - 2)})
        ov = Counter()
        for g in tris:
            post = self.tri.get(g)
            if post:
                for c in post:
                    ov[c] += 1
        # Jaccard, overlap / (|query| + |candidate| - overlap). It is continuous,
        # so exact ties are rare and the pool follows the signal, not the sort.
        ch = {c: v / max(len(tris) + self.tri_n.get(c, 0) - v, 1)
              for c, v in ov.items()}
        rank = lambda kv: (kv[1], -tie_key(kv[0]))   # score desc, tie asc
        lex_top = nlargest(k, ((c, v / norm_l) for c, v in lex.items()), key=rank)
        ch_top = nlargest(k, ch.items(), key=rank)
        return ex, lex_top, ch_top


def rrf(lists, k=RRF_K, cap=POOL, weights=None):
    score = defaultdict(float)
    for name, lst in lists.items():
        w = (weights or {}).get(name, 1.0)
        for r, c in enumerate(lst):
            score[c] += w / (k + r + 1)
    return [c for c, _ in nlargest(cap, score.items(),
                                   key=lambda kv: (kv[1], -tie_key(kv[0])))]


# --------------------------------------------------------------- train index
class TrainIndex:
    """The training-annotation-derived index: exactly the structure that causes
    the blind spot. Mirrors benchmark/data.py CandidateGenerator."""

    def __init__(self, rows, exclude_note=None):
        self.form2c = defaultdict(Counter)
        self.tok2c = defaultdict(Counter)
        self.freq = Counter()
        for r in rows:
            if exclude_note is not None and r["note_id"] == exclude_note:
                continue
            f = norm(r["span"])
            if not f:
                continue
            self.form2c[f][r["mesh"]] += 1
            self.freq[r["mesh"]] += 1
            for t in set(TOKEN_RE.findall(f)):
                self.tok2c[t][r["mesh"]] += 1

    def generate(self, form, k=30):
        """Candidates from the training annotations.

        Both the token loop and the top-k are order-hardened. `Counter.
        most_common` breaks ties by insertion order, and the insertion order
        here would otherwise follow set iteration, which varies with the
        per-process string hash seed. That is enough to change the fused pool
        and, through it, to move the reported accuracies by several points.
        """
        f = norm(form)
        sc = Counter()
        for c, n in self.form2c.get(f, {}).items():
            sc[c] += 5 * n
        for t in sorted(set(TOKEN_RE.findall(f))):
            post = self.tok2c.get(t)
            if post:
                for c, n in nlargest(20, post.items(),
                                     key=lambda kv: (kv[1], -tie_key(kv[0]))):
                    sc[c] += n
        return [c for c, _ in nlargest(k, sc.items(),
                                       key=lambda kv: (kv[1], -tie_key(kv[0])))]

    def prior(self, form, cid):
        c = self.form2c.get(norm(form))
        tot = sum(c.values()) if c else 0
        return (c.get(cid, 0) / tot) if tot else 0.0


def features(cid, ex, lex_s, lex_r, ch_s, ch_r, tr_rank, tidx, form):
    return [
        1.0 if cid in ex else 0.0,
        lex_s.get(cid, 0.0), 1.0 / (lex_r[cid] + 1) if cid in lex_r else 0.0,
        ch_s.get(cid, 0.0), 1.0 / (ch_r[cid] + 1) if cid in ch_r else 0.0,
        1.0 / (tr_rank[cid] + 1) if cid in tr_rank else 0.0,
        tidx.prior(form, cid),
        float(np.log1p(tidx.freq.get(cid, 0))),
        1.0 if cid in tidx.freq else 0.0,
    ]


def md_table(df):
    """Pipe table without the optional `tabulate` dependency."""
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |",
           "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if pd.isna(r[c]) else str(r[c])
                                     for c in cols) + " |")
    return "\n".join(out)


def stratum(cid, train_freq):
    f = train_freq.get(cid, 0)
    if f == 0 or f < 3:
        return "tail"
    return "head" if f >= 20 else "torso"


def scopes_of(cid, train_freq):
    s = stratum(cid, train_freq)
    out = ["all", s]
    if s == "tail":
        out.append("tail_out_of_index" if train_freq.get(cid, 0) == 0
                   else "tail_in_index")
    return out


def build(rows, tidx, nidx, tag, t0, lono_rows=None):
    """Feature matrices for `rows`, in one pass over the (slow) name index.

    The POOL is always built from the FULL training index, so that the standard
    and leave-one-out variants score an identical candidate set and only the
    four train-derived features differ. This mirrors the main study's design
    decision, where the repair is applied to the supervision signal and not to
    retrieval. Without it the two conditions would differ in two ways at once,
    which is not a controlled comparison.

    Returns (X_standard, X_lono_or_None, y, pools).
    """
    X, XL, y, pools = [], [], [], []
    per_note = {}
    for i, r in enumerate(rows):
        form = r["span"]
        ex, lex_top, ch_top = nidx.retrieve(form, POOL)
        lex_s = dict(lex_top); ch_s = dict(ch_top)
        lex_r = {c: n for n, (c, _) in enumerate(lex_top)}
        ch_r = {c: n for n, (c, _) in enumerate(ch_top)}

        tr = tidx.generate(form, POOL)                 # full index -> the pool
        tr_rank = {c: n for n, c in enumerate(tr)}
        pool = rrf({"train": tr, "exact": ex,
                    "lexical": [c for c, _ in lex_top],
                    "char": [c for c, _ in ch_top]}, weights=RRF_WEIGHTS)

        if lono_rows is not None:
            nid = r["note_id"]
            if nid not in per_note:
                per_note[nid] = TrainIndex(lono_rows, exclude_note=nid)
            lidx = per_note[nid]
            l_rank = {c: n for n, c in enumerate(lidx.generate(form, POOL))}

        for c in pool:
            X.append(features(c, ex, lex_s, lex_r, ch_s, ch_r, tr_rank, tidx, form))
            if lono_rows is not None:
                XL.append(features(c, ex, lex_s, lex_r, ch_s, ch_r, l_rank, lidx, form))
            y.append(1 if c == r["mesh"] else 0)
        pools.append(pool)
        if (i + 1) % 2000 == 0:
            print(f"    {tag} {i+1}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)
    return (np.asarray(X, dtype=np.float32),
            np.asarray(XL, dtype=np.float32) if lono_rows is not None else None,
            np.asarray(y, dtype=np.int8), pools)


def accuracies(picks, rows, train_freq):
    scope_hits, scope_n = Counter(), Counter()
    for i, r in enumerate(rows):
        ok = picks.get(i) == r["mesh"]
        for s in scopes_of(r["mesh"], train_freq):
            scope_n[s] += 1
            scope_hits[s] += ok
    return {s: round(scope_hits[s] / scope_n[s], 4) if scope_n[s] else None
            for s in ("all", "head", "torso", "tail",
                      "tail_out_of_index", "tail_in_index")}, scope_n


def picks_from(proba, meta, pools, rows):
    out, off = {}, 0
    for i, pool in enumerate(pools):
        n = len(pool)
        if n:
            out[i] = pool[int(np.argmax(proba[off:off + n]))]
        off += n
    return out


def main():
    t0 = time.time()
    OUT.mkdir(exist_ok=True); CACHE.mkdir(exist_ok=True)
    splits = load_corpus()
    tr_rows, ev_rows = splits["train"], splits["test"]
    print(f"train mentions {len(tr_rows)} from {len({r['note_id'] for r in tr_rows})} abstracts", flush=True)
    print(f"test  mentions {len(ev_rows)} from {len({r['note_id'] for r in ev_rows})} abstracts", flush=True)

    names = load_ctd()
    print(f"CTD concepts {len(names)} ({time.time()-t0:.0f}s)", flush=True)
    nidx_path = CACHE / "nameindex.pkl"
    if nidx_path.exists():
        nidx = pickle.load(open(nidx_path, "rb"))
        print(f"name index loaded ({nidx.n_names} names)", flush=True)
    else:
        nidx = NameIndex(names)
        pickle.dump(nidx, open(nidx_path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        print(f"name index built: {nidx.n_names} names ({time.time()-t0:.0f}s)", flush=True)

    tidx = TrainIndex(tr_rows)
    train_freq = Counter(r["mesh"] for r in tr_rows)

    # ---- manipulation check: how much index variance does LONO create? ------
    gold_in = sum(1 for r in tr_rows if r["mesh"] in tidx.freq)
    lono_in = 0
    per_note = {}
    for r in tr_rows:
        nid = r["note_id"]
        if nid not in per_note:
            per_note[nid] = TrainIndex(tr_rows, exclude_note=nid)
        lono_in += 1 if r["mesh"] in per_note[nid].freq else 0
    ev_in = sum(1 for r in ev_rows if r["mesh"] in tidx.freq)
    print(f"\nMANIPULATION CHECK  train gold-in-index {gold_in/len(tr_rows):.4f} "
          f"-> LONO {lono_in/len(tr_rows):.4f};  eval {ev_in/len(ev_rows):.4f}",
          flush=True)

    print("\nbuilding features...", flush=True)
    fcache = CACHE / "features.npz"
    if fcache.exists():
        z = np.load(fcache, allow_pickle=True)
        Xtr, Xlo, ytr = z["Xtr"], z["Xlo"], z["ytr"]
        Xev, yev = z["Xev"], z["yev"]
        ptr, pev = list(z["ptr"]), list(z["pev"])
        print(f"  loaded from cache: train {Xtr.shape}, eval {Xev.shape}", flush=True)
    else:
        Xtr, Xlo, ytr, ptr = build(tr_rows, tidx, nidx, "train", t0, lono_rows=tr_rows)
        Xev, _, yev, pev = build(ev_rows, tidx, nidx, "eval", t0)
        np.savez_compressed(fcache, Xtr=Xtr, Xlo=Xlo, ytr=ytr, Xev=Xev, yev=yev,
                            ptr=np.array(ptr, dtype=object),
                            pev=np.array(pev, dtype=object))
    assert Xtr.shape == Xlo.shape, f"pool must be identical: {Xtr.shape} vs {Xlo.shape}"
    changed = set(np.flatnonzero((Xtr != Xlo).any(axis=0)).tolist())
    assert changed <= {FNAMES.index(n) for n in LEAKY}, \
        f"LONO changed non-train-derived features: {[FNAMES[i] for i in changed]}"
    print(f"  LONO altered exactly these features: "
          f"{sorted(FNAMES[i] for i in changed)}", flush=True)

    reach = {}
    for i, r in enumerate(ev_rows):
        for s in scopes_of(r["mesh"], train_freq):
            reach.setdefault(s, [0, 0])
            reach[s][1] += 1
            reach[s][0] += r["mesh"] in pev[i]
    print("\npool reachability:", {k: round(v[0] / v[1], 4) for k, v in reach.items()},
          flush=True)

    from sklearn.experimental import enable_hist_gradient_boosting  # noqa
    from sklearn.ensemble import HistGradientBoostingClassifier
    all_cols = list(range(len(FNAMES)))
    hyg_cols = [i for i, n in enumerate(FNAMES) if n not in set(LEAKY)]

    rows_out = []
    # unranked baseline = pool order
    acc, n = accuracies({i: p[0] for i, p in enumerate(pev) if p}, ev_rows, train_freq)
    rows_out.append({"model": "pool order, no ranking", **acc})
    # oracle
    acc, _ = accuracies({i: (r["mesh"] if r["mesh"] in pev[i] else None)
                         for i, r in enumerate(ev_rows)}, ev_rows, train_freq)
    rows_out.append({"model": "oracle (pool ceiling)", **acc})

    for label, X, cols in (("supervised, standard", Xtr, all_cols),
                           ("supervised, features deleted", Xtr, hyg_cols),
                           ("supervised, leave-one-abstract-out", Xlo, all_cols)):
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                             max_depth=6, l2_regularization=1.0,
                                             random_state=SEED)
        clf.fit(X[:, cols], ytr)
        pr = clf.predict_proba(Xev[:, cols])[:, 1]
        acc, _ = accuracies(picks_from(pr, None, pev, ev_rows), ev_rows, train_freq)
        rows_out.append({"model": label, **acc})
        print(f"  {label}: all={acc['all']} head={acc['head']} tail={acc['tail']} "
              f"tail_out={acc['tail_out_of_index']} ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows_out)
    df.to_csv(OUT / "bc5cdr_blind_spot.csv", index=False)
    lines = [
        "# Replication of the in-index blind spot on BC5CDR", "",
        f"BioCreative V CDR · train {len(tr_rows):,} mentions / "
        f"{len({r['note_id'] for r in tr_rows})} abstracts · "
        f"test {len(ev_rows):,} mentions / {len({r['note_id'] for r in ev_rows})} "
        f"abstracts · concepts normalised to MeSH · candidate index built from the "
        f"CTD disease and chemical vocabularies ({nidx.n_names:,} surface names for "
        f"{len(names):,} concepts) · pool = weighted RRF@{POOL} over training prior, "
        "exact name, token and character retrieval. No dense retrieval and no type "
        "predictor: the claim under test concerns the supervision signal, not the "
        "strength of the retriever.", "",
        f"Strata by training frequency, as in the main study: head at twenty or "
        f"more, torso three to nineteen, tail below three, with the tail split by "
        f"whether the gold concept appears in the training annotations at all.", "",
        f"**Manipulation check.** Training gold-in-index is "
        f"{gold_in/len(tr_rows):.4f} by construction and falls to "
        f"{lono_in/len(tr_rows):.4f} under leave-one-abstract-out; the test set is "
        f"{ev_in/len(ev_rows):.4f}.", "",
        "Pool reachability by scope: "
        + ", ".join(f"{k} {v[0]/v[1]:.4f}" for k, v in sorted(reach.items())), "",
        md_table(df), "",
    ]
    (OUT / "BC5CDR_BLIND_SPOT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + df.to_string(index=False))
    print(f"\nwrote {OUT/'BC5CDR_BLIND_SPOT.md'} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
