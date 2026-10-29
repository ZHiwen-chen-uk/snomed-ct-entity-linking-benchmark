"""Dense biomedical retrieval for candidate generation (#1).

Lexical / char / synonym retrieval all match on surface strings, so they cannot
reach gold concepts phrased as a paraphrase of the mention (an illustrative
case: "kidney damage" -> "renal injury"). A biomedical bi-encoder trained on the SNOMED synonym
space (SapBERT) embeds mention and concept names into a shared space where
semantic neighbours are close, recovering exactly those misses.

Design
------
* Encoder : SapBERT ([CLS] embedding, L2-normalised) over strings. CPU is fine;
            the expensive part is the one-time SNOMED encode (see
            build_dense_index.py).
* Index   : every active SNOMED description form (from candidate_booster's
            syn_index) is a row in a float16 memmap matrix; a parallel list maps
            each row (form) to its concept ids. Retrieval scores forms by cosine
            and aggregates to concepts (max over the concept's forms).
* No FAISS: brute-force blocked matmul over ~665k rows is a few seconds for the
            600-mention eval and keeps the dependency surface minimal.
"""

import os
import pickle
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # Anaconda+torch OpenMP clash

HERE = Path(__file__).resolve().parent
SAPBERT = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
EMB_PATH = HERE / ".snomed_sapbert_emb.f16.npy"        # (N, 768) float16 memmap
META_PATH = HERE / ".snomed_sapbert_meta.pkl"          # forms, concepts, dims
PROGRESS_PATH = HERE / ".snomed_sapbert_progress.txt"  # rows encoded so far (resume)
DIM = 768


class SapBertEncoder:
    """Lazily-loaded SapBERT string encoder ([CLS], L2-normalised)."""

    def __init__(self, model_name=SAPBERT, device="cpu", max_length=32, batch_size=128):
        self.model_name = model_name
        self.device = device
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self._tok = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()

    def encode(self, texts, show_every=0):
        self._load()
        torch = self._torch
        out = np.empty((len(texts), DIM), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = [str(t) for t in texts[i : i + self.batch_size]]
                toks = self._tok(batch, padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt").to(self.device)
                cls = self._model(**toks).last_hidden_state[:, 0, :]   # SapBERT: [CLS]
                cls = torch.nn.functional.normalize(cls, p=2, dim=1)
                out[i : i + len(batch)] = cls.cpu().numpy()
                if show_every and (i // self.batch_size) % show_every == 0:
                    print(f"  encoded {i + len(batch)}/{len(texts)}", flush=True)
        return out


class DenseRetriever:
    """Query the pre-built SapBERT SNOMED index. Concept-level cosine retrieval."""

    def __init__(self, emb_path=EMB_PATH, meta_path=META_PATH, encoder=None):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        self.forms = meta["forms"]                 # list[str], length N
        self.form_concepts = meta["form_concepts"] # list[tuple[int]], length N
        self.n = int(meta["n"])
        self.emb = np.load(emb_path, mmap_mode="r") # (N, 768) float16
        assert self.emb.shape[0] == self.n, "emb/meta size mismatch"
        self.encoder = encoder or SapBertEncoder()

    def retrieve_batch(self, queries, k=30, top_forms=400, block=100_000):
        """queries: list[str]. Returns list of [(concept_id, score)] top-k per query."""
        Q = self.encoder.encode(queries)                    # (nq, 768) f32, normalised
        nq = Q.shape[0]
        M = int(top_forms)
        best_s = np.full((nq, M), -1e30, dtype=np.float32)
        best_i = np.zeros((nq, M), dtype=np.int64)
        for s in range(0, self.n, block):
            Eb = np.asarray(self.emb[s : s + block], dtype=np.float32)   # (b, 768)
            sc = Q @ Eb.T                                                # (nq, b)
            b = Eb.shape[0]
            idx = np.arange(s, s + b)[None, :].repeat(nq, 0)
            cs = np.concatenate([best_s, sc], axis=1)
            ci = np.concatenate([best_i, idx], axis=1)
            part = np.argpartition(-cs, M - 1, axis=1)[:, :M]
            best_s = np.take_along_axis(cs, part, axis=1)
            best_i = np.take_along_axis(ci, part, axis=1)
        results = []
        for r in range(nq):
            order = np.argsort(-best_s[r])
            concept_score = {}
            for j in order:
                row = int(best_i[r, j])
                sc = float(best_s[r, j])
                for cid in self.form_concepts[row]:
                    cid = int(cid)
                    if cid not in concept_score or sc > concept_score[cid]:
                        concept_score[cid] = sc
                if len(concept_score) >= k * 4:
                    break
            top = sorted(concept_score.items(), key=lambda kv: kv[1], reverse=True)[:k]
            results.append(top)
        return results


def build_index(syn_index, min_len=2, max_len=120, batch_size=128, step=20_000,
                emb_path=EMB_PATH, meta_path=META_PATH, progress_path=PROGRESS_PATH,
                encoder=None):
    """One-time encode of every SNOMED description form into a float16 memmap.

    syn_index: dict[str form -> tuple[int concepts]] (candidate_booster.syn_index).
    RESUMABLE: `forms` order is deterministic (dict order + deterministic filter),
    metadata is written up front, and a progress marker records how many rows are
    already encoded, so a killed run resumes instead of restarting.
    """
    forms, form_concepts = [], []
    for form, concepts in syn_index.items():
        txt = str(form).strip()
        if min_len <= len(txt) <= max_len:
            forms.append(txt)
            form_concepts.append(tuple(int(c) for c in concepts))
    n = len(forms)
    print(f"forms to encode: {n}", flush=True)

    # metadata first -> the partial index is usable/queryable immediately
    with open(meta_path, "wb") as f:
        pickle.dump({"forms": forms, "form_concepts": form_concepts,
                     "n": n, "dim": DIM, "model": SAPBERT}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)

    # open (resume) or create the memmap
    resume_from = 0
    if Path(emb_path).exists() and Path(progress_path).exists():
        try:
            existing = np.load(emb_path, mmap_mode="r")
            if existing.shape == (n, DIM):
                resume_from = int(Path(progress_path).read_text().strip() or 0)
            del existing
        except Exception:
            resume_from = 0
    mode = "r+" if resume_from > 0 else "w+"
    emb = np.lib.format.open_memmap(emb_path, mode=mode, dtype=np.float16, shape=(n, DIM))
    if resume_from:
        print(f"resuming from row {resume_from}", flush=True)

    enc = encoder or SapBertEncoder(batch_size=batch_size)
    for s in range(resume_from, n, step):
        chunk = forms[s : s + step]
        vecs = enc.encode(chunk)
        emb[s : s + len(chunk)] = vecs.astype(np.float16)
        emb.flush()
        Path(progress_path).write_text(str(s + len(chunk)))
        print(f"  wrote {s + len(chunk)}/{n}", flush=True)
    del emb
    print(f"saved index -> {emb_path} ({n} x {DIM} f16), meta -> {meta_path}", flush=True)
