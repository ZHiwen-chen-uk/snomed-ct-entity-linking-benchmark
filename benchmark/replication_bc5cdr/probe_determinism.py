import hashlib, numpy as np, replicate_blind_spot as R
sp = R.load_corpus(); tr = sp["train"][:600]
nidx = R.NameIndex(R.load_ctd()); tidx = R.TrainIndex(sp["train"])
X, XL, y, pools = R.build(tr, tidx, nidx, "probe", 0.0, lono_rows=sp["train"])
h = lambda a: hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]
print("X=%s XL=%s y=%s pools=%s" % (h(X), h(XL), h(y),
      hashlib.md5(("".join("".join(p) for p in pools)).encode()).hexdigest()[:16]))
