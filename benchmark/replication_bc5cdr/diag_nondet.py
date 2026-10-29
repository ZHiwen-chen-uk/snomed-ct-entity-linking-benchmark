import hashlib, replicate_blind_spot as R
sp = R.load_corpus(); tr = sp["train"][:300]
names = R.load_ctd()
h = lambda s: hashlib.md5(s.encode()).hexdigest()[:12]
print("ctd      ", h("".join(k+"|".join(v) for k,v in names.items())))
nidx = R.NameIndex(names); tidx = R.TrainIndex(sp["train"])
print("tri_n    ", h("".join(f"{k}{v}" for k,v in sorted(nidx.tri_n.items()))))
ex_s=lex_s=ch_s=gen_s=pool_s=""
for r in tr:
    ex, lt, ct = nidx.retrieve(r["span"], 60)
    ex_s += "".join(ex)
    lex_s += "".join(f"{c}{v:.17g}" for c,v in lt)
    ch_s  += "".join(f"{c}{v:.17g}" for c,v in ct)
    g = tidx.generate(r["span"], 60); gen_s += "".join(g)
    pool_s += "".join(R.rrf({"train":g,"exact":ex,"lexical":[c for c,_ in lt],
                             "char":[c for c,_ in ct]}, weights=R.RRF_WEIGHTS))
for n,s in (("exact",ex_s),("lexical",lex_s),("char",ch_s),("train_gen",gen_s),("pool",pool_s)):
    print(f"{n:9s}", h(s))
