"""Colab: rank the tail candidate pool with an LLM and compare against the
leakage-free simple signals. Tests whether world knowledge breaks the
data-limited tail ceiling (target: gated 0.546 on the top-20 pool).

Backends:
  hf        local GPU model via transformers (default Qwen2.5-7B-Instruct, 4-bit)
  openai    OpenAI API (e.g. gpt-4o, gpt-4o-mini), needs --api-key or OPENAI_API_KEY
  anthropic Anthropic API (e.g. claude-sonnet-4-5), needs --api-key or ANTHROPIC_API_KEY

Consumes the CSVs from stage2_export_llm_tail.py; no SNOMED/dense/caches on Colab.

    python colab_llm_tail_ranker.py --backend hf --model Qwen/Qwen2.5-7B-Instruct --limit 800
    python colab_llm_tail_ranker.py --backend openai --model gpt-4o-mini --api-key sk-...
    python colab_llm_tail_ranker.py --backend anthropic --model claude-sonnet-4-5 --api-key sk-ant-...
"""
import argparse
import os
import re
import time
from pathlib import Path

import pandas as pd

SYSTEM = ("You are a clinical terminologist performing SNOMED CT entity linking. "
          "You are shown a clinical mention in context and a numbered list of candidate "
          "SNOMED CT concepts. Choose the single candidate that best matches the mention "
          "in context. Reply with ONLY the number of the best candidate, nothing else.")


def build_prompt(mention_text, terms):
    lines = [f"{i+1}. {t}" for i, t in enumerate(terms)]
    return (f"{mention_text}\n\nCandidates:\n" + "\n".join(lines) +
            f"\n\nReply with only the number (1-{len(terms)}) of the best-matching candidate.")


def parse_idx(text, k):
    m = re.search(r"\d+", text or "")
    idx = (int(m.group()) - 1) if m else 0
    return idx if 0 <= idx < k else 0


def make_hf(model_name, four_bit=True):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(model_name)
    kw = {}
    if four_bit:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
    else:
        kw["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", **kw).eval()

    def pick(prompt, terms):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**inp, max_new_tokens=8, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        ans = tok.decode(gen[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
        return parse_idx(ans, len(terms))
    return pick


def make_openai(model_name, api_key):
    from openai import OpenAI
    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def pick(prompt, terms):
        r = client.chat.completions.create(model=model_name, max_tokens=8, temperature=0,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])
        return parse_idx(r.choices[0].message.content, len(terms))
    return pick


def make_anthropic(model_name, api_key):
    import anthropic
    client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def pick(prompt, terms):
        r = client.messages.create(model=model_name, max_tokens=8, system=SYSTEM,
            messages=[{"role": "user", "content": prompt}])
        return parse_idx(r.content[0].text, len(terms))
    return pick


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--out-dir", default="llm_tail_out")
    ap.add_argument("--backend", choices=["hf", "openai", "anthropic"], default="hf")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--limit", type=int, default=800)
    args = ap.parse_args()
    d = Path(args.data_dir); out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    men = pd.read_csv(d / "llm_tail_mentions.csv")
    cand = pd.read_csv(d / "llm_tail_candidates.csv")
    if args.limit:
        men = men.head(args.limit).copy()
    cand = cand[cand.key.isin(set(men.key))]
    by_key = {k: g.sort_values("rank") for k, g in cand.groupby("key")}

    print(f"backend={args.backend} model={args.model} n={len(men)}", flush=True)
    if args.backend == "hf":
        pick = make_hf(args.model)
    elif args.backend == "openai":
        pick = make_openai(args.model, args.api_key)
    else:
        pick = make_anthropic(args.model, args.api_key)

    llm_pick = {}
    for n, (_, row) in enumerate(men.iterrows()):
        g = by_key.get(row.key)
        if g is None or len(g) == 0:
            llm_pick[row.key] = -1; continue
        ids, terms = list(g.cand_id), list(g.cand_term)
        try:
            idx = pick(build_prompt(row.mention_text, terms), terms)
        except Exception as e:
            print("  err:", str(e)[:120]); idx = 0
        llm_pick[row.key] = ids[idx]
        if (n + 1) % 50 == 0:
            print(f"  {n+1}/{len(men)} ({time.time()-t0:.0f}s)", flush=True)

    men["llm_pick"] = men.key.map(llm_pick)
    n = len(men)
    def acc(col): return round((men[col] == men["gold"]).mean(), 4)
    rows = [("oracle (gold in pool)", round(men["oracle_ok"].mean(), 4)),
            ("dense", acc("dense_pick")), ("type", acc("type_pick")),
            ("gated", acc("gated_pick")), (f"LLM ({args.model.split('/')[-1]})", acc("llm_pick"))]
    print(f"\n=== tail ranking, n={n} (RRF top-20 pool) ===")
    for name, a in rows:
        print(f"  {name:34} {a:.4f}")
    men.to_csv(out / "llm_tail_predictions.csv", index=False)
    with open(out / "LLM_TAIL_RESULTS.md", "w", encoding="utf-8") as f:
        f.write(f"# LLM tail ranking — {args.model}, n={n} (RRF top-20 pool)\n\n")
        f.write("| ranker | tail accuracy |\n|---|---:|\n")
        for name, a in rows:
            f.write(f"| {name} | {a:.4f} |\n")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
