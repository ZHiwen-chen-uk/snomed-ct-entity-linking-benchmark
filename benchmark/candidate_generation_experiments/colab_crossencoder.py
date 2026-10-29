"""#2 Colab: fine-tune a PubMedBERT cross-encoder reranker + full-23k inference.

Consumes the two CSVs exported locally by stage2_export_crossencoder_data.py
(ce_mentions.csv, ce_pairs.csv), no SNOMED/dense needed on Colab, just HF.

Cross-encoder = sequence-pair classifier scoring (mention+context) vs (candidate
SNOMED term). Leakage-free (text vs text, no prior). Reranks the RRF@60 pool;
top-1 = argmax P(relevant). Reports stratified accuracy on the eval note split.

    pip install "transformers==4.30.2" datasets accelerate
    python colab_crossencoder.py --data-dir . --out-dir /content/drive/MyDrive/ce_out
"""
import argparse
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"  # PubMedBERT
MAX_LEN = 256
NEG_PER_MENTION = 24     # train: gold + up to N sampled hard negatives
SEED = 20260724


def load_pairs(data_dir):
    men = pd.read_csv(Path(data_dir) / "ce_mentions.csv")
    pairs = pd.read_csv(Path(data_dir) / "ce_pairs.csv")
    text_by_key = dict(zip(men.key, men.mention_text))
    pairs["mention_text"] = pairs.key.map(text_by_key)
    return men, pairs


def build_train_examples(men, pairs):
    train_keys = set(men[men.split == "train"].key)
    tp = pairs[pairs.key.isin(train_keys)]
    rng = random.Random(SEED)
    rows = []
    for key, grp in tp.groupby("key"):
        pos = grp[grp.label == 1]
        neg = grp[grp.label == 0]
        if len(pos) == 0:
            continue                      # gold not in pool -> no training signal
        keep_neg = neg if len(neg) <= NEG_PER_MENTION else neg.sample(NEG_PER_MENTION, random_state=SEED)
        for r in pd.concat([pos, keep_neg]).itertuples(index=False):
            rows.append({"text_a": r.mention_text, "text_b": r.cand_term, "label": int(r.label)})
    rng.shuffle(rows)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--out-dir", default="ce_out")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    import torch
    from torch.utils.data import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              Trainer, TrainingArguments)

    men, pairs = load_pairs(args.data_dir)
    train_df = build_train_examples(men, pairs)
    pos = int(train_df.label.sum()); neg = len(train_df) - pos
    print(f"train examples={len(train_df)} pos={pos} neg={neg}")

    tok = AutoTokenizer.from_pretrained(MODEL)

    class PairDS(Dataset):
        def __init__(self, a, b, y=None):
            self.a, self.b, self.y = list(a), list(b), (list(y) if y is not None else None)
        def __len__(self): return len(self.a)
        def __getitem__(self, i):
            enc = tok(self.a[i], self.b[i], truncation=True, max_length=MAX_LEN, padding="max_length")
            item = {k: torch.tensor(v) for k, v in enc.items()}
            if self.y is not None:
                item["labels"] = torch.tensor(self.y[i])
            return item

    train_ds = PairDS(train_df.text_a, train_df.text_b, train_df.label)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)

    # class-weighted loss for residual imbalance
    cw = torch.tensor([1.0, max(1.0, neg / max(1, pos))], dtype=torch.float)

    class WTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            out = model(**inputs)
            loss = torch.nn.functional.cross_entropy(out.logits, labels, weight=cw.to(out.logits.device))
            return (loss, out) if return_outputs else loss

    targs = TrainingArguments(
        output_dir=str(out / "ckpt"), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch, learning_rate=args.lr,
        warmup_ratio=0.1, weight_decay=0.01, logging_steps=100,
        save_strategy="no", report_to=[], fp16=torch.cuda.is_available())
    WTrainer(model=model, args=targs, train_dataset=train_ds).train()

    # inference over ALL test pairs
    test_keys = set(men[men.split == "test"].key)
    tp = pairs[pairs.key.isin(test_keys)].reset_index(drop=True)
    infer_ds = PairDS(tp.mention_text, tp.cand_term)
    model.eval()
    scores = np.zeros(len(tp), dtype=np.float32)
    from torch.utils.data import DataLoader
    dl = DataLoader(infer_ds, batch_size=128)
    with torch.no_grad():
        idx = 0
        for batch in dl:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            p = torch.softmax(model(**batch).logits, dim=-1)[:, 1].cpu().numpy()
            scores[idx:idx + len(p)] = p; idx += len(p)
    tp["score"] = scores

    # argmax per mention -> predicted cand_id
    pred = {}
    for key, grp in tp.groupby("key"):
        j = grp.score.values.argmax()
        pred[key] = int(grp.cand_id.values[j])

    meta = men[men.split == "test"].set_index("key")
    def acc(sub_keys):
        sub = [k for k in sub_keys if k in pred]
        if not sub: return None
        return round(np.mean([pred[k] == int(meta.loc[k, "gold"]) for k in sub]), 4)
    ev = meta[meta.val_or_eval == "eval"]
    rows = [{"stratum": "all", "n": len(ev), "acc": acc(list(ev.index))}]
    for s in ("head", "torso", "tail"):
        keys = list(ev[ev.freq_stratum == s].index)
        rows.append({"stratum": s, "n": len(keys), "acc": acc(keys)})
    res = pd.DataFrame(rows)
    res.to_csv(out / "crossencoder_accuracy.csv", index=False)
    (out / "CROSSENCODER_RESULTS.md").write_text(
        "# #2 Cross-encoder reranker (PubMedBERT), full-23k eval split\n\n"
        + res.to_markdown(index=False) + "\n", encoding="utf-8")
    print(res.to_string(index=False))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
