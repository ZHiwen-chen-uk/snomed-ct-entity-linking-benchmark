"""Stage-0 span detection: fine-tune a token classifier and evaluate detection,
for BOTH variants (vanilla vs section_aware). Portable torch loop (transformers
4.30.2 friendly). Runs the full job on Colab GPU; a CPU smoke subset locally.

  # local smoke (reuses cached SapBERT, tiny subset, proves the harness):
  python run_ner.py --model cambridgeltl/SapBERT-from-PubMedBERT-fulltext \
      --variant both --epochs 1 --max_notes_train 12 --max_notes_test 8 \
      --max_len 256 --batch 4 --out smoke_out

  # Colab full (PubMedBERT, all notes):
  python run_ner.py --model microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
      --variant both --epochs 4 --batch 8 --out full_out

Exports predicted TEST spans per variant -> stage0_pred_spans_<variant>.jsonl,
which Stage-1/2 chaining consumes to compute end-to-end (predicted-span) accuracy
against the oracle-span (gold-span) numbers already produced.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ner_common import (LABELS, build_dataset, decode_window, evaluate_detection,
                        format_report, load_jsonl, merge_spans)

HERE = Path(__file__).resolve().parent


class WinDS(Dataset):
    def __init__(self, feats):
        self.f = feats

    def __len__(self):
        return len(self.f)

    def __getitem__(self, i):
        x = self.f[i]
        return {k: torch.tensor(x[k], dtype=torch.long)
                for k in ("input_ids", "attention_mask", "labels")}


def train_one(model, feats, device, epochs, batch, lr=3e-5):
    from torch.optim import AdamW
    dl = DataLoader(WinDS(feats), batch_size=batch, shuffle=True)
    opt = AdamW(model.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        tot = 0.0
        for step, b in enumerate(dl):
            b = {k: v.to(device) for k, v in b.items()}
            out = model(**b)
            out.loss.backward()
            opt.step(); opt.zero_grad()
            tot += out.loss.item()
            if step % 50 == 0:
                print(f"  epoch {ep+1} step {step}/{len(dl)} loss {out.loss.item():.4f}",
                      flush=True)
        print(f"  epoch {ep+1} mean loss {tot/max(1,len(dl)):.4f}", flush=True)


@torch.no_grad()
def predict_spans(model, feats, win_meta, records, device, batch):
    model.eval()
    dl = DataLoader(WinDS(feats), batch_size=batch, shuffle=False)
    preds = []
    for b in dl:
        ids = b["input_ids"].to(device); attn = b["attention_mask"].to(device)
        logits = model(input_ids=ids, attention_mask=attn).logits
        preds.append(logits.argmax(-1).cpu().numpy())
    preds = np.concatenate(preds, 0) if preds else np.zeros((0, 0))
    by_note = {}
    for wi, meta in enumerate(win_meta):
        spans = decode_window(preds[wi].tolist(), meta["offsets"])
        nid = records[meta["note_idx"]]["note_id"]
        by_note.setdefault(nid, []).extend(spans)
    return {nid: merge_spans(sp) for nid, sp in by_note.items()}


def run_variant(variant, tok, ModelCls, model_name, train_recs, test_recs, args, device):
    print(f"\n=== variant: {variant} ===", flush=True)
    tr_feats, _ = build_dataset(tok, train_recs, variant, max_len=args.max_len)
    te_feats, te_meta = build_dataset(tok, test_recs, variant, max_len=args.max_len)
    print(f"  windows: train {len(tr_feats)} / test {len(te_feats)}", flush=True)
    model = ModelCls.from_pretrained(model_name, num_labels=len(LABELS)).to(device)
    train_one(model, tr_feats, device, args.epochs, args.batch)
    pred_by_note = predict_spans(model, te_feats, te_meta, test_recs, device, args.batch)
    metrics = evaluate_detection(test_recs, pred_by_note)
    # export predicted spans for downstream chaining
    out = Path(args.out); out.mkdir(exist_ok=True)
    with open(out / f"stage0_pred_spans_{variant}.jsonl", "w", encoding="utf-8") as f:
        for rec in test_recs:
            f.write(json.dumps({"note_id": rec["note_id"],
                                "pred_spans": pred_by_note.get(rec["note_id"], [])}) + "\n")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    ap.add_argument("--variant", default="both",
                    choices=["vanilla", "section_aware", "both"])
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--max_notes_train", type=int, default=0)   # 0 = all
    ap.add_argument("--max_notes_test", type=int, default=0)
    ap.add_argument("--out", default="full_out")
    args = ap.parse_args()

    from transformers import AutoModelForTokenClassification, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} model={args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)

    train_recs = load_jsonl(HERE / "ner_train.jsonl")
    test_recs = load_jsonl(HERE / "ner_test.jsonl")
    if args.max_notes_train:
        train_recs = train_recs[:args.max_notes_train]
    if args.max_notes_test:
        test_recs = test_recs[:args.max_notes_test]
    print(f"notes: train {len(train_recs)} / test {len(test_recs)}", flush=True)

    variants = ["vanilla", "section_aware"] if args.variant == "both" else [args.variant]
    report = ["# Stage-0 span detection — results", "",
              f"model `{args.model}` · epochs {args.epochs} · max_len {args.max_len} · "
              f"train notes {len(train_recs)} / test notes {len(test_recs)}", ""]
    summary = {}
    for v in variants:
        m = run_variant(v, tok, AutoModelForTokenClassification, args.model,
                        train_recs, test_recs, args, device)
        summary[v] = m
        report.append(format_report(v, m))
        print(format_report(v, m), flush=True)

    if len(summary) == 2:
        a, b = summary["vanilla"], summary["section_aware"]
        report += ["### Δ section_aware − vanilla (the controlled A/B)", "",
                   f"- exact F1: {b['exact_F1']-a['exact_F1']:+.3f}  ·  "
                   f"IoU F1: {b['iou_F1']-a['iou_F1']:+.3f}",
                   "- tail recall (IoU): "
                   f"{b['recall_iou_by_stratum']['tail']-a['recall_iou_by_stratum']['tail']:+.3f}"
                   f"  ·  head recall (IoU): "
                   f"{b['recall_iou_by_stratum']['head']-a['recall_iou_by_stratum']['head']:+.3f}",
                   ""]
    out = Path(args.out); out.mkdir(exist_ok=True)
    (out / "STAGE0_RESULTS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (out / "stage0_metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDONE -> {out}/STAGE0_RESULTS.md", flush=True)


if __name__ == "__main__":
    main()
