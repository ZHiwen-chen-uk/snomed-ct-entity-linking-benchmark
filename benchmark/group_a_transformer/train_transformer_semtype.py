"""Train a transformer semantic-type classifier for Group A.

The model predicts the coarse SNOMED semantic type of a mention from section,
span, and context. The downstream goal is not just raw accuracy: the validation
report also measures high-confidence precision and coverage, because Group A
only helps when high-confidence predictions are safe enough to filter.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = THIS_DIR / "data"
DEFAULT_OUT = THIS_DIR / "outputs" / "pubmedbert_semtype"


class SemTypeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, label2id: Dict[str, int], max_length: int):
        self.texts = df["transformer_text"].astype(str).tolist()
        self.labels = [label2id[str(x)] for x in df["label"].tolist()]
        self.enc = tokenizer(
            self.texts,
            truncation=True,
            padding=False,
            max_length=max_length,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss = torch.nn.functional.cross_entropy(logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


def high_conf_table(probs: np.ndarray, preds: np.ndarray, labels: np.ndarray):
    rows = []
    conf = probs.max(axis=1)
    correct = preds == labels
    for threshold in [0.50, 0.55, 0.60, 0.70, 0.80, 0.90]:
        idx = conf >= threshold
        rows.append({
            "threshold": threshold,
            "coverage": float(idx.mean()) if len(idx) else 0.0,
            "n": int(idx.sum()),
            "precision_when_fired": float(correct[idx].mean()) if idx.any() else math.nan,
        })
    return rows


def compute_class_weights(labels: List[int], n_labels: int):
    counts = np.bincount(np.array(labels), minlength=n_labels)
    total = counts.sum()
    weights = np.zeros(n_labels, dtype=np.float32)
    for i, count in enumerate(counts):
        weights[i] = total / (n_labels * count) if count else 0.0
    return torch.tensor(weights, dtype=torch.float32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(DEFAULT_DATA))
    p.add_argument("--output-dir", default=str(DEFAULT_OUT))
    p.add_argument("--model-name", default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument("--seed", type=int, default=20260720)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--no-class-weights", action="store_true")
    p.add_argument("--save-strategy", default="epoch", choices=["epoch", "steps"])
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--eval-steps", type=int, default=None)
    p.add_argument("--logging-steps", type=int, default=50)
    p.add_argument("--save-total-limit", type=int, default=2)
    p.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Checkpoint path, or 'auto' to resume from the latest checkpoint in output-dir.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_map = json.loads((data_dir / "label_map.json").read_text(encoding="utf-8"))
    labels = label_map["labels"]
    label2id = {str(k): int(v) for k, v in label_map["label2id"].items()}
    id2label = {int(v): str(k) for k, v in label2id.items()}

    train_df = pd.read_csv(data_dir / "semtype_train.csv")
    val_df = pd.read_csv(data_dir / "semtype_val.csv")
    train_df = train_df[train_df["label"].isin(label2id)].copy()
    val_df = val_df[val_df["label"].isin(label2id)].copy()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
    )

    train_ds = SemTypeDataset(train_df, tokenizer, label2id, args.max_length)
    val_ds = SemTypeDataset(val_df, tokenizer, label2id, args.max_length)
    class_weights = None if args.no_class_weights else compute_class_weights(train_ds.labels, len(labels))

    def compute_metrics(eval_pred):
        logits, y_true = eval_pred
        y_pred = np.argmax(logits, axis=1)
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }

    eval_steps = args.eval_steps if args.eval_steps is not None else args.save_steps
    training_kwargs = {
        "output_dir": str(output_dir),
        "save_strategy": args.save_strategy,
        "save_steps": args.save_steps,
        "eval_steps": eval_steps,
        "save_total_limit": args.save_total_limit,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_macro_f1",
        "greater_is_better": True,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size * 2,
        "num_train_epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "fp16": args.fp16,
        "report_to": "none",
        "seed": args.seed,
    }
    strategy_arg = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters else "evaluation_strategy"
    training_kwargs[strategy_arg] = args.save_strategy
    training_args = TrainingArguments(**training_kwargs)

    trainer_kwargs = {
        "class_weights": class_weights,
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "compute_metrics": compute_metrics,
    }
    trainer_tokenizer_arg = "tokenizer" if "tokenizer" in inspect.signature(Trainer.__init__).parameters else "processing_class"
    trainer_kwargs[trainer_tokenizer_arg] = tokenizer
    trainer = WeightedTrainer(**trainer_kwargs)

    resume_checkpoint = None
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint.lower() == "auto":
            resume_checkpoint = get_last_checkpoint(str(output_dir)) if output_dir.exists() else None
        else:
            resume_checkpoint = args.resume_from_checkpoint
    if resume_checkpoint:
        print(f"resuming from checkpoint -> {resume_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(output_dir / "best_model")
    tokenizer.save_pretrained(output_dir / "best_model")
    (output_dir / "best_model" / "label_map.json").write_text(json.dumps(label_map, indent=2), encoding="utf-8")

    pred = trainer.predict(val_ds)
    logits = pred.predictions
    probs = softmax(logits)
    y_true = np.array(val_ds.labels)
    y_pred = probs.argmax(axis=1)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(labels))), zero_division=0
    )
    pd.DataFrame([
        {
            "label": id2label[i],
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(labels))
    ]).to_csv(output_dir / "validation_by_label.csv", index=False)
    pd.DataFrame(high_conf_table(probs, y_pred, y_true)).to_csv(output_dir / "validation_high_confidence.csv", index=False)
    (output_dir / "validation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"saved -> {output_dir / 'best_model'}")


if __name__ == "__main__":
    main()
