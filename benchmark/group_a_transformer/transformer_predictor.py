"""Pipeline-compatible transformer semantic-type predictor.

After training, this can be used to plug the transformer model into Group A.
It follows the same contract as `SemTypePredictor.predict(mention)`:

    return semantic_type, confidence
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def mark_context(context: str, span: str) -> str:
    return context.replace(span, f"[MENTION] {span} [/MENTION]", 1) if span else context


def transformer_text(mention) -> str:
    return (
        f"Section: {mention.section or 'UNKNOWN'}\n"
        f"Mention: {mention.span}\n"
        f"Context: {mark_context(mention.context, mention.span)}"
    )


class TransformerSemTypePredictor:
    def __init__(self, model_dir, max_length=256, device=None):
        model_dir = Path(model_dir)
        label_map = json.loads((model_dir / "label_map.json").read_text(encoding="utf-8"))
        self.labels = list(label_map["labels"])
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.max_length = max_length
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.model.eval()
        self.calls = 0

    def predict(self, mention):
        text = transformer_text(mention)
        enc = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.model(**enc).logits[0]
            probs = torch.softmax(logits, dim=-1)
        idx = int(torch.argmax(probs).item())
        self.calls += 1
        return self.labels[idx], float(probs[idx].item())
