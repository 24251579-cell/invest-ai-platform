"""Steps 3-4: tokenization (Lmax=128) and classification with FinBERT / RuBERT."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from invest_ai import config

log = logging.getLogger(__name__)

LABELS = ("positive", "negative", "neutral")


@dataclass
class SentimentResult:
    label: str
    p_pos: float
    p_neg: float
    p_neu: float

    @property
    def score(self) -> float:
        """Formula (2.1): s_i = p_pos - p_neg."""
        return self.p_pos - self.p_neg


class SentimentModel:
    def __init__(self, checkpoint: str, max_length: int | None = None, batch_size: int | None = None):
        cfg = config.settings()["sentiment"]
        self.checkpoint = checkpoint
        self.max_length = max_length or cfg["max_length"]
        self.batch_size = batch_size or cfg["batch_size"]
        torch.set_num_threads(config.torch_threads())
        models_dir = os.environ.get("LOCAL_MODELS_DIR") or cfg["local_models_dir"]
        local = Path(models_dir) / checkpoint.split("/")[-1]
        source = str(local) if (local / "model.safetensors").exists() else checkpoint
        log.info("Loading %s from %s", checkpoint, source)
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.model = AutoModelForSequenceClassification.from_pretrained(source)
        self.model.eval()
        # Class order differs between checkpoints, so it is taken from id2label.
        id2label = {int(i): name.lower() for i, name in self.model.config.id2label.items()}
        unknown = set(id2label.values()) - set(LABELS)
        if unknown:
            raise ValueError(f"{checkpoint}: unexpected labels {unknown}")
        self.index = {name: i for i, name in id2label.items()}

    @torch.no_grad()
    def predict(self, texts: list[str]) -> list[SentimentResult]:
        results: list[SentimentResult] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            enc = self.tokenizer(batch, truncation=True, max_length=self.max_length,
                                 padding=True, return_tensors="pt")
            probs = torch.softmax(self.model(**enc).logits, dim=-1)
            for row in probs.tolist():
                p = {name: row[i] for name, i in self.index.items()}
                results.append(SentimentResult(label=max(p, key=p.get), p_pos=p["positive"],
                                               p_neg=p["negative"], p_neu=p["neutral"]))
        return results
