"""Sentiment backend interface — swap FinBERT and the LLM baseline behind one type.

Keeping this a Protocol is deliberate: the interview story is "fine-tuned FinBERT vs.
LLM zero-shot — same interface, compared on cost/latency/accuracy."
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from marketsentiment.schemas import SentimentResult


@runtime_checkable
class SentimentClassifier(Protocol):
    def classify(self, texts: list[str]) -> list[SentimentResult]:
        """Classify a batch of texts, returning one result per input, in order."""
        ...
