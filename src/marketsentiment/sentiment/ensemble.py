"""Weighted soft-vote ensemble of FinBERT + LLM.

Runs both classifiers on every post, turns each (label, confidence) into a 3-class score
vector, blends them with ``finbert_weight``, and takes the argmax.

Caveat worth understanding: weighting toward FinBERT keeps FinBERT's answer on
disagreements — which only helps if FinBERT is accurate on your data. While FinBERT is
out-of-domain (trained on financial news, serving on social slang/sarcasm), lean the
weight toward the LLM, or retrain FinBERT on StockTwits first. Also note this runs BOTH
models on all posts, so it's slower + costlier than either alone.
"""

from __future__ import annotations

from marketsentiment.schemas import Sentiment, SentimentResult
from marketsentiment.sentiment.base import SentimentClassifier

_LABELS = ("bearish", "bullish", "neutral")


def _to_vector(res: SentimentResult) -> dict[str, float]:
    """Turn a (label, confidence) result into pseudo-probabilities over the 3 classes."""
    other = (1.0 - res.confidence) / 2.0
    vec = {label: other for label in _LABELS}
    vec[res.label.value] = res.confidence
    return vec


class EnsembleClassifier:
    def __init__(
        self,
        finbert: SentimentClassifier,
        llm: SentimentClassifier,
        finbert_weight: float = 0.6,
    ):
        if not 0.0 <= finbert_weight <= 1.0:
            raise ValueError("finbert_weight must be in [0, 1]")
        self._finbert = finbert
        self._llm = llm
        self._w = finbert_weight

    def classify(self, texts: list[str]) -> list[SentimentResult]:
        if not texts:
            return []
        fb_results = self._finbert.classify(texts)
        llm_results = self._llm.classify(texts)

        out: list[SentimentResult] = []
        for fb, lm in zip(fb_results, llm_results):
            fv, lv = _to_vector(fb), _to_vector(lm)
            blended = {
                label: self._w * fv[label] + (1.0 - self._w) * lv[label] for label in _LABELS
            }
            label = max(blended, key=blended.get)
            out.append(
                SentimentResult(
                    label=Sentiment(label),
                    confidence=round(blended[label], 4),
                    model=f"ensemble(finbert_w={self._w})",
                )
            )
        return out
