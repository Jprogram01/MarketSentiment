"""FinBERT sentiment classifier (the ML workhorse).

Torch/transformers are heavy, so they're imported lazily and gated behind the
``[finbert]`` extra. ProsusAI/finbert emits positive/negative/neutral, which we map
onto bullish/bearish/neutral. Fine-tune it on StockTwits' Bullish/Bearish labels
(scripts/train_finbert.py) to adapt it to social chatter.
"""

from __future__ import annotations

from marketsentiment.observability.logging import get_logger
from marketsentiment.schemas import Sentiment, SentimentResult

log = get_logger(__name__)

_LABEL_MAP = {
    "positive": Sentiment.BULLISH,
    "negative": Sentiment.BEARISH,
    "neutral": Sentiment.NEUTRAL,
    # Some fine-tuned checkpoints emit the target labels directly.
    "bullish": Sentiment.BULLISH,
    "bearish": Sentiment.BEARISH,
}


class FinBERTClassifier:
    def __init__(self, model_name: str = "ProsusAI/finbert", device: int | None = None):
        # device=None auto-selects GPU when available; pass an int to force (0=cuda:0, -1=cpu).
        self._model_name = model_name
        self._device = device
        self._pipe = None  # lazily constructed

    def _ensure_pipe(self):
        if self._pipe is not None:
            return
        try:
            from transformers import pipeline  # noqa: PLC0415 - heavy, lazy import
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "FinBERT backend needs torch + transformers. "
                'Install with: pip install -e ".[finbert]"'
            ) from exc

        device = self._device
        if device is None:
            try:
                import torch  # noqa: PLC0415

                device = 0 if torch.cuda.is_available() else -1
            except ImportError:  # pragma: no cover
                device = -1

        log.info("finbert.loading", model=self._model_name, device=device)
        self._pipe = pipeline(
            "text-classification",
            model=self._model_name,
            device=device,
            truncation=True,
        )

    def classify(self, texts: list[str]) -> list[SentimentResult]:
        if not texts:
            return []
        self._ensure_pipe()
        raw = self._pipe(texts)  # type: ignore[misc]
        results: list[SentimentResult] = []
        for item in raw:
            label = _LABEL_MAP.get(str(item["label"]).lower(), Sentiment.NEUTRAL)
            results.append(
                SentimentResult(
                    label=label,
                    confidence=float(item["score"]),
                    model=self._model_name,
                )
            )
        return results
