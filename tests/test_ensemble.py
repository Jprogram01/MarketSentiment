from marketsentiment.schemas import Sentiment, SentimentResult
from marketsentiment.sentiment.ensemble import EnsembleClassifier


class _Fake:
    """A stand-in classifier that always returns the same (label, confidence)."""

    def __init__(self, label: Sentiment, confidence: float):
        self._label, self._confidence = label, confidence

    def classify(self, texts):
        return [
            SentimentResult(label=self._label, confidence=self._confidence, model="fake")
            for _ in texts
        ]


def test_weight_toward_finbert_keeps_finberts_answer_on_disagreement():
    # This is the GRRR case: FinBERT bullish, LLM bearish. Weighted 0.6 toward FinBERT,
    # FinBERT still wins — i.e. weighting toward a wrong FinBERT does NOT fix it.
    fb = _Fake(Sentiment.BULLISH, 0.9)
    llm = _Fake(Sentiment.BEARISH, 0.8)
    out = EnsembleClassifier(fb, llm, finbert_weight=0.6).classify(["x"])
    assert out[0].label == Sentiment.BULLISH


def test_weight_toward_llm_flips_the_call():
    fb = _Fake(Sentiment.BULLISH, 0.9)
    llm = _Fake(Sentiment.BEARISH, 0.8)
    out = EnsembleClassifier(fb, llm, finbert_weight=0.3).classify(["x"])
    assert out[0].label == Sentiment.BEARISH


def test_agreement_is_preserved():
    fb = _Fake(Sentiment.BEARISH, 0.7)
    llm = _Fake(Sentiment.BEARISH, 0.6)
    out = EnsembleClassifier(fb, llm, finbert_weight=0.5).classify(["a", "b"])
    assert [r.label for r in out] == [Sentiment.BEARISH, Sentiment.BEARISH]


def test_weight_must_be_valid():
    import pytest

    with pytest.raises(ValueError):
        EnsembleClassifier(_Fake(Sentiment.NEUTRAL, 0.5), _Fake(Sentiment.NEUTRAL, 0.5), 1.5)
