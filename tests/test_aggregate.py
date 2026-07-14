from datetime import datetime, timezone

from marketsentiment.aggregation import build_aggregates, detect_hot_stocks
from marketsentiment.schemas import (
    ClassifiedPost,
    RawPost,
    Sentiment,
    SentimentResult,
    SourceType,
)


def _post(symbol: str, label: Sentiment) -> ClassifiedPost:
    return ClassifiedPost(
        post=RawPost(
            id=f"{symbol}-{label.value}-{id(label)}",
            source=SourceType.STOCKTWITS,
            text=f"${symbol} thoughts",
            created_at=datetime.now(timezone.utc),
        ),
        tickers=[symbol],
        sentiment=SentimentResult(label=label, confidence=0.9, model="test"),
    )


def _classified() -> list[ClassifiedPost]:
    return [
        _post("GME", Sentiment.BULLISH),
        _post("GME", Sentiment.BULLISH),
        _post("GME", Sentiment.BULLISH),
        _post("GME", Sentiment.BEARISH),
        _post("AAPL", Sentiment.NEUTRAL),
    ]


def test_build_aggregates_counts_and_score():
    aggs = {a.symbol: a for a in build_aggregates(_classified())}

    gme = aggs["GME"]
    assert gme.n_mentions == 4
    assert (gme.bullish, gme.bearish, gme.neutral) == (3, 1, 0)
    # (3 - 1) / 4
    assert gme.sentiment_score == 0.5

    assert aggs["AAPL"].n_mentions == 1
    assert aggs["AAPL"].sentiment_score == 0.0


def test_detect_hot_respects_min_mentions():
    aggs = build_aggregates(_classified())
    hot = detect_hot_stocks(aggs, min_mentions=2, top_n=10)
    symbols = [h.symbol for h in hot]
    assert symbols == ["GME"]  # AAPL has only 1 mention, filtered out


def test_detect_hot_computes_velocity_from_previous_counts():
    aggs = build_aggregates(_classified())
    hot = detect_hot_stocks(aggs, min_mentions=2, previous_counts={"GME": 2})
    gme = next(h for h in hot if h.symbol == "GME")
    # Volume went 2 -> 4, i.e. +100%.
    assert gme.mention_velocity == 1.0
