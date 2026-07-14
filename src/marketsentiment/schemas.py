"""Pydantic data models — the shared contract every stage of the pipeline speaks."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    STOCKTWITS = "stocktwits"
    REDDIT = "reddit"
    FOURCHAN = "fourchan"
    BLUESKY = "bluesky"


class Sentiment(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class RawPost(BaseModel):
    """A single normalized message from any source."""

    id: str
    source: SourceType
    text: str
    created_at: datetime
    author: str | None = None
    url: str | None = None
    # Cashtags the platform itself attached (e.g. StockTwits `symbols`).
    symbols: list[str] = Field(default_factory=list)
    # Ground-truth label where the platform provides one (StockTwits Bullish/Bearish).
    # This is the free training/eval signal — never overwrite it with a model guess.
    gold_label: Sentiment | None = None


class SentimentResult(BaseModel):
    label: Sentiment
    confidence: float
    model: str


class ClassifiedPost(BaseModel):
    post: RawPost
    tickers: list[str]
    sentiment: SentimentResult


class TickerAggregate(BaseModel):
    symbol: str
    n_mentions: int
    bullish: int
    bearish: int
    neutral: int
    # (bullish - bearish) / n_mentions, in [-1, 1].
    sentiment_score: float
    sources: list[SourceType] = Field(default_factory=list)


class HotStock(BaseModel):
    symbol: str
    n_mentions: int
    sentiment_score: float
    # How fast mention volume is moving vs. the previous run (None on first run).
    mention_velocity: float | None = None
    reason: str
