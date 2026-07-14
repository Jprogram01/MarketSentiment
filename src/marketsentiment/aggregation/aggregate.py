"""Per-ticker aggregation and "hot stock" detection.

sentiment_score = (bullish - bearish) / n_mentions, in [-1, 1].
A ticker is "hot" when it clears a mention floor and ranks high on a blend of
volume and |sentiment_score|. If prior-run counts are supplied, mention_velocity
(relative change in volume) feeds the ranking — spikes are the interesting signal.
"""

from __future__ import annotations

from collections import defaultdict

from marketsentiment.schemas import ClassifiedPost, HotStock, TickerAggregate


def build_aggregates(classified: list[ClassifiedPost]) -> list[TickerAggregate]:
    counts: dict[str, dict] = defaultdict(
        lambda: {"bullish": 0, "bearish": 0, "neutral": 0, "sources": set()}
    )
    for cp in classified:
        for ticker in cp.tickers:
            bucket = counts[ticker]
            bucket[cp.sentiment.label.value] += 1
            bucket["sources"].add(cp.post.source)

    aggregates: list[TickerAggregate] = []
    for symbol, b in counts.items():
        n = b["bullish"] + b["bearish"] + b["neutral"]
        score = (b["bullish"] - b["bearish"]) / n if n else 0.0
        aggregates.append(
            TickerAggregate(
                symbol=symbol,
                n_mentions=n,
                bullish=b["bullish"],
                bearish=b["bearish"],
                neutral=b["neutral"],
                sentiment_score=round(score, 4),
                sources=sorted(b["sources"], key=lambda s: s.value),
            )
        )
    aggregates.sort(key=lambda a: a.n_mentions, reverse=True)
    return aggregates


def detect_hot_stocks(
    aggregates: list[TickerAggregate],
    min_mentions: int = 5,
    top_n: int = 10,
    previous_counts: dict[str, int] | None = None,
) -> list[HotStock]:
    previous_counts = previous_counts or {}
    scored: list[tuple[float, HotStock]] = []

    for agg in aggregates:
        if agg.n_mentions < min_mentions:
            continue

        prev = previous_counts.get(agg.symbol)
        velocity = None
        if prev:
            velocity = (agg.n_mentions - prev) / prev

        # Rank on volume + conviction, boosted by a positive volume spike.
        rank = agg.n_mentions * (1 + abs(agg.sentiment_score))
        if velocity and velocity > 0:
            rank *= 1 + velocity

        reason = _reason(agg, velocity)
        scored.append(
            (
                rank,
                HotStock(
                    symbol=agg.symbol,
                    n_mentions=agg.n_mentions,
                    sentiment_score=agg.sentiment_score,
                    mention_velocity=round(velocity, 3) if velocity is not None else None,
                    reason=reason,
                ),
            )
        )

    scored.sort(key=lambda x: x[0], reverse=True)
    return [hs for _, hs in scored[:top_n]]


def _reason(agg: TickerAggregate, velocity: float | None) -> str:
    tone = (
        "bullish"
        if agg.sentiment_score > 0.15
        else "bearish"
        if agg.sentiment_score < -0.15
        else "mixed"
    )
    parts = [f"{tone} tone", f"{agg.n_mentions} mentions"]
    if velocity is not None and velocity > 0.5:
        parts.append(f"+{velocity:.0%} volume spike")
    return ", ".join(parts)
