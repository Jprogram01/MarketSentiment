"""Shared state passed between graph nodes.

LangGraph threads one state object through every node; each node returns a partial
update that's merged in. That shared, checkpointed state is exactly what a plain
LangChain chain can't give you — and what makes the low-confidence re-route and
resumable runs possible.
"""

from __future__ import annotations

from typing import TypedDict

from marketsentiment.schemas import ClassifiedPost, HotStock, RawPost, TickerAggregate


class PipelineState(TypedDict, total=False):
    # inputs
    run_id: str
    symbols: list[str] | None
    previous_counts: dict[str, int]
    # intermediate / outputs
    raw_posts: list[RawPost]
    classified: list[ClassifiedPost]
    aggregates: list[TickerAggregate]
    hot: list[HotStock]
    brief: str | None
    errors: list[str]
