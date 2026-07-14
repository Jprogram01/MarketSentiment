"""Pluggable source interface. Every connector returns normalized ``RawPost`` objects,
so adding X/Bluesky/Reddit later is a drop-in — the graph never changes."""

from __future__ import annotations

from abc import ABC, abstractmethod

from marketsentiment.schemas import RawPost, SourceType


class Source(ABC):
    source_type: SourceType

    @abstractmethod
    def fetch(self, symbols: list[str] | None = None, limit: int = 100) -> list[RawPost]:
        """Return recent posts.

        Args:
            symbols: cashtags to pull (e.g. ["AAPL", "NVDA"]). If None, the source
                decides what "trending / recent" means for its platform.
            limit: soft cap on posts returned.
        """
        raise NotImplementedError
