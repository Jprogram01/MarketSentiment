"""Bluesky connector — STUB (the open substitute for X/Twitter).

X was descoped: its API is $100+/mo and scraping violates ToS. Bluesky's atproto
firehose is free and open, so it's the pluggable "modern social" source. Lower
finance volume than X had — treat it as breadth, not the sentiment workhorse.

Implementation sketch (`pip install -e ".[bluesky]"`):
    from atproto import Client
    client = Client(); client.login(handle, app_password)
    res = client.app.bsky.feed.search_posts({"q": f"${symbol}", "limit": limit})
    for post in res.posts:
        yield RawPost(id=f"bsky-{post.cid}", source=SourceType.BLUESKY,
                      text=post.record.text, ...)
"""

from __future__ import annotations

from marketsentiment.ingestion.base import Source
from marketsentiment.schemas import RawPost, SourceType


class BlueskySource(Source):
    source_type = SourceType.BLUESKY

    def fetch(self, symbols: list[str] | None = None, limit: int = 100) -> list[RawPost]:
        raise NotImplementedError("Bluesky source is a stub — wire atproto (see module docstring).")
