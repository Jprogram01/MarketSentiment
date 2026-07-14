"""Reddit connector — STUB.

Wire this with PRAW (`pip install -e ".[reddit]"`), reading r/wallstreetbets,
r/stocks, r/investing. High volume, no labels — relies fully on the sentiment model.

Implementation sketch:
    import praw
    reddit = praw.Reddit(client_id=..., client_secret=..., user_agent=...)
    for sub in ("wallstreetbets", "stocks"):
        for post in reddit.subreddit(sub).hot(limit=limit):
            yield RawPost(id=f"rd-{post.id}", source=SourceType.REDDIT,
                          text=f"{post.title}\n{post.selftext}", ...)
"""

from __future__ import annotations

from marketsentiment.ingestion.base import Source
from marketsentiment.schemas import RawPost, SourceType


class RedditSource(Source):
    source_type = SourceType.REDDIT

    def __init__(self, client_id: str | None, client_secret: str | None, user_agent: str):
        self._client_id = client_id
        self._client_secret = client_secret
        self._user_agent = user_agent

    def fetch(self, symbols: list[str] | None = None, limit: int = 100) -> list[RawPost]:
        raise NotImplementedError("Reddit source is a stub — wire PRAW (see module docstring).")
