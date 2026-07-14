"""4chan /biz/ connector — the noisy, unlabeled source.

Uses the official read-only JSON API (a.4cdn.org) — no auth. /biz/ is high-noise and
often toxic, which is a legitimate engineering challenge: heavy filtering/denoising
before the text ever reaches the classifier. Here we pull thread OPs from the catalog
and strip HTML; a production version would add spam/quality filtering.
"""

from __future__ import annotations

import html
import re
from datetime import datetime, timezone

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from marketsentiment.ingestion.base import Source
from marketsentiment.observability.logging import get_logger
from marketsentiment.schemas import RawPost, SourceType

log = get_logger(__name__)

_CATALOG = "https://a.4cdn.org/biz/catalog.json"
_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(comment: str) -> str:
    text = _TAG_RE.sub(" ", comment).replace("<br>", " ")
    return html.unescape(text).strip()


class FourChanBizSource(Source):
    source_type = SourceType.FOURCHAN

    def __init__(self, timeout: float = 10.0):
        self._client = httpx.Client(timeout=timeout, headers={"User-Agent": "market-sentiment/0.1"})

    def fetch(self, symbols: list[str] | None = None, limit: int = 100) -> list[RawPost]:
        data = self._get(_CATALOG)
        out: list[RawPost] = []
        for page in data or []:
            for thread in page.get("threads", []):
                comment = thread.get("com")
                if not comment:
                    continue
                out.append(
                    RawPost(
                        id=f"biz-{thread['no']}",
                        source=SourceType.FOURCHAN,
                        text=_strip_html(comment),
                        created_at=datetime.fromtimestamp(
                            thread.get("time", 0), tz=timezone.utc
                        ),
                        author=None,
                        url=f"https://boards.4chan.org/biz/thread/{thread['no']}",
                    )
                )
                if len(out) >= limit:
                    return out
        return out

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=False)
    def _get(self, url: str) -> list | None:
        try:
            resp = self._client.get(url)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as exc:  # pragma: no cover - network path
            log.warning("fourchan.http_error", url=url, error=str(exc))
            return None
