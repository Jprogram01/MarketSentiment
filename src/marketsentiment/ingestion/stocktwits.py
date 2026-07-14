"""StockTwits connector — the primary source.

Why StockTwits leads: users self-tag messages **Bullish** or **Bearish**, so every
labeled message is free supervised training/eval data for the sentiment model
(see scripts/harvest_labels.py and scripts/train_finbert.py). We still run the
model on all messages — the label is the eval signal, not a shortcut.

API note: the public JSON endpoints are rate-limited and, depending on the account,
some now require an access token. This connector degrades gracefully — on a 4xx/429
it logs and returns what it has rather than raising.
"""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from marketsentiment.ingestion.base import Source
from marketsentiment.observability.logging import get_logger
from marketsentiment.schemas import RawPost, Sentiment, SourceType

log = get_logger(__name__)

_BASE = "https://api.stocktwits.com/api/2"
_TRENDING = f"{_BASE}/trending/symbols.json"
_SYMBOL_STREAM = _BASE + "/streams/symbol/{symbol}.json"


def _map_gold_label(raw: str | None) -> Sentiment | None:
    if raw == "Bullish":
        return Sentiment.BULLISH
    if raw == "Bearish":
        return Sentiment.BEARISH
    return None


class StockTwitsSource(Source):
    source_type = SourceType.STOCKTWITS

    def __init__(self, access_token: str | None = None, timeout: float = 10.0):
        self._token = access_token
        self._client = httpx.Client(timeout=timeout, headers={"User-Agent": "market-sentiment/0.1"})

    # ---- public API ----

    def fetch(self, symbols: list[str] | None = None, limit: int = 100) -> list[RawPost]:
        if symbols is None:
            symbols = self.trending_symbols()
        posts: list[RawPost] = []
        for sym in symbols:
            posts.extend(self._fetch_symbol(sym))
            if len(posts) >= limit:
                break
        return posts[:limit]

    def trending_symbols(self, top: int = 10) -> list[str]:
        data = self._get(_TRENDING)
        symbols = [s["symbol"] for s in (data or {}).get("symbols", [])]
        return symbols[:top]

    # ---- internals ----

    def _fetch_symbol(self, symbol: str) -> list[RawPost]:
        data = self._get(_SYMBOL_STREAM.format(symbol=symbol))
        out: list[RawPost] = []
        for msg in (data or {}).get("messages", []):
            entities = msg.get("entities") or {}
            sentiment = (entities.get("sentiment") or {}).get("basic")
            out.append(
                RawPost(
                    id=f"st-{msg['id']}",
                    source=SourceType.STOCKTWITS,
                    text=msg.get("body", ""),
                    created_at=_parse_ts(msg.get("created_at")),
                    author=(msg.get("user") or {}).get("username"),
                    url=None,
                    symbols=[s["symbol"] for s in msg.get("symbols", [])],
                    gold_label=_map_gold_label(sentiment),
                )
            )
        return out

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=False)
    def _get(self, url: str) -> dict | None:
        params = {"access_token": self._token} if self._token else None
        try:
            resp = self._client.get(url, params=params)
            if resp.status_code == 429:
                log.warning("stocktwits.rate_limited", url=url)
                return None
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as exc:  # pragma: no cover - network path
            log.warning("stocktwits.http_error", url=url, error=str(exc))
            return None


def _parse_ts(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    # StockTwits returns e.g. "2024-05-01T12:34:56Z"
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
