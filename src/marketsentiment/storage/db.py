"""DuckDB persistence — columnar analytics store for runs, aggregates, and briefs.

DuckDB is embedded (no server) and great at the group-by/time-series queries this
project asks of it. One file, cheap to ship in Docker, easy to hand a notebook.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from marketsentiment.schemas import ClassifiedPost, TickerAggregate

_SCHEMA = """
CREATE TABLE IF NOT EXISTS classified_posts (
    run_id     VARCHAR,
    post_id    VARCHAR,
    source     VARCHAR,
    text       VARCHAR,
    tickers    VARCHAR,   -- JSON array
    label      VARCHAR,
    confidence DOUBLE,
    model      VARCHAR,
    created_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS aggregates (
    run_id          VARCHAR,
    run_at          TIMESTAMP,
    symbol          VARCHAR,
    n_mentions      INTEGER,
    bullish         INTEGER,
    bearish         INTEGER,
    neutral         INTEGER,
    sentiment_score DOUBLE
);

CREATE TABLE IF NOT EXISTS briefs (
    run_id VARCHAR,
    run_at TIMESTAMP,
    brief  VARCHAR
);
"""


class Database:
    def __init__(self, path: str = "data/marketsentiment.duckdb"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(path)
        self._conn.execute(_SCHEMA)

    # ---- writes ----

    def save_run(
        self,
        run_id: str,
        classified: list[ClassifiedPost],
        aggregates: list[TickerAggregate],
        brief: str | None,
    ) -> None:
        run_at = datetime.now(timezone.utc)
        for cp in classified:
            self._conn.execute(
                "INSERT INTO classified_posts VALUES (?,?,?,?,?,?,?,?,?)",
                [
                    run_id,
                    cp.post.id,
                    cp.post.source.value,
                    cp.post.text,
                    json.dumps(cp.tickers),
                    cp.sentiment.label.value,
                    cp.sentiment.confidence,
                    cp.sentiment.model,
                    cp.post.created_at,
                ],
            )
        for agg in aggregates:
            self._conn.execute(
                "INSERT INTO aggregates VALUES (?,?,?,?,?,?,?,?)",
                [
                    run_id,
                    run_at,
                    agg.symbol,
                    agg.n_mentions,
                    agg.bullish,
                    agg.bearish,
                    agg.neutral,
                    agg.sentiment_score,
                ],
            )
        if brief:
            self._conn.execute("INSERT INTO briefs VALUES (?,?,?)", [run_id, run_at, brief])

    # ---- reads ----

    def previous_counts(self) -> dict[str, int]:
        """Mention counts from the most recent completed run (for velocity)."""
        row = self._conn.execute("SELECT max(run_at) FROM aggregates").fetchone()
        if not row or row[0] is None:
            return {}
        rows = self._conn.execute(
            "SELECT symbol, n_mentions FROM aggregates WHERE run_at = ?", [row[0]]
        ).fetchall()
        return {sym: n for sym, n in rows}

    def latest_trending(self, limit: int = 20) -> list[dict]:
        rows = self._conn.execute(
            """
            SELECT symbol, n_mentions, sentiment_score
            FROM aggregates
            WHERE run_at = (SELECT max(run_at) FROM aggregates)
            ORDER BY n_mentions DESC
            LIMIT ?
            """,
            [limit],
        ).fetchall()
        return [{"symbol": s, "n_mentions": n, "sentiment_score": sc} for s, n, sc in rows]

    def ticker_history(self, symbol: str, limit: int = 30) -> list[dict]:
        rows = self._conn.execute(
            """
            SELECT run_at, n_mentions, sentiment_score
            FROM aggregates WHERE symbol = ?
            ORDER BY run_at DESC LIMIT ?
            """,
            [symbol.upper(), limit],
        ).fetchall()
        return [
            {"run_at": r.isoformat(), "n_mentions": n, "sentiment_score": sc} for r, n, sc in rows
        ]

    def latest_brief(self) -> str | None:
        row = self._conn.execute(
            "SELECT brief FROM briefs ORDER BY run_at DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else None

    def close(self) -> None:
        self._conn.close()
