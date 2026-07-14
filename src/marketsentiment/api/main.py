"""FastAPI service — serve the pipeline behind an API, matching the ant-detector setup
(FastAPI + Docker + structured logging; LangSmith handles tracing via env)."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from pydantic import BaseModel

from marketsentiment.config import get_settings
from marketsentiment.graph.pipeline import build_default_graph
from marketsentiment.observability.logging import configure_logging, get_logger
from marketsentiment.runner import run_once
from marketsentiment.storage.db import Database

log = get_logger(__name__)


class RunRequest(BaseModel):
    symbols: list[str] | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = get_settings()
    app.state.settings = settings
    app.state.db = Database(settings.db_path)
    app.state.graph, app.state.llm_available = build_default_graph(settings)
    log.info(
        "startup",
        backend=settings.sentiment_backend,
        llm_available=app.state.llm_available,
        model=settings.llm_model,
    )
    yield
    app.state.db.close()


app = FastAPI(title="MarketSentiment", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    log.info(
        "request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        ms=round((time.perf_counter() - start) * 1000, 1),
    )
    return response


@app.get("/health")
def health():
    return {"status": "ok", "llm_available": app.state.llm_available}


@app.post("/run")
def run(req: RunRequest):
    """Trigger one pipeline pass. Synchronous for simplicity — for production, push
    this onto a background worker / scheduled job and return a run_id immediately."""
    state = run_once(app.state.graph, app.state.db, symbols=req.symbols)
    return {
        "run_id": state.get("run_id"),
        "n_posts": len(state.get("raw_posts", [])),
        "n_classified": len(state.get("classified", [])),
        "n_tickers": len(state.get("aggregates", [])),
        "hot": [h.model_dump() for h in state.get("hot", [])],
        "brief": state.get("brief"),
        "errors": state.get("errors", []),
    }


@app.get("/trending")
def trending(limit: int = 20):
    return {"trending": app.state.db.latest_trending(limit)}


@app.get("/ticker/{symbol}")
def ticker(symbol: str, limit: int = 30):
    return {"symbol": symbol.upper(), "history": app.state.db.ticker_history(symbol, limit)}


@app.get("/brief")
def brief():
    return {"brief": app.state.db.latest_brief()}
