"""Run the compiled graph once and persist the result. Shared by the API and CLI."""

from __future__ import annotations

import uuid

from marketsentiment.graph.state import PipelineState
from marketsentiment.storage.db import Database


def run_once(graph, db: Database | None, symbols: list[str] | None = None) -> PipelineState:
    run_id = uuid.uuid4().hex[:12]
    previous_counts = db.previous_counts() if db else {}

    # thread_id is required by the checkpointer; one per run keeps runs independent.
    config = {"configurable": {"thread_id": run_id}}
    state: PipelineState = graph.invoke(
        {"run_id": run_id, "symbols": symbols, "previous_counts": previous_counts},
        config=config,
    )

    if db is not None:
        db.save_run(
            run_id,
            state.get("classified", []),
            state.get("aggregates", []),
            state.get("brief"),
        )
    return state
