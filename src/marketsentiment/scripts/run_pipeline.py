"""Run one pipeline pass from the CLI.

    python -m marketsentiment.scripts.run_pipeline                 # trending symbols
    python -m marketsentiment.scripts.run_pipeline --symbols NVDA TSLA GME
"""

from __future__ import annotations

import argparse

from marketsentiment.config import get_settings
from marketsentiment.graph.pipeline import build_default_graph
from marketsentiment.observability.logging import configure_logging
from marketsentiment.runner import run_once
from marketsentiment.storage.db import Database


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the market-sentiment pipeline once.")
    parser.add_argument("--symbols", nargs="*", default=None, help="Tickers to pull (default: trending)")
    parser.add_argument("--no-db", action="store_true", help="Skip persistence")
    args = parser.parse_args()

    configure_logging()
    settings = get_settings()
    db = None if args.no_db else Database(settings.db_path)
    graph, llm_available = build_default_graph(settings)

    state = run_once(graph, db, symbols=args.symbols)

    print(f"\nRun {state.get('run_id')} — {len(state.get('raw_posts', []))} posts, "
          f"{len(state.get('aggregates', []))} tickers, llm={'on' if llm_available else 'off'}")
    print("\nHot stocks:")
    for h in state.get("hot", []):
        print(f"  ${h.symbol:<6} score {h.sentiment_score:+.2f}  ({h.reason})")
    if state.get("brief"):
        print("\nDaily brief:\n" + state["brief"])
    if db:
        db.close()


if __name__ == "__main__":
    main()
