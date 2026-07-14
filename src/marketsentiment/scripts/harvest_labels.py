"""Harvest StockTwits Bullish/Bearish self-labels into a training set.

This is the project's ML hook: StockTwits users hand-tag their own posts, giving you
free labeled financial-sentiment data. Pull it here, then fine-tune FinBERT on it
(scripts/train_finbert.py) and evaluate on a held-out split.

    python -m marketsentiment.scripts.harvest_labels --symbols AAPL NVDA TSLA --out data/labeled.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from marketsentiment.config import get_settings
from marketsentiment.ingestion.stocktwits import StockTwitsSource


def main() -> None:
    parser = argparse.ArgumentParser(description="Harvest StockTwits labeled messages.")
    parser.add_argument("--symbols", nargs="*", default=None, help="Tickers (default: trending)")
    parser.add_argument("--out", default="data/labeled.csv")
    args = parser.parse_args()

    settings = get_settings()
    src = StockTwitsSource(access_token=settings.stocktwits_access_token)
    posts = src.fetch(symbols=args.symbols, limit=10_000)
    labeled = [p for p in posts if p.gold_label is not None]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        for p in labeled:
            writer.writerow([p.text.replace("\n", " "), p.gold_label.value])

    n_bull = sum(p.gold_label.value == "bullish" for p in labeled)
    n_bear = len(labeled) - n_bull
    print(f"Wrote {len(labeled)} labeled rows to {args.out}  (bullish={n_bull}, bearish={n_bear})")
    if labeled:
        print("NOTE: StockTwits skews bullish — mirror the queen/worker imbalance story: "
              "use class weights or resampling when you train (see train_finbert.py).")


if __name__ == "__main__":
    main()
