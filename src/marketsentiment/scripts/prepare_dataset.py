"""Fetch a public labeled financial-sentiment dataset into data/labeled.csv.

Sidesteps StockTwits rate limits for v1: pull a ready-made labeled corpus and write
the ``text,label`` schema that train_finbert.py expects. Later, append StockTwits-
harvested rows (harvest_labels.py) as domain augmentation.

Default: zeroshot/twitter-financial-news-sentiment (~10k finance tweets, closest to
the social-chatter use case). Needs the [finbert] extra.

    python -m marketsentiment.scripts.prepare_dataset --out data/labeled.csv
    # Financial PhraseBank instead:
    python -m marketsentiment.scripts.prepare_dataset \
        --dataset financial_phrasebank --config sentences_50agree
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

# Integer-label -> our vocabulary (bullish/bearish/neutral), per dataset.
_LABEL_MAPS: dict[str, dict[int, str]] = {
    "zeroshot/twitter-financial-news-sentiment": {0: "bearish", 1: "bullish", 2: "neutral"},
    "financial_phrasebank": {0: "bearish", 1: "neutral", 2: "bullish"},  # neg / neu / pos
}

_STRING_ALIASES = {"positive": "bullish", "negative": "bearish"}


def _to_label(value, mapping: dict[int, str]) -> str:
    if isinstance(value, str):
        v = value.lower()
        return _STRING_ALIASES.get(v, v)
    return mapping.get(int(value), "neutral")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a labeled sentiment dataset -> CSV.")
    parser.add_argument("--dataset", default="zeroshot/twitter-financial-news-sentiment")
    parser.add_argument("--config", default=None, help="HF dataset config (e.g. sentences_50agree)")
    parser.add_argument("--out", default="data/labeled.csv")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    args = parser.parse_args()

    from datasets import load_dataset

    ds = load_dataset(args.dataset, args.config) if args.config else load_dataset(args.dataset)
    mapping = _LABEL_MAPS.get(args.dataset, {})

    rows: list[tuple[str, str]] = []
    for split in ds:  # concatenate every split; the trainer does its own eval split
        for ex in ds[split]:
            text = str(ex[args.text_col]).replace("\n", " ").strip()
            if not text:
                continue
            rows.append((text, _to_label(ex[args.label_col], mapping)))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)

    dist = Counter(label for _, label in rows)
    print(f"Wrote {len(rows)} rows to {args.out}")
    print("Label distribution: " + ", ".join(f"{k}={v}" for k, v in sorted(dist.items())))
    print("(imbalance here is the point — train_finbert.py applies class weights.)")


if __name__ == "__main__":
    main()
