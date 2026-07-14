"""Head-to-head: fine-tuned FinBERT vs. Claude zero-shot on the same held-out split.

Turns "I fine-tuned a model" into "...and justified it against the LLM baseline with
numbers": accuracy, macro-F1, per-class recall, latency, and Claude's estimated cost
per 1k posts. Reproduces train_finbert.py's exact eval split (same seed) so the
comparison is on data neither model trained on.

    python -m marketsentiment.scripts.eval_compare --model models/finbert-st --sample 200

The LLM side runs if OPENAI_API_KEY (uses gpt-4o-mini) or ANTHROPIC_API_KEY is set — it
spends API credits, so keep --sample modest. Force one with --provider openai|anthropic.
FinBERT side is free/local and always runs.
"""

from __future__ import annotations

import argparse
import os
import time

# $ per 1M tokens (input, output) — approximate; edit to match current rates.
_PRICING = {
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.0),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
}

# Definitions + few-shot to fix the LLM's over-flagging of neutral posts.
_SYSTEM_PROMPT = (
    "Label the market sentiment of a financial social-media post as exactly one of "
    "bullish, bearish, or neutral.\n"
    "- bullish: implies the stock/market goes up, or clearly positive sentiment.\n"
    "- bearish: implies it goes down, or clearly negative sentiment.\n"
    "- neutral: factual reporting, a question, or mixed/no directional view. Most plain "
    "statements of fact and news headlines are neutral — do NOT infer sentiment that is "
    "not actually expressed.\n"
    "Account for sarcasm and emoji (rockets/charts up = bullish, chart down = bearish).\n"
    "Examples:\n"
    "  '$AAPL breaking out, to the moon' -> bullish\n"
    "  'Tesla recalls 100k cars over a brake defect' -> bearish\n"
    "  'Apple reports earnings Thursday after the bell' -> neutral\n"
    "  'Is $NVDA overvalued here?' -> neutral"
)


def _metrics(y_true, y_pred, labels):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    _, rc, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": float(f1.mean()),
        "recall": {lab: float(rc[i]) for i, lab in enumerate(labels)},
    }


def _make_chat(provider: str, model: str, max_tokens: int):
    if provider == "openai":
        from langchain_openai import ChatOpenAI  # noqa: PLC0415

        return ChatOpenAI(model=model, max_tokens=max_tokens)
    from langchain_anthropic import ChatAnthropic  # noqa: PLC0415

    return ChatAnthropic(model=model, max_tokens=max_tokens)


def _run_llm(texts, gold, labels, provider, model):
    from typing import Literal

    from pydantic import BaseModel

    class _S(BaseModel):
        label: Literal["bullish", "bearish", "neutral"]

    chat = _make_chat(provider, model, 64).with_structured_output(_S, include_raw=True)
    preds, in_tok, out_tok = [], 0, 0
    t0 = time.perf_counter()
    for text in texts:
        res = chat.invoke([("system", _SYSTEM_PROMPT), ("human", f"Post:\n{text}")])
        parsed = res["parsed"]
        preds.append(parsed.label if parsed else "neutral")
        usage = getattr(res["raw"], "usage_metadata", None) or {}
        in_tok += usage.get("input_tokens", 0)
        out_tok += usage.get("output_tokens", 0)
    elapsed = time.perf_counter() - t0

    p_in, p_out = _PRICING.get(model, (1.0, 3.0))
    cost = in_tok / 1e6 * p_in + out_tok / 1e6 * p_out
    return _metrics(gold, preds, labels), elapsed, {"cost": cost, "in": in_tok, "out": out_tok}


def _print_report(rows, labels, n):
    print(f"\n=== FinBERT vs. LLM — {n} held-out posts ===\n")
    head = f"{'system':<30}{'acc':>7}{'macroF1':>9}"
    head += "".join(f"{'R:' + lab:>12}" for lab in labels)
    head += f"{'ms/post':>10}{'$/1k':>9}"
    print(head)
    print("-" * len(head))
    for name, m, elapsed, cost in rows:
        line = f"{name:<30}{m['accuracy']:>7.3f}{m['macro_f1']:>9.3f}"
        line += "".join(f"{m['recall'][lab]:>12.3f}" for lab in labels)
        line += f"{elapsed / n * 1000:>10.1f}"
        line += f"{cost['cost'] / n * 1000:>9.3f}" if cost else f"{'—':>9}"
        print(line)
    print("\nR:<class> = recall. FinBERT: local, free, fast. Claude: no training data needed,")
    print("but per-call latency + cost. That tradeoff is the point of the comparison.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare FinBERT vs. Claude zero-shot.")
    parser.add_argument("--data", default="data/labeled.csv")
    parser.add_argument("--model", default="models/finbert-st", help="Fine-tuned FinBERT dir")
    parser.add_argument("--sample", type=int, default=200, help="Held-out posts for the head-to-head")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--provider", choices=["auto", "openai", "anthropic"], default="auto")
    parser.add_argument("--llm-model", default=None, help="default per provider (gpt-4o-mini / claude-opus-4-8)")
    args = parser.parse_args()

    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(args.data).dropna(subset=["text", "label"])
    labels = sorted(df["label"].unique())
    y = df["label"].map({lab: i for i, lab in enumerate(labels)})

    # Reproduce train_finbert.py's split, then take a stratified subsample for the head-to-head.
    _, eval_df = train_test_split(df, test_size=args.test_size, stratify=y, random_state=args.seed)
    if args.sample and args.sample < len(eval_df):
        eval_df, _ = train_test_split(
            eval_df, train_size=args.sample, stratify=eval_df["label"], random_state=args.seed
        )
    texts, gold = eval_df["text"].tolist(), eval_df["label"].tolist()
    print(f"Held-out posts: {len(texts)} | labels: {labels}")

    from marketsentiment.sentiment.finbert import FinBERTClassifier

    fb = FinBERTClassifier(args.model)
    t0 = time.perf_counter()
    fb_pred = [r.label.value for r in fb.classify(texts)]
    rows = [("FinBERT (fine-tuned)", _metrics(gold, fb_pred, labels), time.perf_counter() - t0, None)]

    provider = args.provider
    if provider == "auto":
        if os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        elif os.getenv("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        else:
            provider = None
    key_var = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(provider)
    model = args.llm_model or ("gpt-4o-mini" if provider == "openai" else "claude-opus-4-8")

    if provider and os.getenv(key_var):
        try:
            m, elapsed, cost = _run_llm(texts, gold, labels, provider, model)
            rows.append((f"{model} (0-shot)", m, elapsed, cost))
        except Exception as exc:  # pragma: no cover
            print(f"[llm skipped] {exc}")
    else:
        print("[llm skipped] set OPENAI_API_KEY (or ANTHROPIC_API_KEY) to include the LLM baseline")

    _print_report(rows, labels, len(texts))


if __name__ == "__main__":
    main()
