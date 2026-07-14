"""LLM sentiment + synthesis via Claude (LangChain ``ChatAnthropic``).

Three jobs, all through the official Anthropic integration:
  1. LLMClassifier  — zero-shot sentiment baseline to compare against FinBERT, and
     the fallback for low-confidence FinBERT predictions (the graph's conditional edge).
  2. disambiguate_tickers — resolve ambiguous symbols ("apple" the company vs fruit).
  3. build_daily_brief — the synthesis node: turn aggregates into a readable briefing.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from marketsentiment.schemas import HotStock, Sentiment, SentimentResult, TickerAggregate

_DEFAULT_MODEL = "claude-opus-4-8"

_SENTIMENT_LABELS = {
    "bullish": Sentiment.BULLISH,
    "bearish": Sentiment.BEARISH,
    "neutral": Sentiment.NEUTRAL,
}


class _LLMSentiment(BaseModel):
    label: Literal["bullish", "bearish", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)


def _make_chat(model: str, max_tokens: int):
    # Imported lazily so the package imports without langchain-anthropic installed.
    from langchain_anthropic import ChatAnthropic  # noqa: PLC0415

    return ChatAnthropic(model=model, max_tokens=max_tokens)


class LLMClassifier:
    """Zero-shot financial sentiment using structured output."""

    def __init__(self, model: str = _DEFAULT_MODEL, max_tokens: int = 256):
        self._model = model
        self._chat = _make_chat(model, max_tokens).with_structured_output(_LLMSentiment)

    def classify(self, texts: list[str]) -> list[SentimentResult]:
        results: list[SentimentResult] = []
        for text in texts:
            prompt = (
                "Classify the market sentiment of this social-media post about a stock. "
                "Consider sarcasm and emoji (🚀 = bullish). Post:\n\n" + text
            )
            parsed: _LLMSentiment = self._chat.invoke(prompt)  # type: ignore[assignment]
            results.append(
                SentimentResult(
                    label=_SENTIMENT_LABELS[parsed.label],
                    confidence=parsed.confidence,
                    model=f"{self._model} (zero-shot)",
                )
            )
        return results


def build_daily_brief(
    aggregates: list[TickerAggregate],
    hot: list[HotStock],
    model: str = _DEFAULT_MODEL,
    max_tokens: int = 1024,
) -> str:
    """Synthesis node: a short, descriptive market-sentiment briefing.

    Descriptive by design — this reports what the internet is saying and how fast it's
    moving. It is NOT investment advice and must not recommend trades.
    """
    chat = _make_chat(model, max_tokens)
    hot_lines = "\n".join(
        f"- ${h.symbol}: {h.n_mentions} mentions, score {h.sentiment_score:+.2f} ({h.reason})"
        for h in hot
    )
    prompt = (
        "You are a market-sentiment analyst. Using ONLY the aggregated social-sentiment "
        "data below, write a concise daily briefing (<200 words) on what retail social "
        "chatter is saying and where momentum is shifting. Be descriptive, cite the "
        "numbers, and do NOT give buy/sell/hold recommendations or investment advice.\n\n"
        f"Hot tickers:\n{hot_lines}\n"
    )
    resp = chat.invoke(prompt)
    return resp.content if isinstance(resp.content, str) else str(resp.content)


def disambiguate_tickers(candidates: list[str], text: str, model: str = _DEFAULT_MODEL) -> list[str]:
    """Given candidate symbols and their source text, drop ones used non-financially.

    Stub hook for the graph's conditional re-route — implement with a structured-output
    call that returns the subset actually referring to a tradeable security.
    """
    raise NotImplementedError("Wire a structured-output call; see module docstring.")
