"""LLM sentiment + synthesis, provider-agnostic (OpenAI or Anthropic via LangChain).

Three jobs, all behind one ``_make_chat`` so the provider is a config switch:
  1. LLMClassifier  — zero-shot sentiment baseline / low-confidence FinBERT fallback.
  2. build_daily_brief — the synthesis node: aggregates -> readable briefing.
  3. disambiguate_tickers — resolve ambiguous symbols (stub hook).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from marketsentiment.schemas import (
    ClassifiedPost,
    HotStock,
    Sentiment,
    SentimentResult,
    TickerAggregate,
)

_SENTIMENT_LABELS = {
    "bullish": Sentiment.BULLISH,
    "bearish": Sentiment.BEARISH,
    "neutral": Sentiment.NEUTRAL,
}

# Definitions + few-shot — keeps the LLM from over-flagging neutral posts (see eval_compare).
_CLS_SYSTEM = (
    "Label the market sentiment of a financial social-media post as exactly one of "
    "bullish, bearish, or neutral.\n"
    "- bullish: implies the stock/market goes up, or clearly positive sentiment.\n"
    "- bearish: implies it goes down, or clearly negative sentiment.\n"
    "- neutral: factual reporting, a question, or mixed/no directional view. Most plain "
    "statements of fact and news headlines are neutral — do NOT infer sentiment that is "
    "not actually expressed.\n"
    "Account for sarcasm and emoji. Examples:\n"
    "  '$AAPL breaking out, to the moon' -> bullish\n"
    "  'Tesla recalls 100k cars over a brake defect' -> bearish\n"
    "  'Apple reports earnings Thursday after the bell' -> neutral"
)


class _LLMSentiment(BaseModel):
    label: Literal["bullish", "bearish", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)


def _make_chat(provider: str, model: str, max_tokens: int, temperature: float = 0.0):
    """Build a LangChain chat model for the given provider (imported lazily).

    temperature defaults to 0 for consistency — sentiment labels shouldn't wander
    run-to-run. Not passed to Anthropic: newer Claude models reject the parameter.
    """
    if provider == "openai":
        from langchain_openai import ChatOpenAI  # noqa: PLC0415

        return ChatOpenAI(model=model, max_tokens=max_tokens, temperature=temperature)
    from langchain_anthropic import ChatAnthropic  # noqa: PLC0415

    return ChatAnthropic(model=model, max_tokens=max_tokens)


class LLMClassifier:
    """Zero-shot financial sentiment via structured output."""

    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini", max_tokens: int = 64):
        self._provider = provider
        self._model = model
        self._chat = _make_chat(provider, model, max_tokens).with_structured_output(_LLMSentiment)

    def classify(self, texts: list[str]) -> list[SentimentResult]:
        results: list[SentimentResult] = []
        for text in texts:
            parsed: _LLMSentiment = self._chat.invoke(  # type: ignore[assignment]
                [("system", _CLS_SYSTEM), ("human", f"Post:\n{text}")]
            )
            results.append(
                SentimentResult(
                    label=_SENTIMENT_LABELS[parsed.label],
                    confidence=float(parsed.confidence),
                    model=f"{self._model} (zero-shot)",
                )
            )
        return results


def build_daily_brief(
    aggregates: list[TickerAggregate],
    hot: list[HotStock],
    examples: dict[str, list[ClassifiedPost]] | None = None,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    max_tokens: int = 2048,
) -> str:
    """Synthesis node: a per-ticker briefing that explains the WHY from the actual posts.

    Descriptive by design — reports what chatter is saying and why. NOT investment advice.
    """
    examples = examples or {}
    chat = _make_chat(provider, model, max_tokens)

    blocks = []
    for h in hot:
        posts = examples.get(h.symbol, [])
        sample = "\n".join(
            f"    [{cp.sentiment.label.value}] {cp.post.text.strip()[:160]}" for cp in posts
        ) or "    (no sample posts)"
        blocks.append(
            f"${h.symbol} — {h.n_mentions} mentions, score {h.sentiment_score:+.2f}\n{sample}"
        )
    body = "\n\n".join(blocks) or "(no hot tickers today)"

    prompt = (
        "You are a market-sentiment analyst. For EACH ticker below you have its stats and a "
        "sample of the actual posts. Write one short paragraph per ticker covering: (1) the "
        "ticker and company — what it does; (2) the sentiment read, citing mentions + score; "
        "(3) most important — WHY people are saying it, drawn from the sample posts (the real "
        "themes: earnings, a product/deal, hype, fear, a catalyst). Paraphrase the chatter and "
        "you may quote a short phrase. If a ticker is unclear or the posts are just noise/spam, "
        "say so instead of inventing a reason. Finish with a one-sentence overall takeaway. "
        "This is sentiment measurement, NOT investment advice — no buy/sell/hold calls.\n\n"
        + body
    )
    resp = chat.invoke(prompt)
    return resp.content if isinstance(resp.content, str) else str(resp.content)


def disambiguate_tickers(
    candidates: list[str], text: str, provider: str = "openai", model: str = "gpt-4o-mini"
) -> list[str]:
    """Drop candidate symbols used non-financially. Stub hook for the graph's re-route."""
    raise NotImplementedError("Wire a structured-output call; see module docstring.")
