"""Central configuration, loaded from environment / .env (prefix ``MS_``)."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM (keys read from OPENAI_API_KEY / ANTHROPIC_API_KEY, no MS_ prefix) ---
    llm_provider: str = "auto"  # "auto" (openai if key, else anthropic) | "openai" | "anthropic"
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    llm_model: str | None = None  # None -> per-provider default
    llm_max_tokens: int = 2048  # synthesis writes a paragraph per hot ticker

    # --- Sources ---
    stocktwits_access_token: str | None = None
    reddit_client_id: str | None = None
    reddit_client_secret: str | None = None
    reddit_user_agent: str = "market-sentiment/0.1"

    # --- Sentiment ---
    sentiment_backend: str = "finbert"  # "finbert" | "llm" | "ensemble"
    finbert_model: str = "ProsusAI/finbert"
    low_confidence_threshold: float = 0.55
    finbert_weight: float = 0.6  # ensemble backend only: weight on FinBERT vs. the LLM

    # --- Aggregation ---
    hot_min_mentions: int = 5
    hot_top_n: int = 10

    # --- Storage ---
    db_path: str = "data/marketsentiment.duckdb"

    def __init__(self, **kwargs):  # noqa: D401 - pull provider keys (unprefixed)
        import os

        kwargs.setdefault("openai_api_key", os.getenv("OPENAI_API_KEY"))
        kwargs.setdefault("anthropic_api_key", os.getenv("ANTHROPIC_API_KEY"))
        super().__init__(**kwargs)

    def resolved_provider(self) -> str | None:
        """The active LLM provider, or None if no key is available."""
        if self.llm_provider != "auto":
            return self.llm_provider
        if self.openai_api_key:
            return "openai"
        if self.anthropic_api_key:
            return "anthropic"
        return None

    def resolved_model(self) -> str:
        if self.llm_model:
            return self.llm_model
        return "gpt-4o-mini" if self.resolved_provider() == "openai" else "claude-opus-4-8"

    def llm_enabled(self) -> bool:
        provider = self.resolved_provider()
        if provider == "openai":
            return bool(self.openai_api_key)
        if provider == "anthropic":
            return bool(self.anthropic_api_key)
        return False


@lru_cache
def get_settings() -> Settings:
    return Settings()
