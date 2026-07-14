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

    # --- LLM (read straight from ANTHROPIC_API_KEY, no prefix) ---
    anthropic_api_key: str | None = None
    llm_model: str = "claude-opus-4-8"
    llm_max_tokens: int = 1024

    # --- Sources ---
    stocktwits_access_token: str | None = None
    reddit_client_id: str | None = None
    reddit_client_secret: str | None = None
    reddit_user_agent: str = "market-sentiment/0.1"

    # --- Sentiment ---
    sentiment_backend: str = "finbert"  # "finbert" | "llm"
    finbert_model: str = "ProsusAI/finbert"
    low_confidence_threshold: float = 0.55

    # --- Aggregation ---
    hot_min_mentions: int = 5
    hot_top_n: int = 10

    # --- Storage ---
    db_path: str = "data/marketsentiment.duckdb"

    def __init__(self, **kwargs):  # noqa: D401 - pull ANTHROPIC_API_KEY (unprefixed)
        import os

        kwargs.setdefault("anthropic_api_key", os.getenv("ANTHROPIC_API_KEY"))
        super().__init__(**kwargs)


@lru_cache
def get_settings() -> Settings:
    return Settings()
