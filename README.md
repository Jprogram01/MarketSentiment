# MarketSentiment

An **agentic social-sentiment pipeline** for financial markets. It ingests posts from
online market-chatter spaces, extracts the tickers being discussed, classifies bullish
/ bearish / neutral sentiment, aggregates it per stock, flags "hot" tickers by volume
and momentum, and writes a short daily briefing.

Orchestrated with **LangGraph**, with **FinBERT** as the sentiment workhorse and
**Claude** (via `langchain-anthropic`) for ticker disambiguation and synthesis.

> ⚠️ **Not investment advice.** This measures what the internet is *saying* about
> stocks and how fast that's changing. It does not recommend trades and must not be
> used as a trading signal.

---

## Why this design

- **It's a data-engineering + NLP + MLOps system, not a chatbot.** The interview value
  is the pipeline: pluggable ingestion, ticker disambiguation, a fine-tunable classifier,
  aggregation/velocity, and a served + observable deployment.
- **StockTwits is the anchor source** because users self-tag posts **Bullish/Bearish** —
  free labeled financial-sentiment data. Harvest it, fine-tune FinBERT on it, and
  evaluate. The class imbalance (chatter skews bullish) is the engineering story, the
  same shape as a queen-vs-worker imbalance problem: data pipeline + weighted loss.
- **LangGraph over a plain chain** because the flow branches and re-routes: low-confidence
  FinBERT predictions escalate to the LLM, and synthesis only runs when there's something
  to brief on. Shared, checkpointed state makes runs resumable and inspectable.
- **X was descoped on purpose.** Its API is $100+/mo and scraping violates ToS, so
  **Bluesky** is the open substitute. That's a mature engineering decision, documented.

## Architecture

```
 sources (pluggable)            LangGraph pipeline
┌───────────────────┐
│ StockTwits  ✅     │   ingest ─▶ classify ─▶ aggregate ─┬─▶ synthesize ─▶ END
│ 4chan /biz/ ✅     │─▶            │  │                    └───────────────▶ END
│ Reddit      (stub)│              │  └─ low-confidence ──▶ LLM re-classify
│ Bluesky     (stub)│              │                        (conditional)
└───────────────────┘              ▼
                          FinBERT  ·  Claude (langchain-anthropic)
                                          │
                        DuckDB  ◀─────────┘   FastAPI  ·  Docker  ·  LangSmith tracing
```

Sentiment score per ticker = `(bullish − bearish) / mentions ∈ [−1, 1]`. "Hot" =
clears a mention floor and ranks high on volume × conviction, boosted by a volume spike
vs. the previous run.

## Quickstart

```bash
cp .env.example .env          # add ANTHROPIC_API_KEY (optional) + tokens
make install                  # core deps (no torch)
make install-finbert          # add this when you want the local FinBERT model

# one pipeline pass (trending tickers), persisted to DuckDB
python -m marketsentiment.scripts.run_pipeline --symbols NVDA TSLA GME

# serve it
make run                      # http://localhost:8000/docs
# or: make docker
```

Runs on **FinBERT alone with no API key**. Set `ANTHROPIC_API_KEY` to enable the LLM
low-confidence fallback + daily synthesis; set `LANGCHAIN_TRACING_V2=true` for LangSmith.

### API

| Method | Path | Purpose |
|--------|------|---------|
| `GET`  | `/health` | liveness + whether the LLM is wired |
| `POST` | `/run` `{ "symbols": ["NVDA"] }` | run one pass (omit `symbols` for trending) |
| `GET`  | `/trending` | latest per-ticker sentiment, ranked by volume |
| `GET`  | `/ticker/{symbol}` | sentiment history for one ticker |
| `GET`  | `/brief` | latest synthesized briefing |

## The ML loop (the part to talk about in interviews)

```bash
# 1. Get labeled data — a public set (recommended, no scraping)…
python -m marketsentiment.scripts.prepare_dataset --out data/labeled.csv
#    …or harvest StockTwits self-labels: harvest_labels --symbols AAPL NVDA TSLA --out data/labeled.csv

# 2. (torch < 2.6 only) FinBERT's base ships .bin-only weights transformers 5 won't
#    load on old torch — convert once to safetensors:
python -m marketsentiment.scripts.convert_to_safetensors --out models/finbert-base-st

# 3. Fine-tune with class-weighted loss (add --base-model models/finbert-base-st if you did step 2)
python -m marketsentiment.scripts.train_finbert --data data/labeled.csv --out models/finbert-st

# 4. Serve the fine-tuned checkpoint
export MS_FINBERT_MODEL=models/finbert-st   # PowerShell: $env:MS_FINBERT_MODEL="models/finbert-st"
```

Prefer training on **Google Colab** — a self-contained GPU notebook is at
`notebooks/train_colab.ipynb`.

### Results (first fine-tune)

`ProsusAI/finbert` fine-tuned on ~11.9k labeled finance posts
(`zeroshot/twitter-financial-news-sentiment`), 3 epochs, class-weighted loss.
Held-out test (2,387 posts):

| Class | Precision | Recall | F1 | Support |
|-------|:---------:|:------:|:--:|:-------:|
| bearish · minority (15%) | 0.77 | **0.77** | 0.77 | 358 |
| bullish | 0.82 | 0.79 | 0.81 | 480 |
| neutral · majority (65%) | 0.91 | 0.92 | 0.91 | 1549 |
| **accuracy** | | | **0.87** | 2387 |
| **macro-F1** | | | **0.83** | 2387 |

The point: bearish is only 15% of the data, yet its recall holds at **~0.77** rather than
collapsing toward zero — the class-weighted loss earning its keep.

**FinBERT vs. Claude head-to-head** — run both on the same held-out split and compare
accuracy, per-class recall, latency, and cost:

```bash
python -m marketsentiment.scripts.eval_compare --sample 200
# fills in the LLM column when OPENAI_API_KEY (gpt-4o-mini, cheap) or ANTHROPIC_API_KEY is set
```

## Layout

```
src/marketsentiment/
  config.py            env-driven settings
  schemas.py           pydantic contract (RawPost, ClassifiedPost, TickerAggregate, HotStock)
  ingestion/           Source ABC + stocktwits, fourchan (real), reddit, bluesky (stubs)
  nlp/tickers.py       cashtag + known-ticker extraction w/ stoplist
  sentiment/           SentimentClassifier protocol; finbert + llm backends
  aggregation/         per-ticker scoring + hot-stock detection
  graph/               LangGraph state, nodes, compiled pipeline
  storage/db.py        DuckDB persistence
  api/main.py          FastAPI service
  runner.py            run-once + persist (shared by API and CLI)
  scripts/             prepare_dataset, harvest_labels, convert_to_safetensors,
                       train_finbert, eval_compare, run_pipeline
notebooks/             train_colab.ipynb — Colab GPU fine-tune
tests/                 unit tests (ticker extraction, aggregation) — no heavy deps
```

## Roadmap

- Wire the Reddit (PRAW) and Bluesky (atproto) sources — the ingestion layer is already
  pluggable.
- Promote the low-confidence LLM fallback + ticker disambiguation to explicit graph nodes.
- Schedule daily runs (cron), swap `MemorySaver` for a persistent checkpointer.
- Dashboard over the DuckDB aggregates.

## License

MIT
