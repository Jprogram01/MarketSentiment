# MarketSentiment

An **agentic social-sentiment pipeline** for financial markets. It ingests posts from
online market-chatter spaces, extracts the tickers being discussed, classifies bullish
/ bearish / neutral sentiment, aggregates it per stock, flags "hot" tickers by volume
and momentum, and writes a short daily briefing.

Orchestrated with **LangGraph**, with a fine-tuned **FinBERT** as the sentiment workhorse
and an **LLM** (OpenAI or Anthropic, via LangChain) for ticker disambiguation and synthesis.

> ⚠️ **Not investment advice.** This measures what the internet is *saying* about
> stocks and how fast that's changing. It does not recommend trades and must not be
> used as a trading signal.

---

## Why this design

- **A data-engineering + NLP + MLOps system, not a chatbot.** The emphasis is the
  engineering pipeline: pluggable ingestion, ticker disambiguation, a fine-tunable
  classifier, aggregation/velocity, and a served, observable deployment.
- **StockTwits is the anchor source** because users self-tag posts **Bullish/Bearish** —
  free labeled financial-sentiment data to fine-tune and evaluate on. The class imbalance
  (chatter skews heavily toward one class) is the core modeling challenge, addressed with a
  data pipeline plus a class-weighted loss.
- **LangGraph over a plain chain** because the flow branches and re-routes: low-confidence
  FinBERT predictions escalate to the LLM, and synthesis only runs when there's something
  to brief on. Shared, checkpointed state makes runs resumable and inspectable.
- **X was descoped deliberately.** Its API is $100+/mo and scraping violates its ToS, so
  **Bluesky** (open `atproto` API) is the substitute — the ingestion layer stays pluggable.

## Architecture

```
 sources (pluggable)            LangGraph pipeline
┌───────────────────┐
│ StockTwits  ✅     │   ingest ─▶ classify ─▶ aggregate ─┬─▶ synthesize ─▶ END
│ 4chan /biz/ ✅     │─▶            │  │                    └───────────────▶ END
│ Reddit      (stub)│              │  └─ low-confidence ──▶ LLM re-classify
│ Bluesky     (stub)│              │                        (conditional)
└───────────────────┘              ▼
                          FinBERT  ·  LLM (OpenAI / Anthropic)
                                          │
                        DuckDB  ◀─────────┘   FastAPI  ·  Docker  ·  LangSmith tracing
```

Sentiment score per ticker = `(bullish − bearish) / mentions ∈ [−1, 1]`. "Hot" =
clears a mention floor and ranks high on volume × conviction, boosted by a volume spike
vs. the previous run.

## Quickstart

```bash
cp .env.example .env          # add OPENAI_API_KEY (optional) + source tokens
make install                  # core deps (no torch)
make install-finbert          # add this when you want the local FinBERT model

# one pipeline pass (trending tickers), persisted to DuckDB
python -m marketsentiment.scripts.run_pipeline --symbols NVDA TSLA GME

# serve it
make run                      # http://localhost:8000/docs
# or: make docker
```

Runs on **FinBERT alone with no API key**. Set `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY`)
to enable the LLM low-confidence fallback + daily synthesis — provider is auto-detected,
defaulting to OpenAI `gpt-4o-mini`. Set `LANGCHAIN_TRACING_V2=true` for LangSmith.

### API

| Method | Path | Purpose |
|--------|------|---------|
| `GET`  | `/health` | liveness + whether the LLM is wired |
| `POST` | `/run` `{ "symbols": ["NVDA"] }` | run one pass (omit `symbols` for trending) |
| `GET`  | `/trending` | latest per-ticker sentiment, ranked by volume |
| `GET`  | `/ticker/{symbol}` | sentiment history for one ticker |
| `GET`  | `/brief` | latest synthesized briefing |

## Training FinBERT: the class-imbalance problem

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

**FinBERT vs. LLM head-to-head** — run both on the same held-out split and compare
accuracy, per-class recall, latency, and cost:

```bash
python -m marketsentiment.scripts.eval_compare --sample 200
# uses OPENAI_API_KEY (gpt-4o-mini) or ANTHROPIC_API_KEY; stronger model: --llm-model gpt-4o
```

Example run (200 held-out posts; LLMs zero-shot with a definitions + few-shot prompt):

| system | acc | macro-F1 | R:bearish | R:bullish | R:neutral | ms/post | $/1k |
|--------|:---:|:--------:|:---------:|:---------:|:---------:|:-------:|:----:|
| **FinBERT (fine-tuned)** | **0.86** | **0.81** | 0.67 | 0.80 | 0.92 | ~80 | free |
| gpt-4o-mini (0-shot) | 0.81 | 0.76 | 0.70 | 0.58 | 0.90 | 598 | 0.04 |
| gpt-4o (0-shot) | 0.79 | 0.75 | **0.83** | 0.65 | 0.82 | 550 | 0.67 |

What the comparison shows:

1. **The fine-tuned model wins** on accuracy and macro-F1 — while being free and ~7× faster.
   It learned this data's neutral-heavy base rates; the LLMs don't know them.
2. **Prompt > model size here.** A definitions + few-shot prompt lifted gpt-4o-mini from
   0.66 → 0.81 accuracy; upgrading mini → gpt-4o (≈17× the cost) *lowered* accuracy slightly.
   The bigger model was not the answer.
3. **The LLM's edge is bearish recall** (gpt-4o 0.83) — LLMs are good at spotting negativity.
   So the natural role for the LLM here is the low-confidence fallback that catches negatives
   FinBERT misses — exactly the conditional re-route the graph already supports.

(200-sample numbers are noisy; run `--sample 2400` for the full held-out set.)

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
                       train_finbert, eval_compare, run_pipeline, daily_digest
notebooks/             train_colab.ipynb — Colab GPU fine-tune
tests/                 unit tests (ticker extraction, aggregation) — no heavy deps
```

## Roadmap

- Wire the Reddit (PRAW) and Bluesky (atproto) sources — the ingestion layer is already
  pluggable.
- Promote the low-confidence LLM fallback + ticker disambiguation to explicit graph nodes.
- Swap `MemorySaver` for a persistent checkpointer to enable durable, resumable runs.
- Dashboard over the DuckDB aggregates.

## Scheduled daily digest

A serverless job (AWS Lambda + EventBridge Scheduler + SES) runs the pipeline once a day
and emails a sentiment brief. Runs on the LLM backend, so there's no torch to package —
`sam build && sam deploy`. Setup in **[DEPLOY.md](DEPLOY.md)**; entrypoint is
`scripts/daily_digest.py`.

## License

MIT
