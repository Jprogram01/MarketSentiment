"""Graph nodes as dependency-injecting factories.

Each ``make_*`` returns the node callable with its collaborators (sources, classifier,
LLM) closed over — so the graph wiring stays declarative and nodes stay unit-testable.
"""

from __future__ import annotations

from marketsentiment.aggregation import build_aggregates, detect_hot_stocks
from marketsentiment.graph.state import PipelineState
from marketsentiment.ingestion.base import Source
from marketsentiment.nlp import extract_tickers
from marketsentiment.observability.logging import get_logger
from marketsentiment.schemas import ClassifiedPost
from marketsentiment.sentiment.base import SentimentClassifier

log = get_logger(__name__)


def make_ingest_node(sources: list[Source]):
    def ingest(state: PipelineState) -> PipelineState:
        symbols = state.get("symbols")
        raw: list = []
        errors: list[str] = []
        for src in sources:
            try:
                raw.extend(src.fetch(symbols=symbols))
            except NotImplementedError:
                continue  # stub source — skip quietly
            except Exception as exc:  # pragma: no cover - defensive
                errors.append(f"{src.source_type.value}: {exc}")
                log.warning("ingest.source_failed", source=src.source_type.value, error=str(exc))
        log.info("ingest.done", n_posts=len(raw))
        return {"raw_posts": raw, "errors": errors}

    return ingest


def make_classify_node(
    classifier: SentimentClassifier,
    llm_classifier: SentimentClassifier | None,
    known_tickers: frozenset[str] | None,
    low_confidence_threshold: float,
):
    """Extract tickers, classify sentiment, and re-route low-confidence posts to the LLM.

    The low-confidence fallback is the graph's cyclic/branching value in miniature:
    the cheap FinBERT pass handles the bulk, and only the uncertain tail escalates to
    the pricier LLM. (Promote this to its own node + conditional edge if you want it
    surfaced explicitly in the graph topology.)
    """

    def classify(state: PipelineState) -> PipelineState:
        posts = state.get("raw_posts", [])
        if not posts:
            return {"classified": []}

        results = classifier.classify([p.text for p in posts])

        classified: list[ClassifiedPost] = []
        low_conf: list[int] = []
        for post, res in zip(posts, results):
            tickers = extract_tickers(post.text, known=known_tickers) or [
                s.upper() for s in post.symbols
            ]
            if not tickers:
                continue
            if res.confidence < low_confidence_threshold and llm_classifier is not None:
                low_conf.append(len(classified))
            classified.append(ClassifiedPost(post=post, tickers=tickers, sentiment=res))

        if low_conf and llm_classifier is not None:
            log.info("classify.llm_fallback", n=len(low_conf))
            retexts = [classified[i].post.text for i in low_conf]
            for j, res in enumerate(llm_classifier.classify(retexts)):
                idx = low_conf[j]
                classified[idx] = classified[idx].model_copy(update={"sentiment": res})

        log.info("classify.done", n_classified=len(classified))
        return {"classified": classified}

    return classify


def make_aggregate_node(min_mentions: int, top_n: int):
    def aggregate(state: PipelineState) -> PipelineState:
        aggregates = build_aggregates(state.get("classified", []))
        hot = detect_hot_stocks(
            aggregates,
            min_mentions=min_mentions,
            top_n=top_n,
            previous_counts=state.get("previous_counts"),
        )
        log.info("aggregate.done", n_tickers=len(aggregates), n_hot=len(hot))
        return {"aggregates": aggregates, "hot": hot}

    return aggregate


def make_synthesize_node(model: str, max_tokens: int):
    def synthesize(state: PipelineState) -> PipelineState:
        from marketsentiment.sentiment.llm import build_daily_brief

        brief = build_daily_brief(state["aggregates"], state["hot"], model, max_tokens)
        log.info("synthesize.done", chars=len(brief))
        return {"brief": brief}

    return synthesize
