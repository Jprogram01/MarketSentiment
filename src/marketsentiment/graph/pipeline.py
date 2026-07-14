"""Build and compile the LangGraph pipeline.

Topology:

    ingest → classify → aggregate ─┬─▶ synthesize ─▶ END   (hot stocks found + LLM available)
                                    └─────────────▶ END     (nothing to brief on)

The conditional edge after ``aggregate`` is the branching LangGraph buys us over a
straight LCEL chain; ``classify`` additionally re-routes low-confidence posts to the
LLM internally. Compiled with an in-memory checkpointer so runs are resumable and
inspectable — swap ``MemorySaver`` for a persistent checkpointer in production.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from marketsentiment.config import Settings
from marketsentiment.graph.nodes import (
    make_aggregate_node,
    make_classify_node,
    make_ingest_node,
    make_synthesize_node,
)
from marketsentiment.graph.state import PipelineState
from marketsentiment.ingestion.base import Source
from marketsentiment.ingestion.fourchan import FourChanBizSource
from marketsentiment.ingestion.stocktwits import StockTwitsSource
from marketsentiment.sentiment.base import SentimentClassifier


def build_graph(
    sources: list[Source],
    classifier: SentimentClassifier,
    llm_classifier: SentimentClassifier | None,
    settings: Settings,
    known_tickers: frozenset[str] | None = None,
    llm_available: bool = False,
):
    graph = StateGraph(PipelineState)

    graph.add_node("ingest", make_ingest_node(sources))
    graph.add_node(
        "classify",
        make_classify_node(
            classifier, llm_classifier, known_tickers, settings.low_confidence_threshold
        ),
    )
    graph.add_node("aggregate", make_aggregate_node(settings.hot_min_mentions, settings.hot_top_n))
    provider = settings.resolved_provider() or "openai"
    graph.add_node(
        "synthesize",
        make_synthesize_node(provider, settings.resolved_model(), settings.llm_max_tokens),
    )

    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "classify")
    graph.add_edge("classify", "aggregate")

    def route_after_aggregate(state: PipelineState) -> str:
        if llm_available and state.get("hot"):
            return "synthesize"
        return END

    graph.add_conditional_edges(
        "aggregate", route_after_aggregate, {"synthesize": "synthesize", END: END}
    )
    graph.add_edge("synthesize", END)

    return graph.compile(checkpointer=MemorySaver())


def build_default_graph(settings: Settings):
    """Construct sources, classifier, and LLM from settings, then compile the graph.

    Returns (compiled_graph, llm_available). Runs on FinBERT alone if no API key is set.
    """
    sources: list[Source] = [
        StockTwitsSource(access_token=settings.stocktwits_access_token),
        FourChanBizSource(),
    ]

    provider = settings.resolved_provider()
    model = settings.resolved_model()
    llm_available = settings.llm_enabled()

    if settings.sentiment_backend == "llm" and llm_available:
        from marketsentiment.sentiment.llm import LLMClassifier

        classifier: SentimentClassifier = LLMClassifier(provider, model)
        llm_classifier = None  # already the primary
    else:
        from marketsentiment.sentiment.finbert import FinBERTClassifier

        classifier = FinBERTClassifier(settings.finbert_model)
        llm_classifier = None
        if llm_available:
            from marketsentiment.sentiment.llm import LLMClassifier

            llm_classifier = LLMClassifier(provider, model)

    graph = build_graph(
        sources=sources,
        classifier=classifier,
        llm_classifier=llm_classifier,
        settings=settings,
        llm_available=llm_available,
    )
    return graph, llm_available
