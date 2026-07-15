"""Run the pipeline once and email the brief — entrypoint for the scheduled daily job.

Runs on the LLM backend (MS_SENTIMENT_BACKEND=llm) so there's no torch to package —
it deploys as a lightweight AWS Lambda. Emails via SES. Stateless by default; set
MS_DB_S3_URI to persist the DuckDB across runs (enables the mention-velocity feature).

Local dry run:
    MS_SENTIMENT_BACKEND=llm MS_DIGEST_FROM=you@x.com MS_DIGEST_TO=you@x.com \
        python -m marketsentiment.scripts.daily_digest
"""

from __future__ import annotations

import os
from datetime import date

from marketsentiment.config import Settings, get_settings
from marketsentiment.observability.logging import configure_logging, get_logger
from marketsentiment.runner import run_once

log = get_logger(__name__)


def _format_email(state) -> tuple[str, str]:
    """Build (plain-text, html) email bodies from the pipeline state."""
    import html as _html

    from marketsentiment.aggregation import collect_examples

    brief = state.get("brief") or "No brief today (no hot tickers cleared the threshold)."
    hot = state.get("hot", [])
    examples = collect_examples(state.get("classified", []), [h.symbol for h in hot], per_ticker=3)

    # --- plain text ---
    text_lines = [brief, "", "Hot tickers:"]
    text_lines += [
        f"  ${h.symbol}: {h.n_mentions} mentions, score {h.sentiment_score:+.2f} ({h.reason})"
        for h in hot
    ] or ["  (none)"]
    text_lines += ["", "Sample chatter:"]
    for h in hot:
        posts = examples.get(h.symbol, [])
        if not posts:
            continue
        text_lines.append(f"  ${h.symbol}")
        text_lines += [
            f"    [{cp.sentiment.label.value}] {cp.post.text.strip()[:160]}" for cp in posts
        ]
    text = "\n".join(text_lines)

    # --- html ---
    rows = "".join(
        f"<tr><td>${h.symbol}</td><td>{h.n_mentions}</td>"
        f"<td>{h.sentiment_score:+.2f}</td><td>{_html.escape(h.reason)}</td></tr>"
        for h in hot
    )
    chatter = ""
    for h in hot:
        posts = examples.get(h.symbol, [])
        if not posts:
            continue
        items = "".join(
            f"<li><em>[{cp.sentiment.label.value}]</em> {_html.escape(cp.post.text.strip()[:200])}</li>"
            for cp in posts
        )
        chatter += f"<p style='margin:8px 0 2px'><b>${h.symbol}</b></p><ul style='margin:0'>{items}</ul>"
    html = (
        f"<h2>Daily Market Sentiment — {date.today().isoformat()}</h2>"
        f"<p>{_html.escape(brief).replace(chr(10), '<br>')}</p>"
        "<table cellpadding='6' cellspacing='0' border='1' style='border-collapse:collapse'>"
        "<tr><th>Ticker</th><th>Mentions</th><th>Score</th><th>Why</th></tr>"
        f"{rows}</table>"
        f"<h3>Sample chatter</h3>{chatter}"
        "<p style='color:#888;font-size:12px'>Sentiment measurement, not investment advice.</p>"
    )
    return text, html


def _send_ses(subject: str, text: str, html: str) -> None:
    import boto3  # noqa: PLC0415 - provided in the Lambda image, not needed to import module

    ses = boto3.client("ses", region_name=os.getenv("AWS_REGION", "us-east-1"))
    ses.send_email(
        Source=os.environ["MS_DIGEST_FROM"],
        Destination={"ToAddresses": [os.environ["MS_DIGEST_TO"]]},
        Message={
            "Subject": {"Data": subject},
            "Body": {"Text": {"Data": text}, "Html": {"Data": html}},
        },
    )


def _load_db(settings: Settings):
    """Pull the DuckDB file from S3 for cross-run velocity, or return None (stateless)."""
    uri = os.getenv("MS_DB_S3_URI")
    if not uri:
        return None
    import boto3  # noqa: PLC0415

    from marketsentiment.storage.db import Database  # noqa: PLC0415

    bucket, key = uri.removeprefix("s3://").split("/", 1)
    try:
        boto3.client("s3").download_file(bucket, key, settings.db_path)
    except Exception:  # first run — no prior file yet
        log.info("digest.db_first_run")
    return Database(settings.db_path)


def _save_db(settings: Settings) -> None:
    uri = os.getenv("MS_DB_S3_URI")
    if not uri:
        return
    import boto3  # noqa: PLC0415

    bucket, key = uri.removeprefix("s3://").split("/", 1)
    boto3.client("s3").upload_file(settings.db_path, bucket, key)


def run_digest() -> dict:
    configure_logging()
    settings = get_settings()

    db = _load_db(settings)
    graph, llm_available = build_default_graph_lazy(settings)
    state = run_once(graph, db, symbols=None)  # trending tickers
    if db is not None:
        db.close()
        _save_db(settings)

    subject = f"Daily Market Sentiment — {date.today().isoformat()}"
    text, html = _format_email(state)
    _send_ses(subject, text, html)

    result = {
        "date": date.today().isoformat(),
        "posts": len(state.get("raw_posts", [])),
        "hot": len(state.get("hot", [])),
        "llm_available": llm_available,
    }
    log.info("digest.sent", **result)
    return result


def build_default_graph_lazy(settings: Settings):
    # Imported here so the module loads without the graph deps present (e.g. for tooling).
    from marketsentiment.graph.pipeline import build_default_graph  # noqa: PLC0415

    return build_default_graph(settings)


def lambda_handler(event, context):
    return run_digest()


if __name__ == "__main__":
    print(run_digest())
