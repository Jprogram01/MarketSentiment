"""Ticker extraction.

Two-tier, cheap-first: cashtags (`$AAPL`) are unambiguous and free; bare uppercase
tokens are matched against a known-ticker set and filtered through a stoplist to cut
false positives ("CEO", "USD", "YOLO"). The genuinely ambiguous cases — "is 'apple'
the company or the fruit?" — are where an LLM disambiguation node earns its place
(see sentiment/llm.py::disambiguate_tickers), which is why this lives in a graph
that supports a conditional re-route rather than a straight-line chain.
"""

from __future__ import annotations

import re

_CASHTAG_RE = re.compile(r"\$([A-Za-z]{1,5})\b")
_BARE_RE = re.compile(r"\b([A-Z]{1,5})\b")

# Uppercase tokens that look like tickers but almost never are, in market chatter.
DEFAULT_STOPWORDS: frozenset[str] = frozenset(
    {
        "A", "I", "DD", "CEO", "CFO", "IPO", "USD", "USA", "EPS", "ATH", "YOLO",
        "FUD", "FOMO", "WSB", "PR", "SEC", "FDA", "GDP", "CPI", "AI", "EV", "IMO",
        "TL", "DR", "TLDR", "OP", "US", "UK", "EU", "OK", "LOL", "NGL", "IRA",
    }
)


def extract_tickers(
    text: str,
    known: frozenset[str] | set[str] | None = None,
    stopwords: frozenset[str] = DEFAULT_STOPWORDS,
) -> list[str]:
    """Return unique tickers in first-seen order.

    Cashtags always count. Bare uppercase tokens count only when they're in ``known``
    (if provided) and not in ``stopwords``.
    """
    found: list[str] = []
    seen: set[str] = set()

    def _add(sym: str) -> None:
        sym = sym.upper()
        if sym in seen:
            return
        seen.add(sym)
        found.append(sym)

    for m in _CASHTAG_RE.finditer(text):
        _add(m.group(1))

    if known is not None:
        known_upper = {k.upper() for k in known}
        for m in _BARE_RE.finditer(text):
            tok = m.group(1)
            if tok in stopwords:
                continue
            if tok.upper() in known_upper:
                _add(tok)

    return found
