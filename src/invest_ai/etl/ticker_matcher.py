"""Assign a ticker to a news item by the issuer alias dictionary."""

from __future__ import annotations

import re

from invest_ai import config


def _compile() -> dict[str, re.Pattern]:
    patterns = {}
    for ticker, info in config.tickers().items():
        aliases = sorted(info["aliases"], key=len, reverse=True)
        alt = "|".join(re.escape(a.lower()) for a in aliases)
        patterns[ticker] = re.compile(rf"(?<!\w)(?:{alt})(?!\w)")
    return patterns


_PATTERNS: dict[str, re.Pattern] | None = None


def match_ticker(text: str) -> str | None:
    """Ticker if exactly one issuer is mentioned, else None (market-wide news)."""
    global _PATTERNS
    if _PATTERNS is None:
        _PATTERNS = _compile()
    lowered = text.lower().replace("ё", "е")
    found = [t for t, p in _PATTERNS.items() if p.search(lowered)]
    return found[0] if len(found) == 1 else None
