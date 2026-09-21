"""Fixed reference strategies for comparable, no-tuning research studies."""
from __future__ import annotations

from typing import Any


BENCHMARKS = (
    {
        "factor": "value",
        "label": "Value",
        "definition": "Cross-sectional composite of earnings, book, and sales cheapness.",
        "requires_fundamentals": True,
    },
    {
        "factor": "quality",
        "label": "Quality",
        "definition": "Cross-sectional profitability and balance-sheet health composite.",
        "requires_fundamentals": True,
    },
    {
        "factor": "momentum",
        "label": "Momentum (12-1)",
        "definition": "Trailing price momentum with the most recent month skipped.",
        "requires_fundamentals": False,
    },
    {
        "factor": "lowvol",
        "label": "Low volatility",
        "definition": "Cross-sectional preference for lower trailing realized volatility.",
        "requires_fundamentals": False,
    },
)


def requests_for_suite(request: dict[str, Any]) -> tuple[list[tuple[dict, dict]], list[dict]]:
    """Return the fixed compatible requests and explicit, never-imputed skips."""
    provider = str(request.get("provider", "synthetic")).lower()
    declared_trials = max(int(request.get("n_trials", 1)), len(BENCHMARKS))
    runs: list[tuple[dict, dict]] = []
    skipped: list[dict] = []
    for benchmark in BENCHMARKS:
        if benchmark["requires_fundamentals"] and provider == "yfinance":
            skipped.append({
                "factor": benchmark["factor"],
                "label": benchmark["label"],
                "reason": "Yahoo Finance does not supply filing-dated fundamentals for this benchmark.",
            })
            continue
        run = dict(request)
        run["factor"] = benchmark["factor"]
        run["n_trials"] = declared_trials
        runs.append((benchmark, run))
    return runs, skipped
