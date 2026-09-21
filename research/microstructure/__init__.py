"""Auditable market-microstructure research primitives.

The package starts with trade-event research because public historical trade data
is broadly available.  It deliberately does not relabel trade-price markouts as
quote-level adverse selection; that requires dated BBO or order-book data.
"""

from .lab import (
    MicrostructureError,
    evaluate_trade_flow,
    load_binance_trades,
    options_diagnostics,
    run_microstructure_study,
    simulate_quoting,
    synthetic_trade_events,
    validate_trade_events,
)
from .collector import LocalQuoteStore, validate_quote_events
from .itch import parse_itch_messages, reconstruct_top_of_book

__all__ = [
    "MicrostructureError",
    "evaluate_trade_flow",
    "load_binance_trades",
    "options_diagnostics",
    "run_microstructure_study",
    "simulate_quoting",
    "synthetic_trade_events",
    "validate_trade_events",
    "LocalQuoteStore",
    "validate_quote_events",
    "parse_itch_messages",
    "reconstruct_top_of_book",
]
