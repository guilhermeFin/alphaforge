"""Point-in-time fundamentals -> value & quality factors.

The hard, honesty-critical problem with fundamental factors is look-ahead:
  * a fiscal quarter's numbers are only knowable on the FILING date, weeks after
    the period ends, and
  * vendors silently overwrite history with RESTATED values.
Using either naively backdates information you couldn't have had, and is the
single most common way a fundamental backtest lies.

So every observation here carries an ``available_date`` (the filing date), and
``build_fundamentals`` produces panels in which a value appears ONLY from its
available_date forward — never before. tests/test_fundamentals.py proves it with
the same point-in-time truncation test used for prices and text signals.

Free data caveat (documented, not hidden): truly point-in-time fundamental
history is expensive. yfinance and most free sources serve LATEST/restated
numbers, so the only honest offline path today is the synthetic generator
(data.make_synthetic_fundamentals). Real PIT vendors (e.g. point-in-time
Compustat, Sharadar SF1) plug into the same ``build_fundamentals`` interface.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .factors import cross_sectional_zscore, blend


def point_in_time_panel(
    obs: pd.DataFrame,
    index: pd.DatetimeIndex,
    symbols: list[str],
    metric: str,
    max_staleness: int = 400,
) -> pd.DataFrame:
    """Build a (dates x symbols) panel for ``metric`` that is point-in-time safe.

    Each observation is placed at the first trading date >= its ``available_date``
    and forward-filled until the next filing (or it goes stale after
    ``max_staleness`` trading days, modelling a missing/late report). Values are
    never filled backward, so a number cannot influence a date before it was filed.
    NaN means "no current fundamental" — such names get zero weight downstream
    (they are excluded from the cross-section, not treated as a zero signal).
    """
    sub = obs[obs["metric"] == metric].sort_values("available_date")
    panel = pd.DataFrame(np.nan, index=index, columns=symbols, dtype=float)
    col = {s: i for i, s in enumerate(symbols)}
    for r in sub.itertuples(index=False):
        if r.symbol not in col:
            continue
        pos = index.searchsorted(pd.Timestamp(r.available_date))
        if pos >= len(index):
            continue  # filed after the sample — unusable
        panel.iat[pos, col[r.symbol]] = r.value
    return panel.ffill(limit=max_staleness)


@dataclass
class Fundamentals:
    """A bundle of point-in-time metric panels (each dates x symbols)."""
    panels: dict[str, pd.DataFrame]

    def __getitem__(self, metric: str) -> pd.DataFrame:
        return self.panels[metric]

    def has(self, metric: str) -> bool:
        return metric in self.panels


def build_fundamentals(
    obs: pd.DataFrame,
    index: pd.DatetimeIndex,
    symbols: list[str],
    metrics: list[str] | None = None,
    max_staleness: int = 400,
) -> Fundamentals:
    metrics = metrics or sorted(obs["metric"].unique())
    return Fundamentals({m: point_in_time_panel(obs, index, symbols, m, max_staleness) for m in metrics})


# ----------------------------- raw ratios -----------------------------
def earnings_yield(fund: Fundamentals, close: pd.DataFrame) -> pd.DataFrame:
    """E/P — higher is cheaper. Uses the last *filed* EPS over today's price."""
    return fund["eps_ttm"] / close


def book_to_price(fund: Fundamentals, close: pd.DataFrame) -> pd.DataFrame:
    return fund["bvps"] / close


def sales_to_price(fund: Fundamentals, close: pd.DataFrame) -> pd.DataFrame:
    return fund["sps"] / close


# ----------------------------- composite scores -----------------------------
def value_score(fund: Fundamentals, close: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional VALUE composite: cheap on earnings, book, and sales."""
    z = cross_sectional_zscore
    return blend(
        z(earnings_yield(fund, close)),
        z(book_to_price(fund, close)),
        z(sales_to_price(fund, close)),
    )


def quality_score(fund: Fundamentals, close: pd.DataFrame | None = None) -> pd.DataFrame:
    """Cross-sectional QUALITY composite: high ROE, high gross profitability
    (Novy-Marx), low leverage. ``close`` is unused (kept for a uniform signature)."""
    z = cross_sectional_zscore
    return blend(
        z(fund["roe"]),
        z(fund["gross_profitability"]),
        z(-fund["leverage"]),
    )


def value_quality_score(fund: Fundamentals, close: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight blend of the value and quality composites (a classic combo:
    cheap AND good)."""
    return blend(value_score(fund, close), quality_score(fund, close))
