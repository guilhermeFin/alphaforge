"""Deterministic licensed-data bundles for research tests.

Two fixtures with opposite jobs, and the difference matters:

``write_research_bundle`` -- the **normal research path**. Inputs vary the way
real company data varies, so a factor can rank names, take positions, and change
its ranking over time. Use it whenever a test asks "does this analysis work?"

``write_degenerate_bundle`` -- the **error and degenerate path**. Every metric is
one per-symbol constant times a shared base, so the constant cancels in any
ratio and every fundamental ratio factor is identical across names. Use it only
to prove a safeguard fires. It is not a realistic bundle and no research claim
should ever be read off it.

The distinction is the point. A fixture that accidentally produces a flat
cross-section will let a broken analysis pass, because a strategy that never
trades never contradicts anything.

Design rules for the realistic bundle
-------------------------------------
1. **Scale is separated from ratios.** Company size (``ASSET_BASE``) is drawn
   independently of the drivers that set ratios (``TURNOVER``, ``MARGIN``). A
   ratio like ``(revenue - cogs) / total_assets`` reduces to
   ``turnover * margin``, so it is governed by the ratio drivers and is immune
   to the size term. This is the defect the old fixture had: one scale
   multiplied every metric, so it cancelled and every ratio collapsed to a
   single number.
2. **Ratios evolve.** Each driver has its own per-period drift, so factor values
   move through time and the cross-sectional ranking can change.
3. **Prices are independent of fundamentals.** Deterministic trends and
   oscillations with different phases per name, built from their own constants.
   No fundamental driver reaches price generation, so no relationship between a
   factor and future returns is planted. Note what this does *not* claim: an
   independent fixture can still come out positive or negative by chance, and
   neither outcome is informative. Fixture backtest performance is not a research
   finding, so tests never assert return sign, magnitude, Sharpe, or credibility.
4. **Point-in-time is preserved.** A period's figures carry
   ``available_date = period_end + REPORTING_LAG_DAYS``; they are not knowable
   before the filing lands.
5. **No randomness.** Every value comes from an explicit closed form over the
   symbol index and period index, so a test can hand-check any single number.
"""
from __future__ import annotations

import json
import math

import pandas as pd

#: Filing lag: a quarter's figures are not knowable at the period end.
REPORTING_LAG_DAYS = 75

SYMBOLS = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")

# ---- ratio drivers: deliberately NOT ordered like ASSET_BASE -----------------
# gross_profitability reduces to turnover * margin, so these two lists alone set
# it. Neither follows company size, so the factor carries no size information.
TURNOVER = (0.90, 0.72, 1.05, 0.64, 0.83, 0.96)
MARGIN = (0.22, 0.41, 0.28, 0.36, 0.48, 0.31)
#: turnover * margin -> 0.198, 0.295, 0.294, 0.230, 0.398, 0.298
TURNOVER_DRIFT = (0.006, -0.004, 0.002, 0.008, -0.005, 0.001)
MARGIN_DRIFT = (0.004, -0.003, 0.002, 0.005, -0.004, 0.001)

# ---- scale: sets absolute size only, and must not reach any ratio ------------
ASSET_BASE = (5.0e9, 1.2e10, 8.0e8, 2.4e10, 3.1e9, 6.5e9)
SHARE_BASE = (1.0e8, 3.2e8, 4.5e7, 6.0e8, 8.0e7, 1.5e8)
#: Per-name asset growth. This must vary: ``asset_growth`` is
#: ``(ta - ta.shift) / ta.shift``, so ``ASSET_BASE`` cancels and a single shared
#: growth rate would make the factor identical across names -- the same
#: cancellation defect this fixture exists to avoid.
ASSET_GROWTH = (0.006, 0.019, 0.011, 0.024, 0.003, 0.015)

# ---- other ratio drivers, each varying independently ------------------------
EQUITY_RATIO = (0.62, 0.41, 0.75, 0.33, 0.55, 0.48)   # -> leverage varies
SGA_RATE = (0.09, 0.17, 0.12, 0.21, 0.07, 0.14)       # -> operating margin varies
OCF_RATIO = (1.18, 0.92, 1.35, 0.81, 1.05, 1.24)      # -> accruals vary
CAPEX_RATE = (0.04, 0.09, 0.03, 0.11, 0.06, 0.07)     # -> fcf yield varies
ISSUANCE = (0.002, -0.001, 0.004, 0.000, -0.003, 0.001)  # -> share issuance varies

# ---- prices: independent of every driver above ------------------------------
PRICE_BASE = (42.0, 118.0, 17.5, 260.0, 63.0, 88.0)
PRICE_TREND = (0.18, -0.06, 0.31, 0.04, -0.12, 0.09)
PRICE_AMP = (0.07, 0.12, 0.05, 0.09, 0.14, 0.06)
PRICE_PERIOD = (61, 43, 77, 52, 38, 67)
PRICE_PHASE = (0.0, 0.35, 0.7, 0.15, 0.5, 0.85)

METRIC_ORDER = (
    "revenue", "cogs", "sga", "ebit", "interest_expense", "net_income",
    "total_assets", "common_equity", "current_assets", "current_liabilities",
    "total_liabilities", "retained_earnings", "current_debt", "long_term_debt",
    "cash", "op_cash_flow", "capex", "shares_diluted",
)


def _manifest(*, point_in_time_fundamentals: bool, include_universe: bool) -> dict:
    return {
        "schema_version": 1,
        "provider_name": "AlphaForge deterministic research fixture",
        "license_acknowledged": True,
        "prices_adjusted_for_corporate_actions": True,
        "point_in_time_fundamentals": point_in_time_fundamentals,
        "survivorship_free_universe": include_universe,
        "includes_delisted_securities": include_universe,
    }


def _price(j: int, t: int) -> float:
    """Deterministic trend plus phase-shifted oscillation. No link to fundamentals."""
    trend = 1.0 + PRICE_TREND[j] * (t / 500.0)
    wave = PRICE_AMP[j] * math.sin(2.0 * math.pi * (t / PRICE_PERIOD[j] + PRICE_PHASE[j]))
    return round(PRICE_BASE[j] * (trend + wave), 4)


def _period_metrics(j: int, k: int) -> dict[str, float]:
    """One symbol's figures for one fiscal period, from explicit closed forms."""
    turnover = max(0.05, TURNOVER[j] + TURNOVER_DRIFT[j] * k)
    margin = min(0.85, max(0.05, MARGIN[j] + MARGIN_DRIFT[j] * k))

    total_assets = ASSET_BASE[j] * (1.0 + ASSET_GROWTH[j] * k)
    revenue = total_assets * turnover / 4.0
    gross_profit = revenue * margin
    cogs = revenue - gross_profit
    sga = revenue * SGA_RATE[j]
    ebit = gross_profit - sga

    common_equity = total_assets * EQUITY_RATIO[j]
    total_liabilities = total_assets - common_equity
    long_term_debt = total_liabilities * 0.55
    current_debt = total_liabilities * 0.20
    interest_expense = (current_debt + long_term_debt) * 0.011
    net_income = (ebit - interest_expense) * 0.79

    return {
        "revenue": revenue,
        "cogs": cogs,
        "sga": sga,
        "ebit": ebit,
        "interest_expense": interest_expense,
        "net_income": net_income,
        "total_assets": total_assets,
        "common_equity": common_equity,
        "current_assets": total_assets * 0.38,
        "current_liabilities": total_liabilities * 0.25,
        "total_liabilities": total_liabilities,
        "retained_earnings": common_equity * 0.61,
        "current_debt": current_debt,
        "long_term_debt": long_term_debt,
        "cash": total_assets * 0.07,
        "op_cash_flow": net_income * OCF_RATIO[j],
        "capex": revenue * CAPEX_RATE[j],
        "shares_diluted": SHARE_BASE[j] * (1.0 + ISSUANCE[j] * k),
    }


def write_research_bundle(root, *, days: int = 520, symbols=SYMBOLS,
                          point_in_time_fundamentals: bool = True,
                          include_universe: bool = True,
                          period_days: int = 63,
                          ineligible_on_index: int | None = None) -> None:
    """Write a realistic licensed bundle with genuine cross-sectional variation.

    ``ineligible_on_index`` drops the last symbol from the universe on that one
    date, so historical-membership handling stays exercisable.
    """
    root = __import__("pathlib").Path(root)
    names = list(symbols)
    (root / "manifest.json").write_text(
        json.dumps(_manifest(point_in_time_fundamentals=point_in_time_fundamentals,
                             include_universe=include_universe)), encoding="utf-8")

    dates = pd.bdate_range("2020-01-02", periods=days)
    rows = []
    for t, date in enumerate(dates):
        for j, symbol in enumerate(names):
            rows.append({"date": date.date().isoformat(), "symbol": symbol,
                         "close": _price(j, t),
                         "volume": 750_000 + 125_000 * j + 900 * (t % 37)})
    pd.DataFrame(rows).to_csv(root / "prices.csv", index=False)

    if include_universe:
        membership = []
        for t, date in enumerate(dates):
            for symbol in names:
                eligible = not (ineligible_on_index is not None
                                and t == ineligible_on_index and symbol == names[-1])
                membership.append({"date": date.date().isoformat(), "symbol": symbol,
                                   "eligible": eligible})
        pd.DataFrame(membership).to_csv(root / "universe.csv", index=False)

    if point_in_time_fundamentals:
        facts = []
        for k, period_end in enumerate(dates[period_days - 1::period_days]):
            available = period_end + pd.Timedelta(days=REPORTING_LAG_DAYS)
            if available > dates[-1]:
                break
            for j, symbol in enumerate(names):
                metrics = _period_metrics(j, k)
                for metric in METRIC_ORDER:
                    facts.append({"symbol": symbol,
                                  "period_end": period_end.date().isoformat(),
                                  "available_date": available.date().isoformat(),
                                  "metric": metric, "value": float(metrics[metric])})
        pd.DataFrame(facts).to_csv(root / "fundamentals.csv", index=False)


def write_degenerate_bundle(root, *, days: int = 520,
                            point_in_time_fundamentals: bool = True,
                            include_universe: bool = True) -> None:
    """Write a deliberately flat bundle for testing the insufficient-variation guard.

    Every metric is ``base * scale`` with one ``scale`` per symbol, so the scale
    cancels in any ratio and every fundamental ratio factor is identical across
    names. This is not a realistic bundle. It exists so the data-health guard has
    something to catch; never read a research result off it.
    """
    root = __import__("pathlib").Path(root)
    (root / "manifest.json").write_text(
        json.dumps(_manifest(point_in_time_fundamentals=point_in_time_fundamentals,
                             include_universe=include_universe)), encoding="utf-8")
    dates = pd.bdate_range("2020-01-02", periods=days)
    names = ["AAA", "BBB", "CCC"]
    rows = []
    for i, date in enumerate(dates):
        for j, symbol in enumerate(names):
            rows.append({"date": date.date().isoformat(), "symbol": symbol,
                         "close": 100 + j * 10 + i * (0.04 + j * 0.01),
                         "volume": 1_000_000 + j * 100_000})
    pd.DataFrame(rows).to_csv(root / "prices.csv", index=False)
    if include_universe:
        membership = pd.DataFrame({
            "date": [date.date().isoformat() for date in dates for _ in names],
            "symbol": names * len(dates),
            "eligible": [not (date == dates[10] and symbol == "CCC")
                         for date in dates for symbol in names],
        })
        membership.to_csv(root / "universe.csv", index=False)
    if point_in_time_fundamentals:
        facts = []
        for date in dates[20::63]:
            period_end = (date - pd.Timedelta(days=45)).date().isoformat()
            available_date = date.date().isoformat()
            for j, symbol in enumerate(names):
                scale = 1 + j * 0.2  # the defect under test: one scale for every metric
                for metric, value in {
                    "revenue": 3_000_000_000 * scale,
                    "cogs": 1_700_000_000 * scale,
                    "net_income": 400_000_000 * scale,
                    "shares_diluted": 100_000_000 * scale,
                    "total_assets": 5_000_000_000 * scale,
                    "common_equity": 2_500_000_000 * scale,
                    "current_debt": 200_000_000 * scale,
                    "long_term_debt": 700_000_000 * scale,
                    "op_cash_flow": 550_000_000 * scale,
                    "capex": 120_000_000 * scale,
                    "ebit": 600_000_000 * scale,
                    "sga": 250_000_000 * scale,
                    "interest_expense": 30_000_000 * scale,
                }.items():
                    facts.append({"symbol": symbol, "period_end": period_end,
                                  "available_date": available_date,
                                  "metric": metric, "value": value})
        pd.DataFrame(facts).to_csv(root / "fundamentals.csv", index=False)
