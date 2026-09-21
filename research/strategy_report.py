"""Composable strategy-report object for AlphaForge tearsheets.

The report has stable, standard inputs even when a specific analytics module does
not use each input yet.  This keeps later attribution, exposure, trade, and
benchmark modules additive instead of forcing an API redesign.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .metrics import TRADING_DAYS
from . import (advanced_econometrics, benchmark, regimes, research_validity, risk,
               statistical_rigor, tail_risk, temporal_stability)


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


@dataclass(frozen=True)
class StrategyReport:
    """A strategy tearsheet assembled from modular analytical sections.

    Parameters are standard across AlphaForge reports: ``returns`` is required;
    ``positions``, ``trades``, and ``benchmark_returns`` are optional because later
    modules consume them. ``candidate_returns`` unlocks search-level CPCV/PBO and
    White Reality Check diagnostics; ``ic_values`` unlocks HAC-corrected IC tests.
    """

    returns: pd.Series
    positions: pd.DataFrame | pd.Series | None = None
    trades: pd.DataFrame | None = None
    benchmark_returns: pd.Series | None = None
    candidate_returns: pd.DataFrame | None = None
    ic_values: pd.Series | None = None
    #: Optional point-in-time signal panel, used only to measure whether the
    #: cross-section varied enough for the strategy to rank anything.
    signal: pd.DataFrame | None = None
    periods_per_year: int = TRADING_DAYS
    include_white_reality_check: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def inputs(self) -> dict:
        """Availability metadata; raw records remain with their owning workflow."""
        return {
            "n_return_observations": int(pd.Series(self.returns).dropna().size),
            "has_positions": self.positions is not None,
            "has_trade_log": self.trades is not None,
            "has_benchmark": self.benchmark_returns is not None,
            "has_candidate_matrix": self.candidate_returns is not None,
            "has_ic_series": self.ic_values is not None,
            "periods_per_year": int(self.periods_per_year),
        }

    def as_dict(self) -> dict:
        """Compose the currently implemented report modules into plain JSON data."""
        benchmark_section = (
            benchmark.benchmark_relative_summary(
                self.returns, self.benchmark_returns, ic_values=self.ic_values,
                periods_per_year=self.periods_per_year)
            if self.benchmark_returns is not None else None
        )
        rigor_section = statistical_rigor.statistical_rigor_summary(
            self.returns, ic_values=self.ic_values,
            candidate_returns=self.candidate_returns,
            periods_per_year=self.periods_per_year,
            include_white_reality_check=self.include_white_reality_check)
        return _jsonable({
            "inputs": self.inputs(),
            # Answered before any metric is read: did this sample vary enough to
            # evaluate at all?
            "research_validity": research_validity.research_validity_summary(
                self.returns, positions=self.positions, signal=self.signal,
                diagnostic_sources=[benchmark_section, rigor_section]),
            "temporal_stability": temporal_stability.chronological_stability_summary(
                self.ic_values if self.ic_values is not None else [], self.returns),
            "volatility_downside_risk": risk.downside_risk_summary(
                self.returns, periods_per_year=self.periods_per_year),
            "benchmark_relative": benchmark_section,
            "tail_risk_distribution": tail_risk.tail_risk_summary(self.returns),
            "statistical_rigor": rigor_section,
            "advanced_econometrics": advanced_econometrics.advanced_econometrics_summary(
                self.returns,
                candidate_returns=self.candidate_returns,
                benchmark_returns=self.benchmark_returns,
                periods_per_year=self.periods_per_year,
            ),
            "regime_analysis": regimes.point_in_time_regime_analysis(
                self.returns, periods_per_year=self.periods_per_year
            ),
            "chart_specs": {
                "volatility_downside_risk": risk.chart_specs(),
                "benchmark_relative": benchmark.chart_specs(),
                "tail_risk_distribution": tail_risk.chart_specs(),
                "statistical_rigor": statistical_rigor.chart_specs(),
                "advanced_econometrics": [],
                "regime_analysis": [],
            },
            "metadata": self.metadata,
        })


def build_strategy_report(
    returns,
    positions=None,
    trades: pd.DataFrame | None = None,
    benchmark_returns=None,
    candidate_returns=None,
    ic_values=None,
    signal=None,
    periods_per_year: int = TRADING_DAYS,
    include_white_reality_check: bool = False,
    metadata: dict[str, Any] | None = None,
) -> dict:
    """Build a JSON-safe strategy report from standard AlphaForge inputs."""
    return StrategyReport(
        returns=pd.Series(returns), positions=positions, trades=trades,
        benchmark_returns=None if benchmark_returns is None else pd.Series(benchmark_returns),
        candidate_returns=None if candidate_returns is None else pd.DataFrame(candidate_returns),
        ic_values=None if ic_values is None else pd.Series(ic_values),
        signal=None if signal is None else pd.DataFrame(signal),
        periods_per_year=periods_per_year,
        include_white_reality_check=include_white_reality_check,
        metadata=metadata or {},
    ).as_dict()
