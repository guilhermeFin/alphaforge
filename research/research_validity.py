"""Data-health checks that stop a degenerate sample from reading as a result.

A backtest over a universe whose signal never varies still produces an equity
curve, a drawdown series, a report and a verdict. Every number in it is
arithmetically correct and the whole thing is meaningless: nothing was ever
traded, so there is no performance to evaluate.

The observed case is the three-symbol licensed test bundle, where every
fundamental metric is scaled by one per-symbol constant. Ratio factors such as
gross profitability are then identical across all names, the cross-section has
no dispersion, positions net to nothing, and returns are flat at zero.

These helpers answer the prior question -- was there enough variation to
support any performance claim at all? -- and say so in one plain sentence.
They describe the sample only; they never adjust a metric or a verdict.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

#: Share of dates that must show cross-sectional variation before a signal is
#: treated as varying at all.
MIN_VARYING_DATE_SHARE = 0.10

STATUS_INSUFFICIENT = "insufficient_variation"
STATUS_LIMITED = "limited_variation"
STATUS_OK = "sufficient_variation"


def _finite(values) -> np.ndarray:
    array = np.asarray(pd.Series(values, dtype="float64"), dtype=float)
    return array[np.isfinite(array)]


def return_health(returns) -> dict:
    """Count what a return series actually contains before anything is derived."""
    values = _finite(returns)
    size = int(values.size)
    non_flat = int(np.count_nonzero(values != 0.0))
    variance = float(np.var(values, ddof=1)) if size > 1 else float("nan")
    is_flat = bool(size > 1 and np.isfinite(variance) and variance <= 0.0)
    return {
        "effective_observations": size,
        "non_flat_returns": non_flat,
        "flat_returns": size - non_flat,
        "distinct_return_values": int(np.unique(values).size) if size else 0,
        "return_variance": variance if np.isfinite(variance) else None,
        "is_flat": is_flat,
    }


def signal_dispersion(signal) -> dict:
    """Cross-sectional dispersion of a signal panel, or spread of a flat series.

    A panel is measured per date across names, because a signal that is
    identical for every name on a date cannot rank anything on that date, no
    matter how it moves through time.
    """
    if signal is None:
        return {"available": False, "reason": "No signal panel was supplied."}
    if isinstance(signal, pd.Series) or (isinstance(signal, np.ndarray) and signal.ndim == 1):
        values = _finite(signal)
        if values.size < 2:
            return {"available": False, "reason": "Fewer than two finite signal observations."}
        spread = float(np.std(values, ddof=1))
        return {
            "available": True, "shape": "series",
            "mean_dispersion": spread,
            "zero_dispersion_share": 1.0 if spread <= 0.0 else 0.0,
            "dates_measured": int(values.size),
            "is_degenerate": bool(spread <= 0.0),
        }
    frame = pd.DataFrame(signal).astype("float64")
    if frame.empty or frame.shape[1] < 2:
        return {"available": False,
                "reason": "A cross-sectional dispersion check needs at least two names."}
    per_date = frame.std(axis=1, ddof=1)
    measured = per_date.dropna()
    if measured.empty:
        return {"available": False, "reason": "No date had enough names to measure dispersion."}
    zero_share = float((measured <= 0.0).mean())
    return {
        "available": True, "shape": "panel",
        "mean_dispersion": float(measured.mean()),
        "median_dispersion": float(measured.median()),
        "zero_dispersion_share": zero_share,
        "dates_measured": int(measured.size),
        "names": int(frame.shape[1]),
        "average_coverage": float(frame.notna().mean(axis=1).mean()),
        "is_degenerate": bool(zero_share > 1.0 - MIN_VARYING_DATE_SHARE),
    }


def position_health(positions) -> dict:
    """Average active names and concentration, when a position panel exists."""
    if positions is None:
        return {"available": False, "reason": "No position panel was supplied."}
    frame = pd.DataFrame(positions).astype("float64")
    if frame.empty:
        return {"available": False, "reason": "The position panel is empty."}
    absolute = frame.abs().fillna(0.0)
    active = (absolute > 0.0).sum(axis=1)
    gross = absolute.sum(axis=1)
    # Herfindahl on the gross-weight shares: 1.0 is a single-name book.
    with np.errstate(divide="ignore", invalid="ignore"):
        shares = absolute.div(gross, axis=0)
    concentration = (shares**2).sum(axis=1).where(gross > 0.0)
    return {
        "available": True,
        "average_active_names": float(active.mean()),
        "minimum_active_names": int(active.min()),
        "dates_with_no_positions": int((active == 0).sum()),
        "average_gross_exposure": float(gross.mean()),
        "concentration_herfindahl": (
            float(concentration.mean()) if concentration.notna().any() else None
        ),
        "is_degenerate": bool((gross <= 0.0).all()),
    }


def _conclusion(returns: dict, dispersion: dict, positions: dict,
                undefined: list[str]) -> tuple[str, str]:
    """Pick one status and one plain-English sentence a non-specialist can act on."""
    if returns["effective_observations"] < 2:
        return STATUS_INSUFFICIENT, (
            "There are fewer than two usable observations, so no performance claim can be made."
        )
    if returns["is_flat"]:
        detail = "The return series never changes"
        if positions.get("available") and positions.get("is_degenerate"):
            detail += " because no position was ever taken"
        elif dispersion.get("available") and dispersion.get("is_degenerate"):
            detail += " because the signal is identical across every name, so nothing could be ranked"
        return STATUS_INSUFFICIENT, (
            detail + ". This sample has insufficient variation for a performance claim: the "
            "statistics below are undefined rather than weak."
        )
    if dispersion.get("available") and dispersion.get("is_degenerate"):
        return STATUS_INSUFFICIENT, (
            "The signal is identical across names on most dates, so it cannot rank the universe. "
            "Any returns shown come from something other than the signal being tested."
        )
    if positions.get("available") and positions.get("dates_with_no_positions"):
        return STATUS_LIMITED, (
            f"The strategy held no position on {positions['dates_with_no_positions']} dates. "
            "Results describe a partially invested sample."
        )
    if undefined:
        return STATUS_LIMITED, (
            "Some statistics could not be computed for this sample ("
            + ", ".join(sorted(undefined)) + "). They are undefined, not weak."
        )
    if returns["non_flat_returns"] < returns["effective_observations"] * 0.5:
        return STATUS_LIMITED, (
            f"Only {returns['non_flat_returns']} of {returns['effective_observations']} "
            "observations moved at all. Treat dispersion-based statistics with care."
        )
    return STATUS_OK, (
        f"{returns['effective_observations']} usable observations with genuine variation. "
        "This sample can support the statistics below; it does not by itself make them credible."
    )


def _collect_undefined(sources: Iterable[Any] | None) -> list[str]:
    """Pull statistic names out of any statistic_diagnostics blocks handed in."""
    names: list[str] = []
    for source in sources or []:
        if not isinstance(source, dict):
            continue
        for item in source.get("statistic_diagnostics") or []:
            if isinstance(item, dict) and item.get("statistic"):
                names.append(str(item["statistic"]))
    return list(dict.fromkeys(names))


def research_validity_summary(returns, *, positions=None, signal=None,
                              diagnostic_sources: Iterable[Any] | None = None) -> dict:
    """Compact data-health read-out for a strategy report or an event study.

    ``diagnostic_sources`` accepts report sections that already expose
    ``statistic_diagnostics`` (benchmark-relative, statistical rigor), so the
    undefined-statistic flags appear in one place instead of being scattered.
    """
    health = return_health(returns)
    dispersion = signal_dispersion(signal)
    book = position_health(positions)
    undefined = _collect_undefined(diagnostic_sources)
    status, conclusion = _conclusion(health, dispersion, book, undefined)
    return {
        "status": status,
        "conclusion": conclusion,
        "supports_performance_claim": status != STATUS_INSUFFICIENT,
        "returns": health,
        "signal_dispersion": dispersion,
        "positions": book,
        "undefined_statistics": undefined,
        "has_undefined_statistics": bool(undefined),
        "note": (
            "This describes whether the sample contains enough variation to evaluate. It is not a "
            "performance measure and it does not adjust any statistic."
        ),
    }
