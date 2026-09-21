"""Decision evidence gates for AlphaForge research.

These checks deliberately classify evidence, never instruments or trades.  A
strong backtest on weak data remains exploratory; that distinction belongs in
the result object, not in a footnote.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def data_integrity_audit(provider: str, close: pd.DataFrame, requested_symbols: list[str],
                         licensed_status: dict[str, Any] | None = None) -> dict:
    """Describe what the selected data can and cannot support."""
    requested = list(dict.fromkeys(str(s).upper() for s in requested_symbols))
    observed = [str(s) for s in close.columns]
    available = close.notna()
    per_symbol = available.mean(axis=0) if not close.empty else pd.Series(dtype=float)
    facts = {
        "provider": provider,
        "requested_symbols": len(requested) if requested else len(observed),
        "observed_symbols": len(observed),
        "date_coverage": {
            "start": str(close.index.min().date()) if len(close.index) else None,
            "end": str(close.index.max().date()) if len(close.index) else None,
            "trading_days": int(len(close)),
        },
        "panel_coverage": float(available.to_numpy().mean()) if close.size else 0.0,
        "minimum_symbol_coverage": float(per_symbol.min()) if not per_symbol.empty else 0.0,
        "missing_price_cells": int((~available).sum().sum()) if close.size else 0,
        "point_in_time_fundamentals": False,
        "historical_universe_membership": False,
        "delisted_securities": False,
        "corporate_actions_adjusted": False,
        "status": "exploratory",
        "reasons": [],
    }
    if provider == "synthetic":
        facts["status"] = "exploratory"
        facts["reasons"].append("Synthetic data validates the engine only; it cannot establish a live-investable result.")
    elif provider == "yfinance":
        facts["status"] = "exploratory"
        facts["corporate_actions_adjusted"] = True
        facts["reasons"].append("Current-ticker public data does not attest to historical membership or delisted securities.")
    elif provider == "licensed_bundle":
        status = licensed_status or {}
        facts.update({
            "point_in_time_fundamentals": bool(status.get("point_in_time_fundamentals")),
            "historical_universe_membership": bool(status.get("survivorship_free_universe")),
            "delisted_securities": bool(status.get("includes_delisted_securities")),
            "corporate_actions_adjusted": bool(status.get("prices_adjusted_for_corporate_actions")),
        })
        issues = list(status.get("issues") or [])
        if (facts["historical_universe_membership"] and facts["delisted_securities"]
                and facts["corporate_actions_adjusted"] and facts["panel_coverage"] >= 0.95):
            facts["status"] = "supported"
        else:
            facts["status"] = "limited"
            if not facts["historical_universe_membership"]:
                issues.append("Historical universe membership is not attested.")
            if not facts["delisted_securities"]:
                issues.append("Delisted-security coverage is not attested.")
            if not facts["corporate_actions_adjusted"]:
                issues.append("Corporate-action adjustment is not attested.")
            if facts["panel_coverage"] < 0.95:
                issues.append("The selected price panel has material missing coverage.")
        facts["reasons"] = list(dict.fromkeys(issues))
    else:
        facts["status"] = "limited"
        facts["reasons"].append("The provider has no registered data-integrity policy.")
    return facts


def evidence_card(*, data_audit: dict, scorecard: dict, signal_quality: dict,
                  out_of_sample: dict, walk_forward: dict, pbo: dict,
                  robustness: dict) -> dict:
    """Combine existing honesty checks into a conservative, inspectable status."""
    gates: list[dict] = []
    data_status = data_audit.get("status", "limited")
    gates.append({
        "id": "data", "label": "Data integrity",
        "status": "pass" if data_status == "supported" else "warning",
        "detail": ("Historical-universe, delisting and corporate-action attestations are present."
                   if data_status == "supported" else "; ".join(data_audit.get("reasons") or ["Data is exploratory or limited."]))
    })
    wf_passes = walk_forward.get("passes") if "error" not in walk_forward else False
    oos_ok = not out_of_sample.get("overfit_warning", True)
    gates.append({
        "id": "out_of_sample", "label": "Out-of-sample validation",
        "status": "pass" if wf_passes and oos_ok else "fail",
        "detail": ("Walk-forward validation passed without an in-sample to out-of-sample decay warning."
                   if wf_passes and oos_ok else "The walk-forward or holdout check did not clear its bar.")
    })
    ic_t = signal_quality.get("ic_tstat")
    ic_ok = ic_t is not None and np.isfinite(ic_t) and float(ic_t) >= 1.96
    gates.append({
        "id": "signal", "label": "Signal quality", "status": "pass" if ic_ok else "warning",
        "detail": (f"Cross-sectional IC t-stat is {float(ic_t):.2f}." if ic_t is not None else
                   "Signal IC could not be estimated with enough observations.")
    })
    pbo_value = pbo.get("pbo")
    pbo_ok = pbo_value is not None and np.isfinite(pbo_value) and float(pbo_value) <= 0.50
    gates.append({
        "id": "multiple_testing", "label": "Multiple-testing check",
        "status": "pass" if pbo_ok else "warning",
        "detail": (f"PBO is {float(pbo_value):.1%}." if pbo_value is not None else
                   "PBO was unavailable for this configuration; it is not treated as a pass.")
    })
    robust_pass = robustness.get("status") == "stable"
    gates.append({
        "id": "stress", "label": "Cost and delay stress", "status": "pass" if robust_pass else "warning",
        "detail": robustness.get("summary", "Stress scenarios were unavailable.")
    })
    if data_status != "supported":
        overall, headline = "exploratory", "Exploratory evidence"
    elif all(g["status"] == "pass" for g in gates):
        overall, headline = "research_supported", "Research-supported evidence"
    elif any(g["status"] == "fail" for g in gates):
        overall, headline = "inconclusive", "Inconclusive evidence"
    else:
        overall, headline = "needs_review", "Evidence needs review"
    return {
        "status": overall, "headline": headline, "gates": gates,
        "note": "This is a research-evidence classification, not an investment recommendation.",
        "full_sample_deflated_sharpe": scorecard.get("deflated_sr"),
    }
