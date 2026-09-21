"""Execute immutable chronological research stages from a registered protocol.

The protocol store keeps the audit ledger.  This module adds the missing
operational control: a final-holdout run receives only its declared date window
for evaluation.  Earlier prices remain available solely to form lagged signals
and positions, never to contribute to the reported performance statistics.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from .reproducibility import fingerprint
from .research_protocol import ProtocolError, ResearchProtocolStore, STAGES


MIN_EVALUATION_DAYS = 60


def _date(value: object, label: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(value).normalize()
    except (TypeError, ValueError) as error:
        raise ProtocolError(f"{label} must be an ISO date (YYYY-MM-DD).") from error


def _next_day(value: pd.Timestamp) -> pd.Timestamp:
    """Use the next calendar day; the provider selects the next trading session."""
    return value + pd.Timedelta(days=1)


def _specification(request: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return the immutable strategy assumptions, excluding stage endpoints.

    The date range changes by design as a study advances. Everything that could
    change the tested strategy, its data source, or its reported economics stays
    locked once the first protected stage is executed.
    """
    keys = (
        "provider", "symbols", "factor", "lookback", "skip", "cost_bps",
        "gross", "n_trials", "seed", "start",
    )
    values = {key: request.get(key) for key in keys}
    values["provider"] = str(values["provider"] or "synthetic").lower()
    values["factor"] = str(values["factor"] or "momentum").lower()
    values["symbols"] = sorted({str(item).strip().upper() for item in (values["symbols"] or []) if str(item).strip()})
    values["start"] = _date(values["start"] or "2015-01-02", "request start").date().isoformat()
    return fingerprint(values), values


class ProtocolRunner:
    """Build and record stage-bounded work without exposing the holdout casually."""

    def __init__(self, store: ResearchProtocolStore) -> None:
        self.store = store

    def prepare(self, study_id: str, stage: str, request: dict[str, Any]) -> tuple[dict, dict]:
        """Return a copied request constrained to one protocol evaluation stage.

        The source window begins at the pre-registered request start.  The
        ``evaluation_start`` marker tells the backtest to exclude all earlier
        bars from every reported statistic while retaining legitimate signal
        warm-up.  Synthetic data receives enough deterministic periods to reach
        the declared endpoint exactly.
        """
        if stage not in STAGES:
            raise ProtocolError(f"stage must be one of {STAGES}")
        study = self.store.summary(study_id)
        specification_fingerprint, specification = _specification(request)
        recorded_specs = {
            run.get("metadata", {}).get("specification_fingerprint")
            for run in study["runs"]
            if run.get("metadata", {}).get("specification_fingerprint")
        }
        if len(recorded_specs) > 1:
            raise ProtocolError("The protocol has conflicting recorded specifications; fork it before continuing.")
        if recorded_specs and specification_fingerprint not in recorded_specs:
            raise ProtocolError(
                "This protocol is locked to its first protected strategy specification. "
                "Fork the study before changing factor, universe, costs, or other research assumptions."
            )
        boundaries = {key: _date(value, key) for key, value in study["boundaries"].items()}
        source_start = _date(request.get("start", "2015-01-02"), "request start")
        research_end = boundaries["research_end"]
        if source_start > research_end:
            raise ProtocolError("The request start must be on or before the protocol research end.")

        stage_start, stage_end = {
            "exploration": (source_start, research_end),
            "validation": (_next_day(research_end), boundaries["validation_end"]),
            "final_holdout": (_next_day(boundaries["validation_end"]), boundaries["final_holdout_end"]),
        }[stage]
        evaluation_days = len(pd.bdate_range(stage_start, stage_end))
        if evaluation_days < MIN_EVALUATION_DAYS:
            raise ProtocolError(
                f"The {stage.replace('_', ' ')} window has only {evaluation_days} business days; "
                f"at least {MIN_EVALUATION_DAYS} are required for an honest stage evaluation."
            )
        if stage == "final_holdout" and not study["final_holdout_available"]:
            raise ProtocolError("The final holdout has already been consumed. Fork the study before a new final test.")

        prepared = dict(request)
        prepared["start"] = source_start.date().isoformat()
        prepared["end"] = stage_end.date().isoformat()
        prepared["evaluation_start"] = stage_start.date().isoformat()
        # The source endpoint changes by stage, but it is not a new searched
        # strategy. The shared workflow uses this stable identity for its trial
        # ledger while the manifest retains the full stage-bounded request.
        prepared["_trial_identity"] = specification
        if str(prepared.get("provider", "synthetic")).lower() == "synthetic":
            required_periods = len(pd.bdate_range(source_start, stage_end))
            if required_periods > 6_000:
                raise ProtocolError("The declared protocol window exceeds AlphaForge's 6,000-day synthetic limit.")
            prepared["periods"] = max(120, required_periods)
        window = {
            "source_start": source_start.date().isoformat(),
            "evaluation_start": stage_start.date().isoformat(),
            "evaluation_end": stage_end.date().isoformat(),
            "evaluation_business_days": evaluation_days,
        }
        return prepared, {
            "study": study,
            "window": window,
            "specification": specification,
            "specification_fingerprint": specification_fingerprint,
        }

    def run(
        self,
        study_id: str,
        stage: str,
        request: dict[str, Any],
        execute: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> dict[str, Any]:
        """Execute one protected stage and append its result to the protocol ledger."""
        prepared, context = self.prepare(study_id, stage, request)
        result = execute(prepared)
        manifest = result.get("manifest") or {}
        fingerprint = str(manifest.get("research_fingerprint") or "")
        if not fingerprint:
            raise ProtocolError("The stage did not produce a reproducibility fingerprint and was not recorded.")

        meta = result.get("meta") or {}
        audit = result.get("data_audit") or {}
        actual_days = meta.get("n_days")
        if actual_days is not None and int(actual_days) < MIN_EVALUATION_DAYS:
            raise ProtocolError(
                f"The provider returned only {actual_days} usable sessions for the protected stage; "
                f"at least {MIN_EVALUATION_DAYS} are required and no result was recorded."
            )
        record = self.store.record(study_id, stage, fingerprint, {
            "window": context["window"],
            "specification": context["specification"],
            "specification_fingerprint": context["specification_fingerprint"],
            "request": prepared,
            "result": {
                "verdict": result.get("verdict"),
                "evidence_status": (result.get("evidence_card") or {}).get("status"),
                "data_status": audit.get("status"),
                "actual_start": meta.get("start_date"),
                "actual_end": meta.get("end_date"),
                "annual_sharpe": (result.get("scorecard") or {}).get("ann_sharpe"),
            },
        })
        result["protocol"] = {
            "study_id": study_id,
            "run_id": record["id"],
            "stage": stage,
            "hypothesis": context["study"]["hypothesis"],
            "boundaries": context["study"]["boundaries"],
            "window": context["window"],
            "specification_fingerprint": context["specification_fingerprint"],
            "final_holdout_consumed": stage == "final_holdout",
        }
        if isinstance(result.get("meta"), dict):
            result["meta"]["protocol_stage"] = stage
            result["meta"]["protocol_evaluation_start"] = context["window"]["evaluation_start"]
            result["meta"]["protocol_evaluation_end"] = context["window"]["evaluation_end"]
        return result
