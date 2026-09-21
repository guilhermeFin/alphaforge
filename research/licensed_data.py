"""Vendor-neutral, local intake for a customer's licensed market-data export.

Paid providers differ in APIs and licensing, but a point-in-time research engine
needs the same four facts from each of them: dated prices and volume, membership
of the historical universe, dated fundamental observations, and provenance.  This
module makes that contract explicit without bundling, transmitting, or exposing
any vendor credentials.

Set ``ALPHAFORGE_LICENSED_DATA_PATH`` to a local folder containing:

* ``manifest.json`` -- provenance and point-in-time attestations;
* ``prices.csv`` -- ``date,symbol,close,volume``;
* ``universe.csv`` -- ``date,symbol,eligible`` when the manifest claims a
  survivorship-free universe;
* ``fundamentals.csv`` -- optional canonical observation table used by
  fundamental factors.

The strict validation is intentional.  A customer can supply any vendor export,
but AlphaForge never upgrades its research-quality claims merely because a folder
exists.  Missing membership, dates, or as-reported availability cause an explicit
error or visible readiness warning instead of a silent fallback.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .providers import CANONICAL_FIELDS, OBS_COLUMNS


BUNDLE_ENV = "ALPHAFORGE_LICENSED_DATA_PATH"
MANIFEST_FILE = "manifest.json"
PRICES_FILE = "prices.csv"
UNIVERSE_FILE = "universe.csv"
FUNDAMENTALS_FILE = "fundamentals.csv"
SCHEMA_VERSION = 1

REQUIRED_MANIFEST_FIELDS = (
    "schema_version",
    "provider_name",
    "license_acknowledged",
    "prices_adjusted_for_corporate_actions",
    "point_in_time_fundamentals",
    "survivorship_free_universe",
    "includes_delisted_securities",
)
PRICE_COLUMNS = ("date", "symbol", "close", "volume")
UNIVERSE_COLUMNS = ("date", "symbol", "eligible")


class LicensedDataError(ValueError):
    """User-facing failure when a licensed-data bundle breaks its contract."""


def _configured_path(path: str | Path | None = None) -> Path | None:
    candidate = path if path is not None else os.environ.get(BUNDLE_ENV)
    if candidate is None or not str(candidate).strip():
        return None
    return Path(str(candidate)).expanduser()


def _require_folder(path: str | Path | None = None) -> Path:
    root = _configured_path(path)
    if root is None:
        raise LicensedDataError(
            f"No licensed data bundle is configured. Set {BUNDLE_ENV} to a local bundle folder."
        )
    if not root.is_dir():
        raise LicensedDataError(f"Licensed data bundle folder does not exist: {root}")
    return root


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], file_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise LicensedDataError(f"{file_name} is missing required columns: {', '.join(missing)}")


def _normalise_dates(values: pd.Series, label: str) -> pd.Series:
    dates = pd.to_datetime(values, errors="coerce")
    if dates.isna().any():
        raise LicensedDataError(f"{label} contains an invalid date")
    try:
        return dates.dt.tz_localize(None).dt.normalize()
    except TypeError:  # already timezone-naive
        return dates.dt.normalize()


def _normalise_symbols(values: pd.Series, label: str) -> pd.Series:
    symbols = values.astype(str).str.strip().str.upper()
    if (symbols == "").any():
        raise LicensedDataError(f"{label} contains a blank symbol")
    return symbols


def _manifest(root: Path) -> dict[str, Any]:
    path = root / MANIFEST_FILE
    if not path.is_file():
        raise LicensedDataError(f"Licensed data bundle is missing {MANIFEST_FILE}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LicensedDataError(f"Could not read {MANIFEST_FILE}: {type(error).__name__}") from error
    if not isinstance(payload, dict):
        raise LicensedDataError(f"{MANIFEST_FILE} must contain one JSON object")
    missing = [field for field in REQUIRED_MANIFEST_FIELDS if field not in payload]
    if missing:
        raise LicensedDataError(f"{MANIFEST_FILE} is missing required fields: {', '.join(missing)}")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise LicensedDataError(
            f"Unsupported licensed-data schema version {payload['schema_version']!r}; expected {SCHEMA_VERSION}"
        )
    if not isinstance(payload["provider_name"], str) or not payload["provider_name"].strip():
        raise LicensedDataError("manifest provider_name must be a non-empty string")
    for field in REQUIRED_MANIFEST_FIELDS[2:]:
        if not isinstance(payload[field], bool):
            raise LicensedDataError(f"manifest field {field!r} must be true or false")
    return payload


def load_manifest(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate only the public provenance manifest for a local bundle."""
    return _manifest(_require_folder(path))


def bundle_status(path: str | Path | None = None) -> dict[str, Any]:
    """Return secret-free connection/readiness facts without opening large files."""
    root = _configured_path(path)
    base = {
        "id": "licensed_bundle",
        "label": "Licensed data bundle",
        "configured": root is not None,
        "ready_for_price_research": False,
        "ready_for_fundamental_research": False,
        "provider_name": None,
        "point_in_time_fundamentals": False,
        "survivorship_free_universe": False,
        "includes_delisted_securities": False,
        "prices_adjusted_for_corporate_actions": False,
        "issues": [],
    }
    if root is None:
        base["issues"].append(f"Set {BUNDLE_ENV} when a licensed export is available.")
        return base
    if not root.is_dir():
        base["issues"].append("Configured bundle folder does not exist.")
        return base
    try:
        manifest = _manifest(root)
    except LicensedDataError as error:
        base["issues"].append(str(error))
        return base

    base.update({
        "provider_name": manifest["provider_name"].strip(),
        "point_in_time_fundamentals": manifest["point_in_time_fundamentals"],
        "survivorship_free_universe": manifest["survivorship_free_universe"],
        "includes_delisted_securities": manifest["includes_delisted_securities"],
        "prices_adjusted_for_corporate_actions": manifest["prices_adjusted_for_corporate_actions"],
    })
    has_prices = (root / PRICES_FILE).is_file()
    has_universe = (root / UNIVERSE_FILE).is_file()
    has_fundamentals = (root / FUNDAMENTALS_FILE).is_file()
    if not has_prices:
        base["issues"].append(f"Missing {PRICES_FILE}.")
    if manifest["survivorship_free_universe"] and not has_universe:
        base["issues"].append(f"Manifest claims a survivorship-free universe but {UNIVERSE_FILE} is missing.")
    if not manifest["survivorship_free_universe"]:
        base["issues"].append("Historical universe membership is not attested; current-symbol bias may remain.")
    if not manifest["includes_delisted_securities"]:
        base["issues"].append("Delisted securities are not attested; historical coverage is incomplete.")
    if not manifest["point_in_time_fundamentals"]:
        base["issues"].append("Point-in-time fundamentals are not attested; fundamental factors stay blocked.")
    base["ready_for_price_research"] = bool(has_prices)
    base["ready_for_fundamental_research"] = bool(
        has_prices and has_fundamentals and manifest["point_in_time_fundamentals"]
    )
    return base


def _read_prices(root: Path, symbols: list[str] | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = root / PRICES_FILE
    if not path.is_file():
        raise LicensedDataError(f"Licensed data bundle is missing {PRICES_FILE}")
    try:
        rows = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as error:
        raise LicensedDataError(f"Could not read {PRICES_FILE}: {type(error).__name__}") from error
    _require_columns(rows, PRICE_COLUMNS, PRICES_FILE)
    rows = rows.loc[:, PRICE_COLUMNS].copy()
    rows["date"] = _normalise_dates(rows["date"], PRICES_FILE)
    rows["symbol"] = _normalise_symbols(rows["symbol"], PRICES_FILE)
    rows["close"] = pd.to_numeric(rows["close"], errors="coerce")
    rows["volume"] = pd.to_numeric(rows["volume"], errors="coerce")
    if rows[["close", "volume"]].isna().any().any() or not np.isfinite(rows[["close", "volume"]]).all().all():
        raise LicensedDataError(f"{PRICES_FILE} contains non-numeric or non-finite price/volume values")
    if (rows["close"] <= 0).any() or (rows["volume"] < 0).any():
        raise LicensedDataError(f"{PRICES_FILE} requires positive close and non-negative volume")
    if rows.duplicated(["date", "symbol"]).any():
        raise LicensedDataError(f"{PRICES_FILE} has duplicate date/symbol rows")

    wanted = [str(symbol).strip().upper() for symbol in (symbols or []) if str(symbol).strip()]
    available = set(rows["symbol"])
    missing = sorted(set(wanted) - available)
    if missing:
        raise LicensedDataError(f"Licensed data has no price history for: {', '.join(missing)}")
    if wanted:
        rows = rows[rows["symbol"].isin(wanted)]
    close = rows.pivot(index="date", columns="symbol", values="close").sort_index()
    volume = rows.pivot(index="date", columns="symbol", values="volume").reindex_like(close)
    return close, volume


def _as_bool(values: pd.Series) -> pd.Series:
    normalized = values.astype(str).str.strip().str.lower()
    truthy, falsy = {"true", "1", "yes", "y"}, {"false", "0", "no", "n"}
    invalid = ~normalized.isin(truthy | falsy)
    if invalid.any():
        raise LicensedDataError(f"{UNIVERSE_FILE} eligible must use true/false values")
    return normalized.isin(truthy)


def _apply_eligibility(root: Path, manifest: dict[str, Any], close: pd.DataFrame,
                       volume: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = root / UNIVERSE_FILE
    if not path.is_file():
        if manifest["survivorship_free_universe"]:
            raise LicensedDataError(
                f"Manifest claims a survivorship-free universe, but {UNIVERSE_FILE} is missing"
            )
        return close, volume
    rows = pd.read_csv(path)
    _require_columns(rows, UNIVERSE_COLUMNS, UNIVERSE_FILE)
    rows = rows.loc[:, UNIVERSE_COLUMNS].copy()
    rows["date"] = _normalise_dates(rows["date"], UNIVERSE_FILE)
    rows["symbol"] = _normalise_symbols(rows["symbol"], UNIVERSE_FILE)
    rows["eligible"] = _as_bool(rows["eligible"])
    if rows.duplicated(["date", "symbol"]).any():
        raise LicensedDataError(f"{UNIVERSE_FILE} has duplicate date/symbol rows")
    eligible = rows.pivot(index="date", columns="symbol", values="eligible")
    eligible = eligible.reindex(index=close.index, columns=close.columns).fillna(False).astype(bool)
    return close.where(eligible), volume.where(eligible)


def load_price_panel(symbols: list[str], path: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load an audited local price/volume panel, applying historical eligibility."""
    root = _require_folder(path)
    manifest = _manifest(root)
    close, volume = _read_prices(root, symbols)
    close, volume = _apply_eligibility(root, manifest, close, volume)
    close = close.dropna(how="all", axis=0).sort_index()
    volume = volume.reindex_like(close)
    if close.empty or close.shape[1] < 2:
        raise LicensedDataError("Licensed data needs at least two historically eligible symbols")
    return close, volume


def load_fundamentals(symbols: list[str], path: str | Path | None = None,
                      start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Load canonical, filing-dated fundamental observations from a local bundle."""
    root = _require_folder(path)
    manifest = _manifest(root)
    if not manifest["point_in_time_fundamentals"]:
        raise LicensedDataError(
            "Licensed bundle does not attest to point-in-time fundamentals; fundamental factors are blocked."
        )
    path = root / FUNDAMENTALS_FILE
    if not path.is_file():
        raise LicensedDataError(f"Licensed data bundle is missing {FUNDAMENTALS_FILE}")
    rows = pd.read_csv(path)
    _require_columns(rows, tuple(OBS_COLUMNS), FUNDAMENTALS_FILE)
    rows = rows.loc[:, OBS_COLUMNS].copy()
    rows["symbol"] = _normalise_symbols(rows["symbol"], FUNDAMENTALS_FILE)
    rows["period_end"] = _normalise_dates(rows["period_end"], FUNDAMENTALS_FILE)
    rows["available_date"] = _normalise_dates(rows["available_date"], FUNDAMENTALS_FILE)
    rows["metric"] = rows["metric"].astype(str).str.strip()
    rows["value"] = pd.to_numeric(rows["value"], errors="coerce")
    if rows["value"].isna().any() or not np.isfinite(rows["value"]).all():
        raise LicensedDataError(f"{FUNDAMENTALS_FILE} contains non-numeric or non-finite values")
    unknown = sorted(set(rows["metric"]) - set(CANONICAL_FIELDS))
    if unknown:
        raise LicensedDataError(f"{FUNDAMENTALS_FILE} has unknown canonical metrics: {', '.join(unknown)}")
    if (rows["available_date"] < rows["period_end"]).any():
        raise LicensedDataError(f"{FUNDAMENTALS_FILE} has an available date before its period end")
    if rows.duplicated(["symbol", "period_end", "available_date", "metric"]).any():
        raise LicensedDataError(f"{FUNDAMENTALS_FILE} has duplicate observation rows")
    wanted = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
    rows = rows[rows["symbol"].isin(wanted)]
    if start is not None:
        rows = rows[rows["available_date"] >= pd.Timestamp(start)]
    if end is not None:
        rows = rows[rows["available_date"] <= pd.Timestamp(end)]
    if rows.empty:
        raise LicensedDataError("Licensed bundle has no fundamental observations for the selected symbols/date range")
    return rows.sort_values(["available_date", "symbol", "metric", "period_end"]).reset_index(drop=True)
