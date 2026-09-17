"""Point-in-time FRED/ALFRED macro observations.

FRED's default response answers "what is known today" and may therefore contain
revised historical values.  This module requests output_type=2 (all vintages) and
keeps ``realtime_start`` as each value's availability date.  Consumers can then
join a macro series to a trading calendar without seeing a release or revision
before it was available.
"""
from __future__ import annotations

import json
import os
from typing import Callable
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

MACRO_COLUMNS = ["series_id", "observation_date", "available_date", "vintage_end", "value"]
FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"


def _fetch_json(url: str) -> dict:
    with urlopen(url, timeout=30) as response:  # nosec B310 - fixed HTTPS endpoint
        return json.load(response)


def fred_vintages_to_observations(payload: dict, series_id: str) -> pd.DataFrame:
    """Normalize FRED output_type=2 JSON to explicit point-in-time observations.

    ``realtime_start`` is when that vintage became available.  Missing FRED values
    are denoted by ``."`` and are omitted rather than converted to zero.
    """
    rows = []
    for item in payload.get("observations", []):
        available = item.get("realtime_start")
        observation = item.get("date")
        value = pd.to_numeric(item.get("value"), errors="coerce")
        if available is None or observation is None or pd.isna(value):
            continue
        rows.append((series_id, observation, available, item.get("realtime_end"), float(value)))
    if not rows:
        return pd.DataFrame(columns=MACRO_COLUMNS)
    out = pd.DataFrame(rows, columns=MACRO_COLUMNS)
    for column in ("observation_date", "available_date", "vintage_end"):
        out[column] = pd.to_datetime(out[column], errors="coerce")
    return out.dropna(subset=["observation_date", "available_date", "value"]).sort_values(
        ["available_date", "observation_date"]
    ).reset_index(drop=True)


def macro_asof_panel(
    observations: pd.DataFrame,
    index: pd.DatetimeIndex,
    series_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Return the latest *known* observation of every macro series on each date.

    A revision replaces an older value only at its own ``available_date``.  On a
    given day the most recent observation period wins; for that period, the latest
    available vintage wins.  This daily, date-granular contract deliberately makes
    no claim about availability before a same-day market close.
    """
    required = {"series_id", "observation_date", "available_date", "value"}
    missing = required.difference(observations.columns)
    if missing:
        raise KeyError(f"macro observations missing columns: {sorted(missing)}")
    ids = series_ids or sorted(observations["series_id"].astype(str).unique())
    out = pd.DataFrame(np.nan, index=index, columns=ids, dtype=float)
    data = observations.copy()
    data["observation_date"] = pd.to_datetime(data["observation_date"])
    data["available_date"] = pd.to_datetime(data["available_date"])
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data = data.dropna(subset=["observation_date", "available_date", "value"])

    for date in index:
        known = data[data["available_date"] <= date]
        for series_id in ids:
            rows = known[known["series_id"].astype(str) == series_id]
            if rows.empty:
                continue
            latest_period = rows["observation_date"].max()
            latest = rows[rows["observation_date"] == latest_period]
            latest = latest.sort_values("available_date")
            out.loc[date, series_id] = float(latest.iloc[-1]["value"])
    return out


class FredAlfredProvider:
    """Fetch all FRED/ALFRED vintages for a series using ``FRED_API_KEY``.

    The returned table is intentionally not a backfilled current-value series.
    Pass it to :func:`macro_asof_panel` before modelling.  The public API is called
    only from ``observations``; constructing this object is network-free.
    """

    name = "fred_alfred"
    is_point_in_time = True

    def __init__(
        self,
        api_key: str | None = None,
        fetch_json: Callable[[str], dict] | None = None,
    ):
        self.api_key = api_key
        self._fetch_json = fetch_json or _fetch_json

    def observations(
        self,
        series_id: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        key = (self.api_key or os.environ.get("FRED_API_KEY", "")).strip()
        if not key:
            raise RuntimeError(
                "FRED API key not found. Set $FRED_API_KEY in alphaforge/.env. "
                "Use ALFRED vintages rather than a current-revision FRED export."
            )
        params = {
            "series_id": series_id,
            "api_key": key,
            "file_type": "json",
            "output_type": 2,  # observations by vintage date, all observations
            "realtime_start": "1776-07-04",
            "realtime_end": "9999-12-31",
            "limit": 100000,
        }
        if start is not None:
            params["observation_start"] = start
        if end is not None:
            params["observation_end"] = end
        payload = self._fetch_json(f"{FRED_OBSERVATIONS_URL}?{urlencode(params)}")
        return fred_vintages_to_observations(payload, series_id)
