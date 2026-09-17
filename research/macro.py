"""Point-in-time FRED/ALFRED macro observations.

FRED's default response answers "what is known today" and may therefore contain
revised historical values. This module requests output_type=1 over the full
real-time period; it returns one record per value revision and keeps
``realtime_start`` as each value's availability date. Consumers can then join a
macro series to a trading calendar without seeing a release or revision before it
was available.
"""
from __future__ import annotations

import os
from typing import Callable
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

from .http import decode_json_bytes

MACRO_COLUMNS = ["series_id", "observation_date", "available_date", "vintage_end", "value"]
FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_PAGE_SIZE = 100_000
FRED_REALTIME_CHUNK_DAYS = 365


def _fetch_json(url: str) -> dict:
    try:
        with urlopen(url, timeout=30) as response:  # nosec B310 - fixed HTTPS endpoint
            return decode_json_bytes(response.read(), response.headers.get("Content-Encoding"))
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace").lower()
        if error.code == 400 and "api_key" in body and "not registered" in body:
            raise RuntimeError(
                "FRED rejected FRED_API_KEY as unregistered. Replace FRED_API_KEY in .env "
                "with an active key from https://fred.stlouisfed.org/docs/api/api_key.html."
            ) from error
        raise


def fred_vintages_to_observations(payload: dict, series_id: str) -> pd.DataFrame:
    """Normalize FRED output_type=2 JSON to explicit point-in-time observations.

    ``realtime_start`` is when that vintage became available.  Missing FRED values
    are denoted by ``."`` and are omitted rather than converted to zero.
    """
    items = payload.get("observations", [])
    required = {"date", "realtime_start", "value"}
    if any(not required.issubset(item) for item in items):
        raise ValueError(
            "FRED returned an unexpected observation shape; expected output_type=1 "
            "records with date, realtime_start, and value."
        )
    rows = []
    for item in items:
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
    return (
        out.dropna(subset=["observation_date", "available_date", "value"])
        .drop_duplicates()
        .sort_values(["available_date", "observation_date"])
        .reset_index(drop=True)
    )


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
    """Fetch FRED/ALFRED real-time periods for a series using ``FRED_API_KEY``.

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

    def _fetch_pages(self, params: dict[str, object]) -> list[dict]:
        rows: list[dict] = []
        offset = 0
        while True:
            page_params = {**params, "offset": offset}
            payload = self._fetch_json(f"{FRED_OBSERVATIONS_URL}?{urlencode(page_params)}")
            page_rows = payload.get("observations", [])
            rows.extend(page_rows)
            total = int(payload.get("count", len(page_rows)))
            if not page_rows or offset + len(page_rows) >= total:
                return rows
            offset += len(page_rows)

    def _fetch_realtime_chunks(
        self,
        params: dict[str, object],
        start: str | None,
        end: str | None,
    ) -> list[dict]:
        """Fetch high-frequency series in bounded real-time windows.

        FRED rejects a single all-vintages response for certain daily series. A
        yearly real-time window preserves the same availability data while keeping
        each response within the provider's practical response-size limits.
        """
        first = pd.Timestamp(start or "1776-07-04")
        last = pd.Timestamp(end or pd.Timestamp.now().date())
        rows: list[dict] = []
        while first <= last:
            window_end = min(first + pd.Timedelta(days=FRED_REALTIME_CHUNK_DAYS - 1), last)
            rows.extend(
                self._fetch_pages(
                    {
                        **params,
                        "realtime_start": str(first.date()),
                        "realtime_end": str(window_end.date()),
                    }
                )
            )
            first = window_end + pd.Timedelta(days=1)
        return rows

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
            "output_type": 1,  # observations by real-time period, including revisions
            "realtime_start": "1776-07-04",
            "realtime_end": "9999-12-31",
            "limit": FRED_PAGE_SIZE,
        }
        if start is not None:
            params["observation_start"] = start
        if end is not None:
            params["observation_end"] = end
        try:
            rows = self._fetch_pages(params)
        except HTTPError as error:
            if error.code != 400:
                raise
            rows = self._fetch_realtime_chunks(params, start, end)
        return fred_vintages_to_observations({"observations": rows}, series_id)
