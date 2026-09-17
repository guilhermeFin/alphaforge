"""Vendor-agnostic fundamental data providers — the paid-data drop-in layer.

AlphaForge's paid tiers are defined by DATA QUALITY:
    free (synthetic + yfinance)  ->  SimFin (budget)  ->  Sharadar SF1 (PIT)
    ->  S&P Capital IQ / Compustat (institutional).

Every provider here returns the SAME long "observation" table:

    columns = [symbol, period_end, available_date, metric, value]

which is exactly what ``research.fundamentals.build_fundamentals`` already
consumes (and what ``data.make_synthetic_fundamentals`` already emits). So the day
a paid key is added, NOTHING downstream changes — the provider is selected by
name and the point-in-time panels, factors, backtester and honesty layer all run
unchanged. This is how you build for a vendor you cannot afford yet: the seam is
the table, not the vendor.

Adapters are KEY-GATED. With no API key they raise a clear, actionable error and
the engine stays on the synthetic/yfinance path. No network call happens at import
time or in the test suite — the mapping logic is a PURE function
(``vendor_frame_to_obs``) unit-tested against tiny in-memory vendor frames.

POINT-IN-TIME honesty — the whole reason this layer exists:
  * Sharadar SF1 exposes ``datekey`` = the date a filing first became publicly
    known, and a ``dimension`` distinguishing AS-REPORTED (ARQ/ART) from later
    RESTATED (MRQ/MRT) figures. We use AS-REPORTED + ``datekey`` -> genuine
    point-in-time, with no restatement leak. This is the honest tier.
  * SimFin's STANDARD datasets carry the latest RESTATED values (NOT point-in-
    time — per SimFin's own docs). The most honest proxy is to set
    ``available_date`` = Publish Date (the 10-K/10-Q date), and we flag loudly
    that the *values* themselves may be restated. It is a budget tier, labelled
    as such — never silently passed off as true PIT.

Canonical metric names (vendor-agnostic) are the keys of ``CANONICAL_FIELDS``.
Factor code (research/fundamentals.py + the factor library) reads these names, so
swapping SimFin for Sharadar for Compustat never touches the factor math.
"""
from __future__ import annotations

import os
import time
from typing import Callable, Protocol, runtime_checkable
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from .http import decode_json_bytes

# --------------------------------------------------------------------------
# Canonical, vendor-agnostic raw fields. Factor formulas (Piotroski, Altman,
# Sloan accruals, gross profitability, value ratios ...) are written against
# THESE names. Compustat mnemonics are given as the stable academic reference.
# --------------------------------------------------------------------------
CANONICAL_FIELDS: dict[str, str] = {
    # income statement
    "revenue": "SALE",
    "cogs": "COGS",
    "gross_profit": "GP",
    "sga": "XSGA",
    "rnd": "XRD",
    "ebit": "EBIT",
    "interest_expense": "XINT",
    "net_income": "NI",            # IB (income before extraordinary) in some libs
    "shares_diluted": "CSHO",
    # balance sheet
    "total_assets": "AT",
    "current_assets": "ACT",
    "cash": "CHE",
    "receivables": "RECT",
    "inventory": "INVT",
    "current_liabilities": "LCT",
    "accounts_payable": "AP",
    "current_debt": "DLC",
    "long_term_debt": "DLTT",
    "total_liabilities": "LT",
    "common_equity": "CEQ",
    "retained_earnings": "RE",
    # cash-flow statement
    "op_cash_flow": "OANCF",
    "capex": "CAPX",
}
OBS_COLUMNS = ["symbol", "period_end", "available_date", "metric", "value"]

# SEC company-facts tag aliases.  A reporting taxonomy can change over time, so
# each canonical field may have multiple standard US-GAAP names.  We use the first
# available tag and retain its filing date rather than backdating it to period end.
SEC_XBRL_TAGS: dict[str, tuple[str, ...]] = {
    "revenue": ("RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet", "Revenues"),
    "cogs": ("CostOfRevenue", "CostOfGoodsAndServicesSold"),
    "gross_profit": ("GrossProfit",),
    "sga": ("SellingGeneralAndAdministrativeExpense",),
    "rnd": ("ResearchAndDevelopmentExpense",),
    "ebit": ("OperatingIncomeLoss",),
    "interest_expense": ("InterestExpenseNonoperating", "InterestExpense"),
    "net_income": ("NetIncomeLoss",),
    "shares_diluted": ("WeightedAverageNumberOfDilutedSharesOutstanding",),
    "total_assets": ("Assets",),
    "current_assets": ("AssetsCurrent",),
    "cash": ("CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"),
    "receivables": ("AccountsReceivableNetCurrent",),
    "inventory": ("InventoryNet",),
    "current_liabilities": ("LiabilitiesCurrent",),
    "accounts_payable": ("AccountsPayableCurrent",),
    "current_debt": ("ShortTermBorrowings", "ShortTermDebt"),
    "long_term_debt": ("LongTermDebtCurrent", "LongTermDebtNoncurrent"),
    "total_liabilities": ("Liabilities",),
    "common_equity": ("StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
    "retained_earnings": ("RetainedEarningsAccumulatedDeficit",),
    "op_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
    "capex": ("PaymentsToAcquirePropertyPlantAndEquipment",),
}
SEC_REGULAR_FORMS = {"10-K", "10-Q", "20-F", "40-F"}
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# The SEC current-ticker directory maps XOM to a 2026 successor filing entity.
# Its predecessor CIK holds the public company-facts history before that change.
# Keep both sources so the canonical XOM panel does not silently begin in 2026.
SEC_TICKER_PREDECESSOR_CIKS: dict[str, tuple[int, ...]] = {"XOM": (34_088,)}


# --------------------------------------------------------------------------
# Vendor column maps: {canonical_metric -> vendor_column_name}.
# Confirm exact strings against the vendor schema before going live (SimFin:
# `from simfin.names import *`; Sharadar SF1 table docs). Unmapped concepts are
# simply skipped — partial coverage is fine and honest (the factor that needs a
# missing field returns NaN rather than a fabricated value).
# --------------------------------------------------------------------------
SIMFIN_INCOME_MAP = {
    "revenue": "Revenue",
    "cogs": "Cost of Revenue",
    "gross_profit": "Gross Profit",
    "sga": "Selling, General & Administrative",
    "rnd": "Research & Development",
    "ebit": "Operating Income (Loss)",
    "interest_expense": "Interest Expense, Net",
    "net_income": "Net Income",
    "shares_diluted": "Shares (Diluted)",
}
SIMFIN_BALANCE_MAP = {
    "total_assets": "Total Assets",
    "current_assets": "Total Current Assets",
    "cash": "Cash, Cash Equivalents & Short Term Investments",
    "receivables": "Accounts & Notes Receivable",
    "inventory": "Inventories",
    "current_liabilities": "Total Current Liabilities",
    "accounts_payable": "Payables & Accruals",
    "current_debt": "Short Term Debt",
    "long_term_debt": "Long Term Debt",
    "total_liabilities": "Total Liabilities",
    "common_equity": "Total Equity",
    "retained_earnings": "Retained Earnings",
}
SIMFIN_CASHFLOW_MAP = {
    "op_cash_flow": "Net Cash from Operating Activities",
    "capex": "Change in Fixed Assets & Intangibles",
}

# Sharadar SF1 uses lowercase short codes in a single table.
SHARADAR_SF1_MAP = {
    "revenue": "revenue",
    "cogs": "cor",
    "gross_profit": "gp",
    "sga": "sgna",
    "rnd": "rnd",
    "ebit": "opinc",
    "interest_expense": "intexp",
    "net_income": "netinc",
    "shares_diluted": "shareswadil",
    "total_assets": "assets",
    "current_assets": "assetsc",
    "cash": "cashneq",
    "receivables": "receivables",
    "inventory": "inventory",
    "current_liabilities": "liabilitiesc",
    "accounts_payable": "payables",
    "current_debt": "debtc",
    "long_term_debt": "debtnc",
    "total_liabilities": "liabilities",
    "common_equity": "equity",
    "retained_earnings": "retearn",
    "op_cash_flow": "ncfo",
    "capex": "capex",
}


# --------------------------------------------------------------------------
# PURE CORE — the unit-tested heart. No network, no vendor SDK. Turns ANY wide
# vendor frame into the canonical long obs table, honestly carrying the
# available_date and dropping NaNs (a missing field is absent, never zero).
# --------------------------------------------------------------------------
def vendor_frame_to_obs(
    df: pd.DataFrame,
    *,
    field_map: dict[str, str],
    symbol_col: str,
    period_end_col: str,
    available_date_col: str,
) -> pd.DataFrame:
    """Melt a wide vendor frame into [symbol, period_end, available_date, metric, value].

    ``field_map`` maps canonical metric names -> the vendor's column names. Only
    mapped columns that actually exist in ``df`` are emitted. Rows with a NaN
    value, NaT period_end, or NaT available_date are dropped (you cannot trade on
    a number you never had). The result is sorted by available_date so the
    downstream point-in-time placement is deterministic.
    """
    present = {canon: col for canon, col in field_map.items() if col in df.columns}
    if not present:
        return pd.DataFrame(columns=OBS_COLUMNS)
    for required in (symbol_col, period_end_col, available_date_col):
        if required not in df.columns:
            raise KeyError(f"vendor frame missing required column {required!r}")

    base = df[[symbol_col, period_end_col, available_date_col]].copy()
    base.columns = ["symbol", "period_end", "available_date"]
    base["period_end"] = pd.to_datetime(base["period_end"], errors="coerce")
    base["available_date"] = pd.to_datetime(base["available_date"], errors="coerce")

    frames = []
    for canon, col in present.items():
        part = base.copy()
        part["metric"] = canon
        part["value"] = pd.to_numeric(df[col].to_numpy(), errors="coerce")
        frames.append(part)

    obs = pd.concat(frames, ignore_index=True)[OBS_COLUMNS]
    obs = obs.dropna(subset=["period_end", "available_date", "value"])
    obs["symbol"] = obs["symbol"].astype(str)
    return obs.sort_values("available_date").reset_index(drop=True)


# --------------------------------------------------------------------------
# Provider protocol + adapters.
# --------------------------------------------------------------------------
@runtime_checkable
class FundamentalProvider(Protocol):
    """A source of point-in-time fundamentals. ``is_point_in_time`` tells the UI
    whether the values are genuinely as-reported (Sharadar) or restated (SimFin)."""

    name: str
    is_point_in_time: bool

    def fundamentals(
        self, symbols: list[str], start: str | None = None, end: str | None = None
    ) -> pd.DataFrame:  # returns the canonical OBS_COLUMNS table
        ...


def _require_key(env_var: str, vendor: str, signup: str) -> str:
    key = os.environ.get(env_var, "").strip()
    if not key:
        raise RuntimeError(
            f"{vendor} API key not found. Set ${env_var} (e.g. in alphaforge/.env). "
            f"Get a key at {signup}. Until then, use provider='synthetic' (free)."
        )
    return key


def _sec_fetch_json(url: str, headers: dict[str, str]) -> dict:
    request = Request(url, headers=headers)
    with urlopen(request, timeout=30) as response:  # nosec B310 - fixed SEC HTTPS endpoints
        return decode_json_bytes(response.read(), response.headers.get("Content-Encoding"))


def sec_ticker_map(payload: dict) -> dict[str, int]:
    """Normalize the SEC ticker payload to ``{TICKER: CIK}`` without network I/O."""
    values = payload.values() if isinstance(payload, dict) else payload
    out = {}
    for row in values:
        ticker, cik = row.get("ticker"), row.get("cik_str")
        if ticker is not None and cik is not None:
            out[str(ticker).upper()] = int(cik)
    return out


def sec_company_facts_to_obs(payload: dict, symbol: str) -> pd.DataFrame:
    """Turn SEC Company Facts JSON into the canonical as-filed observation table.

    ``filed`` is date-granular, so this adapter is suitable for daily strategies
    only.  The same period can occur again in a later filing; that later value is
    retained as a newly available revision rather than overwriting prior history.
    """
    facts = payload.get("facts", {}).get("us-gaap", {})
    rows = []
    for metric, aliases in SEC_XBRL_TAGS.items():
        concept = next((facts[tag] for tag in aliases if tag in facts), None)
        if concept is None:
            continue
        units = concept.get("units", {})
        preferred_units = ("shares", "pure") if metric == "shares_diluted" else ("USD",)
        items = next((units[unit] for unit in preferred_units if unit in units), [])
        for item in items:
            if item.get("form") not in SEC_REGULAR_FORMS:
                continue
            period_end, filed, value = item.get("end"), item.get("filed"), item.get("val")
            if period_end is None or filed is None or value is None:
                continue
            rows.append((symbol.upper(), period_end, filed, metric, value))
    if not rows:
        return pd.DataFrame(columns=OBS_COLUMNS)
    out = pd.DataFrame(rows, columns=OBS_COLUMNS)
    out["period_end"] = pd.to_datetime(out["period_end"], errors="coerce")
    out["available_date"] = pd.to_datetime(out["available_date"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.dropna(subset=["period_end", "available_date", "value"]).sort_values(
        ["available_date", "period_end", "metric"]
    ).reset_index(drop=True)


class SimFinProvider:
    """SimFin budget tier. Values are LATEST-RESTATED (not true PIT); we anchor
    ``available_date`` to the Publish Date as the honest proxy and flag it.

    Free tier exists (delayed bulk data): https://simfin.com/data/api
    Requires the ``simfin`` package and $SIMFIN_API_KEY.
    """

    name = "simfin"
    is_point_in_time = False  # standard datasets carry restated figures — flagged in UI

    def __init__(self, variant: str = "quarterly", market: str = "us"):
        self.variant = variant
        self.market = market

    def fundamentals(self, symbols, start=None, end=None) -> pd.DataFrame:  # pragma: no cover - needs key+network
        key = _require_key("SIMFIN_API_KEY", "SimFin", "https://simfin.com/data/api")
        try:
            import simfin as sf
            from simfin.names import TICKER, REPORT_DATE, PUBLISH_DATE
        except ImportError as e:
            raise ImportError("simfin not installed. `pip install alphaforge[vendors]`") from e

        sf.set_api_key(key)
        sf.set_data_dir(os.environ.get("SIMFIN_DATA_DIR", os.path.expanduser("~/simfin_data/")))
        idx = [TICKER, REPORT_DATE]
        income = sf.load_income(variant=self.variant, market=self.market).reset_index()
        balance = sf.load_balance(variant=self.variant, market=self.market).reset_index()
        cashflow = sf.load_cashflow(variant=self.variant, market=self.market).reset_index()

        wanted = set(s.upper() for s in symbols)
        out = []
        for df, fmap in ((income, SIMFIN_INCOME_MAP),
                         (balance, SIMFIN_BALANCE_MAP),
                         (cashflow, SIMFIN_CASHFLOW_MAP)):
            df = df[df[TICKER].str.upper().isin(wanted)]
            out.append(vendor_frame_to_obs(
                df, field_map=fmap, symbol_col=TICKER,
                period_end_col=REPORT_DATE, available_date_col=PUBLISH_DATE))
        obs = pd.concat(out, ignore_index=True)
        if start is not None:
            obs = obs[obs["available_date"] >= pd.Timestamp(start)]
        if end is not None:
            obs = obs[obs["available_date"] <= pd.Timestamp(end)]
        return obs.sort_values("available_date").reset_index(drop=True)


class SharadarProvider:
    """Sharadar SF1 (Nasdaq Data Link) — genuine point-in-time tier.

    Uses dimension AS-REPORTED (``ARQ`` quarterly / ``ART`` trailing-twelve) and
    ``datekey`` (the date the filing became public) as ``available_date`` — no
    restatement leak. Requires ``nasdaq-data-link`` and $NASDAQ_DATA_LINK_API_KEY.
    """

    name = "sharadar"
    is_point_in_time = True

    def __init__(self, dimension: str = "ARQ"):
        if dimension not in ("ARQ", "ART", "ARY"):
            raise ValueError("use an AS-REPORTED dimension (ARQ/ART/ARY) for point-in-time")
        self.dimension = dimension

    def fundamentals(self, symbols, start=None, end=None) -> pd.DataFrame:  # pragma: no cover - needs key+network
        key = _require_key("NASDAQ_DATA_LINK_API_KEY", "Sharadar/Nasdaq Data Link",
                           "https://data.nasdaq.com/databases/SF1")
        try:
            import nasdaqdatalink
        except ImportError as e:
            raise ImportError("nasdaq-data-link not installed. `pip install alphaforge[vendors]`") from e

        nasdaqdatalink.ApiConfig.api_key = key
        params = {"dimension": self.dimension, "ticker": [s.upper() for s in symbols]}
        if start is not None:
            params["datekey"] = {"gte": str(start)}
        df = nasdaqdatalink.get_table("SHARADAR/SF1", paginate=True, **params)
        if end is not None and not df.empty:
            df = df[pd.to_datetime(df["datekey"]) <= pd.Timestamp(end)]
        return vendor_frame_to_obs(
            df, field_map=SHARADAR_SF1_MAP, symbol_col="ticker",
            period_end_col="reportperiod", available_date_col="datekey")


class SecEdgarProvider:
    """Free SEC Company Facts fundamentals with filing-date availability.

    It sends the declared ``SEC_USER_AGENT`` on every request and spaces requests
    below the SEC's published ten-requests-per-second limit.  The SEC field is a
    public filing date, not an intraday dissemination timestamp.
    """

    name = "sec_edgar"
    is_point_in_time = True

    def __init__(
        self,
        user_agent: str | None = None,
        fetch_json: Callable[[str, dict[str, str]], dict] | None = None,
        request_interval: float = 0.12,
    ):
        self.user_agent = user_agent
        self._fetch_json = fetch_json or _sec_fetch_json
        self.request_interval = request_interval
        self._last_request_at: float | None = None

    def _get(self, url: str) -> dict:
        user_agent = (self.user_agent or os.environ.get("SEC_USER_AGENT", "")).strip()
        if not user_agent or "example.com" in user_agent:
            raise RuntimeError(
                "SEC user agent not configured. Set $SEC_USER_AGENT to a descriptive "
                "application name and monitored contact email in alphaforge/.env."
            )
        if self._last_request_at is not None:
            remaining = self.request_interval - (time.monotonic() - self._last_request_at)
            if remaining > 0:
                time.sleep(remaining)
        payload = self._fetch_json(url, {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self._last_request_at = time.monotonic()
        return payload

    def fundamentals(self, symbols, start=None, end=None) -> pd.DataFrame:
        tickers = sec_ticker_map(self._get(SEC_TICKERS_URL))
        frames = []
        for symbol in symbols:
            normalized_symbol = symbol.upper()
            ciks = list(SEC_TICKER_PREDECESSOR_CIKS.get(normalized_symbol, ()))
            current_cik = tickers.get(normalized_symbol)
            if current_cik is not None:
                ciks.append(current_cik)
            if not ciks:
                continue
            for cik in dict.fromkeys(ciks):
                facts = self._get(SEC_COMPANY_FACTS_URL.format(cik=cik))
                frames.append(sec_company_facts_to_obs(facts, normalized_symbol))
        if not frames:
            return pd.DataFrame(columns=OBS_COLUMNS)
        obs = pd.concat(frames, ignore_index=True)
        if start is not None:
            obs = obs[obs["available_date"] >= pd.Timestamp(start)]
        if end is not None:
            obs = obs[obs["available_date"] <= pd.Timestamp(end)]
        return obs.sort_values("available_date").reset_index(drop=True)


class SyntheticFundamentalProvider:
    """Free offline tier — wraps the synthetic World generator so the SAME code
    path (provider -> obs table -> build_fundamentals) is exercised without a key.

    Note: the current synthetic generator emits derived ratio-metrics (roe, eps_ttm,
    ...), not the raw canonical fields. It is point-in-time-correct by construction
    (available_date = period_end + reporting lag). Raw-field synthetic generation
    is a follow-up so the full factor library can run offline too.
    """

    name = "synthetic"
    is_point_in_time = True

    def __init__(self, periods: int = 1512, seed: int = 0, reporting_lag_days: int = 75):
        self.periods, self.seed, self.reporting_lag_days = periods, seed, reporting_lag_days

    def fundamentals(self, symbols, start="2015-01-02", end=None) -> pd.DataFrame:
        from . import data
        world = data.make_synthetic_world(symbols, periods=self.periods, start=start, seed=self.seed)
        return data.make_synthetic_fundamentals(
            world, reporting_lag_days=self.reporting_lag_days, seed=self.seed)


_PROVIDERS = {
    "synthetic": SyntheticFundamentalProvider,
    "simfin": SimFinProvider,
    "sharadar": SharadarProvider,
    "sec_edgar": SecEdgarProvider,
}


def get_fundamental_provider(name: str = "synthetic", **kwargs) -> FundamentalProvider:
    """Select a provider by name. Adapters are key-gated: constructing one is
    free, but calling ``.fundamentals(...)`` on a paid vendor without its API key
    raises a clear error pointing at the env var and signup URL."""
    if name not in _PROVIDERS:
        raise ValueError(f"unknown provider {name!r}; choose from {sorted(_PROVIDERS)}")
    return _PROVIDERS[name](**kwargs)
