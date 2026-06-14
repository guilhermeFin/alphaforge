"""Factor library over RAW point-in-time fundamentals — value, quality, profitability,
growth, accruals, distress screens, plus price/volume factors.

Every factor here consumes a :class:`research.fundamentals.Fundamentals` bundle (each
metric a point-in-time, dates x symbols panel) plus the price ``close`` (and ``volume``
where needed), and returns a panel aligned to ``close``. Because the inputs are already
point-in-time (``build_fundamentals`` only reveals a number from its filing date
forward), the factors inherit that honesty for free — no factor here peeks across its
own row into the future.

Design intent on the synthetic world (``data.make_synthetic_raw_fundamentals``):
  * QUALITY / PROFITABILITY factors (gross_profitability, ROE, margins, low accruals,
    low leverage) carry a GENUINE point-in-time signal — they track the latent
    ``world.quality`` that drives next-day drift.
  * VALUE factors (earnings/book/sales/fcf-to-price) are deliberately NOT planted; on
    synthetic data they should look like scale-free noise. The engine should be
    honestly unconvinced by value here — that is the point of the brand.

Honesty rules obeyed throughout:
  * Every ratio goes through :func:`safe_div`, which returns NaN (never inf/huge) on a
    zero or invalid denominator. NaN means "absent", never zero.
  * A factor that needs a field the bundle does not carry returns an all-NaN panel
    (tolerate partial data) rather than raising — partial coverage is honest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .factors import cross_sectional_zscore, blend, momentum
from .fundamentals import Fundamentals

TRADING_DAYS = 252


# ----------------------------- nan-safe helpers -----------------------------
def safe_div(num, den, eps: float = 1e-12) -> pd.DataFrame:
    """Element-wise division that returns NaN (never inf/huge) on a zero/invalid
    denominator. Works on DataFrames, Series, or scalars; result follows pandas
    broadcasting. A denominator with magnitude <= ``eps`` is treated as invalid."""
    num = num if isinstance(num, (pd.DataFrame, pd.Series)) else np.asarray(num, dtype=float)
    den = den if isinstance(den, (pd.DataFrame, pd.Series)) else np.asarray(den, dtype=float)
    if isinstance(num, (pd.DataFrame, pd.Series)) or isinstance(den, (pd.DataFrame, pd.Series)):
        out = num / den
        # mask out invalid denominators (|den| <= eps, or NaN den)
        bad_den = (den.abs() <= eps) if isinstance(den, (pd.DataFrame, pd.Series)) else (np.abs(den) <= eps)
        out = out.where(~bad_den)
        return out.replace([np.inf, -np.inf], np.nan)
    # pure scalar / array path
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.divide(num, den)
    out = np.where(np.abs(den) <= eps, np.nan, out)
    out = np.where(np.isfinite(out), out, np.nan)
    return out


def avg2(a, b):
    """Nan-aware average of two aligned panels (useful for 2-point balance-sheet
    averages, e.g. average assets). If one side is NaN the other is used; if both
    NaN the result is NaN."""
    return pd.concat([a, b]).groupby(level=0).mean() if isinstance(a, pd.Series) else (
        (a + b) / 2.0
    ).where(a.notna() & b.notna()).combine_first(a).combine_first(b)


def delta(curr, prev):
    """Change ``curr - prev`` preserving NaN where either side is missing."""
    return curr - prev


def binary(cond) -> pd.DataFrame:
    """Map a boolean panel to 1.0/0.0, keeping NaN where the condition is undefined
    (i.e. where the underlying comparison involved a NaN)."""
    out = cond.astype(float)
    # where cond came from a comparison on NaN, pandas yields False; we cannot
    # distinguish that here, so callers pass an explicit validity mask via .where.
    return out


# ----------------------------- field access -----------------------------
def _missing_panel(close: pd.DataFrame) -> pd.DataFrame:
    """An all-NaN panel aligned to ``close`` — the honest result when a required
    field is absent from the bundle (partial-data tolerant)."""
    return pd.DataFrame(np.nan, index=close.index, columns=close.columns, dtype=float)


def _get(fund: Fundamentals, metric: str, close: pd.DataFrame) -> pd.DataFrame:
    """Fetch a metric panel aligned to ``close``; all-NaN if the bundle lacks it."""
    if not fund.has(metric):
        return _missing_panel(close)
    return fund[metric].reindex(index=close.index, columns=close.columns)


def market_cap(fund: Fundamentals, close: pd.DataFrame) -> pd.DataFrame:
    """Price * diluted shares — the common denominator of the value ratios."""
    return close * _get(fund, "shares_diluted", close)


# ----------------------------- profitability / quality -----------------------------
def gross_profitability(fund: Fundamentals, close: pd.DataFrame) -> pd.DataFrame:
    """(revenue - cogs) / total_assets — Novy-Marx gross profitability."""
    rev, cogs, ta = _get(fund, "revenue", close), _get(fund, "cogs", close), _get(fund, "total_assets", close)
    return safe_div(rev - cogs, ta)


def operating_profitability(fund: Fundamentals, close: pd.DataFrame) -> pd.DataFrame:
    """(revenue - cogs - sga - interest_expense) / common_equity (Fama-French RMW-ish)."""
    rev, cogs = _get(fund, "revenue", close), _get(fund, "cogs", close)
    sga, intexp = _get(fund, "sga", close), _get(fund, "interest_expense", close)
    ceq = _get(fund, "common_equity", close)
    return safe_div(rev - cogs - sga - intexp, ceq)


def roa(fund: Fundamentals, close: pd.DataFrame) -> pd.DataFrame:
    """net_income / total_assets."""
    return safe_div(_get(fund, "net_income", close), _get(fund, "total_assets", close))


def roe(fund: Fundamentals, close: pd.DataFrame) -> pd.DataFrame:
    """net_income / common_equity."""
    return safe_div(_get(fund, "net_income", close), _get(fund, "common_equity", close))


def gross_margin(fund: Fundamentals, close: pd.DataFrame) -> pd.DataFrame:
    """(revenue - cogs) / revenue."""
    rev, cogs = _get(fund, "revenue", close), _get(fund, "cogs", close)
    return safe_div(rev - cogs, rev)


def operating_margin(fund: Fundamentals, close: pd.DataFrame) -> pd.DataFrame:
    """ebit / revenue."""
    return safe_div(_get(fund, "ebit", close), _get(fund, "revenue", close))


def asset_turnover(fund: Fundamentals, close: pd.DataFrame) -> pd.DataFrame:
    """revenue / total_assets."""
    return safe_div(_get(fund, "revenue", close), _get(fund, "total_assets", close))


# ----------------------------- value -----------------------------
def earnings_yield(fund: Fundamentals, close: pd.DataFrame) -> pd.DataFrame:
    """net_income / market_cap (E/P) — higher is cheaper."""
    return safe_div(_get(fund, "net_income", close), market_cap(fund, close))


def book_to_price(fund: Fundamentals, close: pd.DataFrame) -> pd.DataFrame:
    """common_equity / market_cap (B/P)."""
    return safe_div(_get(fund, "common_equity", close), market_cap(fund, close))


def sales_to_price(fund: Fundamentals, close: pd.DataFrame) -> pd.DataFrame:
    """revenue / market_cap (S/P)."""
    return safe_div(_get(fund, "revenue", close), market_cap(fund, close))


def fcf_yield(fund: Fundamentals, close: pd.DataFrame) -> pd.DataFrame:
    """(op_cash_flow - capex) / market_cap — free-cash-flow yield."""
    ocf, capex = _get(fund, "op_cash_flow", close), _get(fund, "capex", close)
    return safe_div(ocf - capex, market_cap(fund, close))


# ----------------------------- leverage / liquidity -----------------------------
def leverage(fund: Fundamentals, close: pd.DataFrame) -> pd.DataFrame:
    """(current_debt + long_term_debt) / total_assets — total debt to assets."""
    cd, ltd = _get(fund, "current_debt", close), _get(fund, "long_term_debt", close)
    return safe_div(cd + ltd, _get(fund, "total_assets", close))


def current_ratio(fund: Fundamentals, close: pd.DataFrame) -> pd.DataFrame:
    """current_assets / current_liabilities."""
    return safe_div(_get(fund, "current_assets", close), _get(fund, "current_liabilities", close))


# ----------------------------- growth / issuance / accruals -----------------------------
def asset_growth(fund: Fundamentals, close: pd.DataFrame, lag: int = TRADING_DAYS) -> pd.DataFrame:
    """Year-over-year change in total_assets on the daily PIT panel.

    Because the panel is a point-in-time, forward-filled daily series (a quarterly
    number repeats day to day until the next filing), a ``lag``-bar shift of ~252
    trading days approximates the one-year-ago value of total_assets. Documented lag:
    this is *daily-bar* YoY, not calendar-exact; near the warmup it is NaN until a
    full ``lag`` of history exists. Higher asset growth is the (negative) anomaly.
    """
    ta = _get(fund, "total_assets", close)
    return safe_div(ta - ta.shift(lag), ta.shift(lag))


def net_equity_issuance(fund: Fundamentals, close: pd.DataFrame, lag: int = TRADING_DAYS) -> pd.DataFrame:
    """log(shares_diluted / shares_diluted.shift(lag)) — net share issuance (~1yr).

    Positive = issuance (dilution, a negative anomaly); negative = buyback. Uses the
    same ~252-bar daily shift as ``asset_growth``; NaN through the warmup and where
    the ratio is non-positive (log undefined)."""
    sh = _get(fund, "shares_diluted", close)
    ratio = safe_div(sh, sh.shift(lag))
    valid = ratio > 0
    return np.log(ratio.where(valid))


def sloan_accruals(fund: Fundamentals, close: pd.DataFrame) -> pd.DataFrame:
    """(net_income - op_cash_flow) / total_assets — Sloan accruals.

    High accruals (earnings not backed by cash) are a NEGATIVE anomaly; lower is
    better quality."""
    ni, ocf = _get(fund, "net_income", close), _get(fund, "op_cash_flow", close)
    return safe_div(ni - ocf, _get(fund, "total_assets", close))


# ----------------------------- composites -----------------------------
def quality_score(fund: Fundamentals, close: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional QUALITY composite: high gross profitability, high ROE, high
    operating margin, LOW accruals, LOW leverage. On the synthetic world this is the
    planted, point-in-time signal."""
    z = cross_sectional_zscore
    return blend(
        z(gross_profitability(fund, close)),
        z(roe(fund, close)),
        z(operating_margin(fund, close)),
        z(-sloan_accruals(fund, close)),
        z(-leverage(fund, close)),
    )


def value_score(fund: Fundamentals, close: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional VALUE composite: cheap on earnings, book, sales, and free cash
    flow. Deliberately NOT planted in the synthetic world (scale-free noise)."""
    z = cross_sectional_zscore
    return blend(
        z(earnings_yield(fund, close)),
        z(book_to_price(fund, close)),
        z(sales_to_price(fund, close)),
        z(fcf_yield(fund, close)),
    )


# ----------------------------- distress / quality screens -----------------------------
def altman_z(fund: Fundamentals, close: pd.DataFrame) -> pd.DataFrame:
    """Altman Z-score (manufacturing, 1968), built from available canonical fields.

    Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
      X1 = (current_assets - current_liabilities) / total_assets   (working capital)
      X2 = retained_earnings / total_assets
      X3 = ebit / total_assets
      X4 = market_cap / total_liabilities                          (equity mkt value)
      X5 = revenue / total_assets                                  (asset turnover)
    Low Z (<1.8) = distress zone. Returns NaN where any required input is missing
    (NaN-tolerant): a NaN in any term propagates, which is honest — we do not
    fabricate a score from partial data."""
    ta = _get(fund, "total_assets", close)
    ca, cl = _get(fund, "current_assets", close), _get(fund, "current_liabilities", close)
    re, ebit = _get(fund, "retained_earnings", close), _get(fund, "ebit", close)
    rev, tl = _get(fund, "revenue", close), _get(fund, "total_liabilities", close)
    x1 = safe_div(ca - cl, ta)
    x2 = safe_div(re, ta)
    x3 = safe_div(ebit, ta)
    x4 = safe_div(market_cap(fund, close), tl)
    x5 = safe_div(rev, ta)
    return 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5


def ohlson_o(fund: Fundamentals, close: pd.DataFrame) -> pd.DataFrame:
    """Ohlson O-score (1980) bankruptcy probability proxy — field-hungry.

    O = -1.32 - 0.407*log(TA) + 6.03*TL/TA - 1.43*WC/TA + 0.0757*CL/CA
        - 1.72*OENEG - 2.37*NI/TA - 1.83*FU/TL + 0.285*INTWO - 0.521*CHIN
    where WC = working capital, OENEG = 1 if total_liabilities > total_assets,
    FU ~ op_cash_flow (funds from operations proxy), INTWO = 1 if net_income < 0 for
    two consecutive periods (approximated as current period < 0; conservative),
    CHIN = scaled change in NI. Higher O = higher distress probability. Returns NaN
    where inputs are missing — on sparse synthetic data this may be largely NaN, which
    is the honest behaviour (see deviations)."""
    ta = _get(fund, "total_assets", close)
    tl = _get(fund, "total_liabilities", close)
    ca, cl = _get(fund, "current_assets", close), _get(fund, "current_liabilities", close)
    ni, ocf = _get(fund, "net_income", close), _get(fund, "op_cash_flow", close)
    wc = ca - cl
    oeneg = (tl > ta).astype(float).where(ta.notna() & tl.notna())
    intwo = (ni < 0).astype(float).where(ni.notna())
    ni_prev = ni.shift(TRADING_DAYS)
    chin = safe_div(ni - ni_prev, ni.abs() + ni_prev.abs())
    return (
        -1.32
        - 0.407 * np.log(ta.where(ta > 0))
        + 6.03 * safe_div(tl, ta)
        - 1.43 * safe_div(wc, ta)
        + 0.0757 * safe_div(cl, ca)
        - 1.72 * oeneg
        - 2.37 * safe_div(ni, ta)
        - 1.83 * safe_div(ocf, tl)
        + 0.285 * intwo
        - 0.521 * chin
    )


def beneish_m(fund: Fundamentals, close: pd.DataFrame) -> pd.DataFrame:
    """Beneish M-score (earnings-manipulation detector) — uses year-over-year ratios.

    Implements the subset of the 8 indices computable from the canonical fields, on
    the daily PIT panel with a ~252-bar (one-year) shift:
      DSRI = (receivables/revenue) vs prior year
      GMI  = prior gross margin / current gross margin
      AQI  = (1 - (current_assets)/total_assets) vs prior year
      SGI  = revenue / prior revenue
      TATA = (net_income - op_cash_flow) / total_assets  (total accruals to assets)
    M ~= -4.84 + 0.92*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI + 4.679*TATA
        (DEPI and LVGI dropped — depreciation schedule / leverage-change inputs not
         separately available; documented in deviations). Higher M = more likely
         manipulation. NaN through the one-year warmup and where inputs are missing."""
    rev = _get(fund, "revenue", close)
    rect = _get(fund, "receivables", close)
    ta = _get(fund, "total_assets", close)
    ca = _get(fund, "current_assets", close)
    gp = _get(fund, "gross_profit", close)
    ni, ocf = _get(fund, "net_income", close), _get(fund, "op_cash_flow", close)
    L = TRADING_DAYS

    dsri = safe_div(safe_div(rect, rev), safe_div(rect.shift(L), rev.shift(L)))
    gm_curr, gm_prev = safe_div(gp, rev), safe_div(gp.shift(L), rev.shift(L))
    gmi = safe_div(gm_prev, gm_curr)
    aqi_curr = 1.0 - safe_div(ca, ta)
    aqi_prev = 1.0 - safe_div(ca.shift(L), ta.shift(L))
    aqi = safe_div(aqi_curr, aqi_prev)
    sgi = safe_div(rev, rev.shift(L))
    tata = safe_div(ni - ocf, ta)
    return -4.84 + 0.92 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi + 4.679 * tata


# ----------------------------- price / volume factors -----------------------------
def momentum_12_1(close: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """12-1 month price momentum: return over the past ~12 months skipping the most
    recent ~1 month (avoids short-term reversal). Delegates to ``factors.momentum`` so
    the point-in-time (shifted, past-only) construction is shared and consistent."""
    return momentum(close, lookback=lookback, skip=skip)


def amihud_illiquidity(close: pd.DataFrame, volume: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Amihud (2002) illiquidity: rolling mean of |daily return| / dollar volume.

    Higher = more price impact per dollar traded = less liquid. Uses only past data
    (rolling window ending at t). Dollar volume = close * volume; a zero-volume day
    yields NaN for that day's ratio (via ``safe_div``), excluded from the mean."""
    ret = close.pct_change(fill_method=None)
    dollar_vol = close * volume
    daily = safe_div(ret.abs(), dollar_vol)
    return daily.rolling(window, min_periods=max(2, window // 2)).mean()
