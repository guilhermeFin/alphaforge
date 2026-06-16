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

from .factors import cross_sectional_zscore, blend, momentum, trailing_volatility
from .fundamentals import Fundamentals
from . import factors

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


def enterprise_value(fund: Fundamentals, close: pd.DataFrame) -> pd.DataFrame:
    """Enterprise value = market_cap + total_debt - cash.

    total_debt = current_debt + long_term_debt. The takeover-cost view of a firm; the
    natural denominator for EBIT/EV (a capital-structure-neutral cheapness ratio).
    NaN where any component is missing (inherits ``safe_div``/NaN tolerance from the
    underlying fields)."""
    cd, ltd = _get(fund, "current_debt", close), _get(fund, "long_term_debt", close)
    cash = _get(fund, "cash", close)
    return market_cap(fund, close) + (cd + ltd) - cash


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


# ----------------------------- composite alphas -----------------------------
# These are the "alpha" panels the coordinator long/short-z-scores. They combine the
# documented building blocks above. NaN-handling choice (documented): each component
# is first cross-sectionally z-scored; we then combine the z-scores treating a MISSING
# z-component as a 0 contribution (i.e. neutral, the cross-sectional mean) rather than
# letting one absent screen wipe the whole name. This keeps a name scored on the
# components it DOES have — partial coverage stays honest, not silently dropped. A name
# with NO computable component at a date is left NaN (nothing to say).
def _zsum(*panels: pd.DataFrame) -> pd.DataFrame:
    """Sum already-z-scored panels, treating NaN as 0 (neutral) per cell, but keeping a
    cell NaN only where EVERY input is NaN (no information at all)."""
    arrs = [p.to_numpy(dtype=float) for p in panels]
    stack = np.stack(arrs, axis=0)
    total = np.nansum(stack, axis=0)
    all_nan = np.all(np.isnan(stack), axis=0)
    total = np.where(all_nan, np.nan, total)
    return pd.DataFrame(total, index=panels[0].index, columns=panels[0].columns)


def _zprod(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Product of two z-scored panels, NaN only where EITHER input is NaN (a product
    needs both legs; there is no neutral substitute that preserves the AND semantics)."""
    out = a * b
    return out


def value_with_fraud_guardrail(fund: Fundamentals, close: pd.DataFrame) -> pd.DataFrame:
    """Cheap names, PENALISED for manipulation/distress.

      score = z(book_to_price) + z(ebit/enterprise_value) - z(beneish_m) - z(ohlson_o)

    The two value legs reward cheapness (high B/P, high EBIT/EV); the two screen legs
    SUBTRACT manipulation (Beneish M) and bankruptcy risk (Ohlson O), so a cheap name
    that also looks manipulated or distressed is pulled back down. Higher = better
    (cheap AND clean). NaN-tolerant via ``_zsum`` (a missing screen contributes 0, so a
    name still scores on the value legs it has)."""
    z = cross_sectional_zscore
    ebit_ev = safe_div(_get(fund, "ebit", close), enterprise_value(fund, close))
    return _zsum(
        z(book_to_price(fund, close)),
        z(ebit_ev),
        -z(beneish_m(fund, close)),
        -z(ohlson_o(fund, close)),
    )


def profitable_value(fund: Fundamentals, close: pd.DataFrame) -> pd.DataFrame:
    """Only names that are BOTH very cheap AND very profitable score high.

      score = z(book_to_price) * z(gross_profitability)

    The product is an AND gate: it is large-positive only when both legs are high (or
    both low — a cheap, low-quality value trap scores negative, which is intended), and
    near zero when either is average. Higher = better. NaN where either leg is missing
    (the AND needs both)."""
    z = cross_sectional_zscore
    return _zprod(z(book_to_price(fund, close)), z(gross_profitability(fund, close)))


def conservative_compounder(fund: Fundamentals, close: pd.DataFrame) -> pd.DataFrame:
    """Quality + investment discipline + low risk.

      score = z(gross_profitability) - z(asset_growth) - z(realized_vol)

    Rewards high gross profitability, penalises aggressive asset growth (the
    asset-growth anomaly) and high realised volatility (the low-vol anomaly).
    ``realized_vol`` is ``factors.trailing_volatility(close)`` (annualised trailing
    63-day vol, past-only). Higher = better. NaN-tolerant via ``_zsum``."""
    z = cross_sectional_zscore
    realized_vol = factors.trailing_volatility(close)
    return _zsum(
        z(gross_profitability(fund, close)),
        -z(asset_growth(fund, close)),
        -z(realized_vol),
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


def piotroski_f_score(fund: Fundamentals, close: pd.DataFrame) -> pd.DataFrame:
    """Piotroski F-score (2000): an integer 0..9 quality screen on point-in-time
    fundamentals. Each of nine binary tests scores 1 if passed, else 0; the score is
    their sum (higher = financially stronger).

    The nine signals (current vs prior year on the daily PIT panel, ~252-bar shift):
      1. ROA > 0                          (net_income / beginning total_assets)
      2. CFO > 0                          (op_cash_flow / beginning total_assets)
      3. ΔROA > 0                         (ROA improved year-over-year)
      4. accrual: CFO/TA > ROA            (cash earnings exceed accrual earnings)
      5. ΔLeverage < 0                    (long_term_debt / avg total_assets fell)
      6. ΔCurrentRatio > 0                (current_assets / current_liabilities rose)
      7. no share issuance                (shares_diluted did not increase YoY)
      8. ΔGrossMargin > 0                 (gross_profit / revenue rose)
      9. ΔAssetTurnover > 0               (revenue / beginning total_assets rose)

    Following Piotroski, profitability and turnover are scaled by BEGINNING-of-period
    assets (i.e. the lagged total_assets), and leverage by AVERAGE assets. The score
    is NaN-tolerant: each test contributes NaN where its inputs are missing, and the
    name's score is NaN only when FEWER than ~7 of the nine tests are computable (a
    screen, not a guaranteed quantity). Returns a (dates x symbols) integer-ish panel
    with values in [0, 9] (or NaN). NaN through the one-year warmup (needs a prior
    year for the four Δ tests and the beginning-assets scaling)."""
    L = TRADING_DAYS
    ta = _get(fund, "total_assets", close)
    ni = _get(fund, "net_income", close)
    ocf = _get(fund, "op_cash_flow", close)
    ltd = _get(fund, "long_term_debt", close)
    ca, cl = _get(fund, "current_assets", close), _get(fund, "current_liabilities", close)
    sh = _get(fund, "shares_diluted", close)
    gp, rev = _get(fund, "gross_profit", close), _get(fund, "revenue", close)

    ta_beg = ta.shift(L)        # beginning-of-period assets (Piotroski scales by these)
    ta_beg2 = ta.shift(2 * L)   # assets two years ago (beginning for the prior year)

    # 1. ROA > 0 ; 2. CFO > 0 (both scaled by beginning assets)
    roa_c = safe_div(ni, ta_beg)
    roa_p = safe_div(ni.shift(L), ta_beg2)
    cfo_c = safe_div(ocf, ta_beg)
    f_roa = (roa_c > 0).astype(float).where(roa_c.notna())
    f_cfo = (cfo_c > 0).astype(float).where(cfo_c.notna())
    # 3. ΔROA > 0
    f_droa = (roa_c - roa_p > 0).astype(float).where(roa_c.notna() & roa_p.notna())
    # 4. accrual: CFO/TA > ROA  (compared on the same beginning-assets scaling)
    f_accr = (cfo_c > roa_c).astype(float).where(cfo_c.notna() & roa_c.notna())
    # 5. ΔLeverage < 0 (LTD / average assets)
    avg_ta = avg2(ta, ta_beg)
    avg_ta_p = avg2(ta.shift(L), ta_beg2)
    lev_c = safe_div(ltd, avg_ta)
    lev_p = safe_div(ltd.shift(L), avg_ta_p)
    f_lev = (lev_c - lev_p < 0).astype(float).where(lev_c.notna() & lev_p.notna())
    # 6. ΔCurrentRatio > 0
    cr_c = safe_div(ca, cl)
    cr_p = safe_div(ca.shift(L), cl.shift(L))
    f_cr = (cr_c - cr_p > 0).astype(float).where(cr_c.notna() & cr_p.notna())
    # 7. no share issuance (shares did not increase)
    sh_p = sh.shift(L)
    f_iss = (sh - sh_p <= 0).astype(float).where(sh.notna() & sh_p.notna())
    # 8. ΔGrossMargin > 0
    gm_c = safe_div(gp, rev)
    gm_p = safe_div(gp.shift(L), rev.shift(L))
    f_gm = (gm_c - gm_p > 0).astype(float).where(gm_c.notna() & gm_p.notna())
    # 9. ΔAssetTurnover > 0 (revenue / beginning assets)
    at_c = safe_div(rev, ta_beg)
    at_p = safe_div(rev.shift(L), ta_beg2)
    f_at = (at_c - at_p > 0).astype(float).where(at_c.notna() & at_p.notna())

    tests = [f_roa, f_cfo, f_droa, f_accr, f_lev, f_cr, f_iss, f_gm, f_at]
    # nan-aware sum: a missing test contributes 0 to the score but is counted as
    # "not available"; require >= 7 of 9 available or the whole score is NaN (honest:
    # too sparse to call). Stack along a new axis to count availability per cell.
    stack = np.stack([t.to_numpy(dtype=float) for t in tests], axis=0)  # (9, T, N)
    avail = np.sum(~np.isnan(stack), axis=0)                            # (T, N)
    score = np.nansum(stack, axis=0)                                   # (T, N)
    score = np.where(avail >= 7, score, np.nan)
    return pd.DataFrame(score, index=close.index, columns=close.columns)


def ohlson_o(fund: Fundamentals, close: pd.DataFrame) -> pd.DataFrame:
    """Ohlson O-score (1980): the full nine-term logit bankruptcy proxy.

    O = -1.32 - 0.407*log(TA/GNP) + 6.03*(TL/TA) - 1.43*(WC/TA) + 0.0757*(CL/CA)
        - 1.72*OENEG - 2.37*(NI/TA) - 1.83*(FFO/TL)
        + 0.285*INTWO - 0.521*CHIN
    where
      WC    = working_capital = current_assets - current_liabilities,
      OENEG = 1 if total_liabilities > total_assets (negative book equity), else 0,
      FFO   = funds from operations; here op_cash_flow is the documented proxy (no
              explicit FFO line in the canonical fields),
      INTWO = 1 if net_income < 0 in BOTH the current and prior year, else 0,
      CHIN  = (NI - NI_prev) / (|NI| + |NI_prev|), a scale-free change in net income,
      GNP   = a GNP / price-deflator index; with no index available we use the
              documented proxy GNP = 1.0, so log(TA/GNP) = log(TA).
    Higher O = higher implied distress probability. Returns NaN where inputs are
    missing (the screen tolerates partial data rather than fabricating a score). The
    two-year terms (INTWO, CHIN) are NaN through the one-year warmup."""
    GNP = 1.0  # no GNP/price-deflator index available -> documented unit proxy
    L = TRADING_DAYS
    ta = _get(fund, "total_assets", close)
    tl = _get(fund, "total_liabilities", close)
    ca, cl = _get(fund, "current_assets", close), _get(fund, "current_liabilities", close)
    ni = _get(fund, "net_income", close)
    ffo = _get(fund, "op_cash_flow", close)  # funds-from-operations proxy (documented)
    wc = ca - cl
    oeneg = (tl > ta).astype(float).where(ta.notna() & tl.notna())
    ni_prev = ni.shift(L)
    intwo = ((ni < 0) & (ni_prev < 0)).astype(float).where(ni.notna() & ni_prev.notna())
    chin = safe_div(ni - ni_prev, ni.abs() + ni_prev.abs())
    return (
        -1.32
        - 0.407 * np.log((ta / GNP).where(ta > 0))
        + 6.03 * safe_div(tl, ta)
        - 1.43 * safe_div(wc, ta)
        + 0.0757 * safe_div(cl, ca)
        - 1.72 * oeneg
        - 2.37 * safe_div(ni, ta)
        - 1.83 * safe_div(ffo, tl)
        + 0.285 * intwo
        - 0.521 * chin
    )


def beneish_m(fund: Fundamentals, close: pd.DataFrame) -> pd.DataFrame:
    """Beneish M-score (1999): the full eight-variable earnings-manipulation detector.

    M = -4.84 + 0.92*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI + 0.115*DEPI
        - 0.172*SGAI + 4.679*TATA - 0.327*LVGI
    Year-over-year ratios on the daily PIT panel (~252-bar shift):
      DSRI = (receivables/sales) / prior(receivables/sales)
      GMI  = prior gross margin / current gross margin
      AQI  = (1 - (current_assets + ppe_net + securities)/total_assets) vs prior
      SGI  = sales / prior sales
      DEPI = prior depreciation-rate / current depreciation-rate,
             depreciation-rate = depreciation / (depreciation + ppe_net)
      SGAI = (sga/sales) / prior(sga/sales)
      TATA = (income_cont_ops - op_cash_flow) / total_assets   (total accruals)
      LVGI = ((current_liabilities + long_term_debt)/total_assets) vs prior
    Higher M (toward/above ~-1.78 / -2.22) = more likely manipulation. Each component
    is NaN-tolerant via ``safe_div``; the score is NaN through the one-year warmup and
    where a required field is missing."""
    L = TRADING_DAYS
    rev = _get(fund, "revenue", close)
    rect = _get(fund, "receivables", close)
    ta = _get(fund, "total_assets", close)
    ca = _get(fund, "current_assets", close)
    ppe = _get(fund, "ppe_net", close)
    sec = _get(fund, "securities", close)
    gp = _get(fund, "gross_profit", close)
    dep = _get(fund, "depreciation", close)
    sga = _get(fund, "sga", close)
    ico = _get(fund, "income_cont_ops", close)
    ocf = _get(fund, "op_cash_flow", close)
    cl = _get(fund, "current_liabilities", close)
    ltd = _get(fund, "long_term_debt", close)

    # DSRI: days-sales-in-receivables index
    dsri = safe_div(safe_div(rect, rev), safe_div(rect.shift(L), rev.shift(L)))
    # GMI: gross-margin index (prior / current)
    gm_curr, gm_prev = safe_div(gp, rev), safe_div(gp.shift(L), rev.shift(L))
    gmi = safe_div(gm_prev, gm_curr)
    # AQI: asset-quality index — non-(CA+PPE+securities) share of assets vs prior
    aqi_curr = 1.0 - safe_div(ca + ppe + sec, ta)
    aqi_prev = 1.0 - safe_div(ca.shift(L) + ppe.shift(L) + sec.shift(L), ta.shift(L))
    aqi = safe_div(aqi_curr, aqi_prev)
    # SGI: sales growth index
    sgi = safe_div(rev, rev.shift(L))
    # DEPI: depreciation index (prior dep-rate / current dep-rate)
    dep_rate_c = safe_div(dep, dep + ppe)
    dep_rate_p = safe_div(dep.shift(L), dep.shift(L) + ppe.shift(L))
    depi = safe_div(dep_rate_p, dep_rate_c)
    # SGAI: SG&A index (current SGA/sales vs prior)
    sgai = safe_div(safe_div(sga, rev), safe_div(sga.shift(L), rev.shift(L)))
    # TATA: total accruals to total assets (continuing-ops income less CFO)
    tata = safe_div(ico - ocf, ta)
    # LVGI: leverage index ((CL+LTD)/TA current vs prior)
    lev_c = safe_div(cl + ltd, ta)
    lev_p = safe_div(cl.shift(L) + ltd.shift(L), ta.shift(L))
    lvgi = safe_div(lev_c, lev_p)

    return (
        -4.84
        + 0.92 * dsri
        + 0.528 * gmi
        + 0.404 * aqi
        + 0.892 * sgi
        + 0.115 * depi
        - 0.172 * sgai
        + 4.679 * tata
        - 0.327 * lvgi
    )


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
