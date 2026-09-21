"""Data ingestion with a stable interface.

The MVP needs to run offline, deterministically, in tests and demos — so the
default provider is a *synthetic* market generator with two properties that make
it a fair stress-test of an honest backtester:

  * a 2-state (bull/bear) Markov regime drives the market factor — so strategies
    must survive regime change, not just a single trend (Blitzstein Ch.11);
  * innovations are Student-t (fat-tailed) by default — so the engine's tail and
    normality diagnostics have something real to flag (Blitzstein Ch.6/Ch.10).

A free, keyless yfinance provider is included for real tickers. A customer can
also point ``ALPHAFORGE_LICENSED_DATA_PATH`` at a locally exported, validated
licensed-data bundle; that vendor-neutral route never uploads or redistributes
the source data. We deliberately serve *derived* results downstream.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class Panel:
    """A point-in-time-aligned panel: dates x symbols."""
    close: pd.DataFrame
    volume: pd.DataFrame
    regime: pd.Series | None = None  # 1=bull, 0=bear (synthetic only)

    @property
    def symbols(self) -> list[str]:
        return list(self.close.columns)


def _innovations(rng: np.random.Generator, size, fat_tails: bool, t_df: int) -> np.ndarray:
    """Unit-variance shocks. Student-t (standardised) if fat_tails else Normal."""
    if fat_tails:
        if t_df <= 2:
            raise ValueError("t_df must be > 2 for finite variance")
        x = rng.standard_t(t_df, size=size)
        return x / np.sqrt(t_df / (t_df - 2.0))  # standardise to unit variance
    return rng.standard_normal(size)


def make_synthetic_panel(
    symbols: list[str],
    periods: int = 1260,
    start: str = "2015-01-02",
    seed: int = 0,
    regime: bool = True,
    fat_tails: bool = True,
    t_df: int = 4,
) -> Panel:
    """Generate a deterministic synthetic OHLCV panel.

    Returns daily prices for ``symbols`` over ``periods`` business days. Each name
    has a market beta plus a small persistent alpha (so cross-sectional momentum
    has something faint and noisy to find — honest, not planted to look easy) and
    idiosyncratic fat-tailed noise. A shared Markov regime flips drift/vol.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=periods)
    n = len(symbols)
    T = periods

    # --- shared market regime (2-state Markov chain) ---
    if regime:
        # rows = current state, cols = next state; state 1 = bull, 0 = bear
        P = np.array([[0.95, 0.05],   # from bear
                      [0.02, 0.98]])  # from bull (bulls persist longer)
        states = np.empty(T, dtype=int)
        states[0] = 1
        u = rng.random(T)
        for t in range(1, T):
            states[t] = 1 if u[t] < P[states[t - 1], 1] else 0
    else:
        states = np.ones(T, dtype=int)

    mkt_mu = np.where(states == 1, 0.0005, -0.0009)   # daily market drift by regime
    mkt_sd = np.where(states == 1, 0.008, 0.020)      # bear vol > bull vol
    market = mkt_mu + mkt_sd * _innovations(rng, T, fat_tails, t_df)

    betas = rng.uniform(0.6, 1.4, n)
    alphas = rng.uniform(-0.00025, 0.00045, n)        # persistent winners/losers
    idio_sd = rng.uniform(0.15, 0.45, n) / np.sqrt(TRADING_DAYS) * 0.7

    rets = np.empty((T, n))
    for j in range(n):
        rets[:, j] = alphas[j] + betas[j] * market + idio_sd[j] * _innovations(rng, T, fat_tails, t_df)

    close = pd.DataFrame(100.0 * np.cumprod(1.0 + rets, axis=0), index=dates, columns=symbols)
    # volume: lognormal, mildly higher on big-move days (crude but plausible)
    base = rng.uniform(5e5, 5e6, n)
    vmult = np.exp(0.5 * np.abs(rets) / idio_sd - 0.125)
    volume = pd.DataFrame(base * vmult * np.exp(rng.normal(0, 0.2, (T, n))),
                          index=dates, columns=symbols).round()

    return Panel(close=close, volume=volume,
                 regime=pd.Series(states, index=dates, name="regime") if regime else None)


def _yf_close_panel(symbols: list[str], start: str, end: str | None) -> Panel:
    try:
        import yfinance as yf
    except ImportError as e:  # pragma: no cover
        raise ImportError("yfinance not installed. `pip install alphaforge[data]`") from e
    raw = yf.download(symbols, start=start, end=end, auto_adjust=True, progress=False)
    close = raw["Close"].copy()
    volume = raw["Volume"].copy()
    if isinstance(close, pd.Series):  # single symbol
        close = close.to_frame(symbols[0])
        volume = volume.to_frame(symbols[0])
    close = close.ffill().dropna(how="all")
    volume = volume.reindex(close.index)
    return Panel(close=close, volume=volume)


def get_panel(
    symbols: list[str],
    start: str = "2015-01-02",
    end: str | None = None,
    provider: str = "synthetic",
    **kwargs,
) -> Panel:
    """Stable entry point for synthetic, Yahoo, or a local licensed bundle."""
    if provider == "synthetic":
        panel = make_synthetic_panel(symbols, start=start, **kwargs)
        if end is None:
            return panel
        cutoff = pd.Timestamp(end)
        return Panel(close=panel.close.loc[panel.close.index <= cutoff],
                     volume=panel.volume.loc[panel.volume.index <= cutoff],
                     regime=None if panel.regime is None else panel.regime.loc[panel.regime.index <= cutoff])
    if provider in ("yfinance", "yahoo"):
        return _yf_close_panel(symbols, start, end)
    if provider in ("licensed_bundle", "licensed"):
        # Delayed import prevents an optional local-data path from adding any
        # start-up work to the deterministic synthetic/Yahoo providers.
        from .licensed_data import load_price_panel
        close, volume = load_price_panel(symbols, path=kwargs.get("bundle_path"))
        if start is not None:
            close, volume = close.loc[close.index >= pd.Timestamp(start)], volume.loc[volume.index >= pd.Timestamp(start)]
        if end is not None:
            close, volume = close.loc[close.index <= pd.Timestamp(end)], volume.loc[volume.index <= pd.Timestamp(end)]
        return Panel(close=close, volume=volume)
    raise ValueError(f"unknown provider: {provider!r}")


@dataclass
class World:
    """A synthetic world for testing quantamental pipelines end-to-end.

    Prices are driven by a *latent* per-name fundamental state ``quality`` that is
    known at each date and drives the NEXT day's drift. ``events`` are points in
    time where that state would be 'disclosed' (e.g. an earnings call), carrying
    the ground-truth latent sentiment. Text is generated from those events
    elsewhere (research/quantamental.py); the backtester only ever sees a signal
    *extracted from text*, never the latent state — so a strategy that works is a
    genuine, point-in-time-correct predictive signal, not a leak.
    """
    close: pd.DataFrame
    volume: pd.DataFrame
    quality: pd.DataFrame   # latent fundamental state (ground truth), dates x symbols
    events: pd.DataFrame    # columns: symbol, date, latent_sentiment
    regime: pd.Series | None = None

    @property
    def symbols(self) -> list[str]:
        return list(self.close.columns)


def make_synthetic_world(
    symbols: list[str],
    periods: int = 1512,
    start: str = "2015-01-02",
    seed: int = 0,
    n_events_per_symbol: int = 12,
    quality_persistence: float = 0.985,
    quality_to_drift: float = 0.0006,
    fat_tails: bool = True,
    t_df: int = 4,
) -> World:
    """Coupled price + latent-fundamental generator (see World)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=periods)
    n, T = len(symbols), periods

    # latent fundamental quality: stationary AR(1) per name (unit variance)
    rho = quality_persistence
    q = np.empty((T, n))
    q[0] = rng.standard_normal(n)
    shock = rng.standard_normal((T, n))
    for t in range(1, T):
        q[t] = rho * q[t - 1] + np.sqrt(1.0 - rho ** 2) * shock[t]

    # market factor with a 2-state regime
    P = np.array([[0.95, 0.05], [0.02, 0.98]])
    states = np.empty(T, dtype=int)
    states[0] = 1
    u = rng.random(T)
    for t in range(1, T):
        states[t] = 1 if u[t] < P[states[t - 1], 1] else 0
    market = np.where(states == 1, 0.0005, -0.0009) + \
        np.where(states == 1, 0.008, 0.020) * _innovations(rng, T, fat_tails, t_df)

    betas = rng.uniform(0.6, 1.4, n)
    idio_sd = rng.uniform(0.15, 0.45, n) / np.sqrt(TRADING_DAYS) * 0.7
    q_lag = np.vstack([q[0:1], q[:-1]])  # yesterday's quality drives today's drift -> no same-day leak

    rets = np.empty((T, n))
    for j in range(n):
        rets[:, j] = betas[j] * market + quality_to_drift * q_lag[:, j] + idio_sd[j] * _innovations(rng, T, fat_tails, t_df)

    close = pd.DataFrame(100.0 * np.cumprod(1.0 + rets, axis=0), index=dates, columns=symbols)
    volume = pd.DataFrame(rng.uniform(5e5, 5e6, n) * np.exp(rng.normal(0, 0.2, (T, n))),
                          index=dates, columns=symbols).round()
    quality = pd.DataFrame(q, index=dates, columns=symbols)

    warm = 10
    rows = []
    cand = np.arange(warm, T - 5)
    for j, sym in enumerate(symbols):
        picks = np.sort(rng.choice(cand, size=min(n_events_per_symbol, cand.size), replace=False))
        for d in picks:
            rows.append((sym, dates[d], float(np.tanh(q[d, j]))))  # disclosed sentiment ~ current quality
    events = pd.DataFrame(rows, columns=["symbol", "date", "latent_sentiment"]).sort_values("date").reset_index(drop=True)

    return World(close=close, volume=volume, quality=quality, events=events,
                 regime=pd.Series(states, index=dates, name="regime"))


def make_synthetic_fundamentals(
    world: World,
    reporting_lag_days: int = 75,
    period_days: int = 63,
    seed: int = 0,
) -> pd.DataFrame:
    """Quarterly fundamentals for a synthetic World, with a realistic reporting lag.

    Returns a long table [symbol, period_end, available_date, metric, value]. The
    KEY honesty property: ``available_date`` = fiscal period end + ``reporting_lag_days``
    (a quarter's numbers are not knowable until the filing lands ~75 days later).
    Quality-type metrics (roe, gross_profitability, low leverage) track the latent
    ``world.quality`` so the QUALITY factor has a genuine, point-in-time-correct
    signal to find; the price-ratio inputs (eps/bvps/sps) are made ~scale-free, so
    the VALUE factor is intentionally NOT planted (on synthetic data it should look
    like noise — and the engine should be honestly unconvinced by it).
    """
    rng = np.random.default_rng(seed)
    idx = world.close.index
    T = len(idx)
    rows = []
    period_ends = idx[period_days - 1::period_days]
    for pe in period_ends:
        pos = idx.searchsorted(pe + pd.Timedelta(days=reporting_lag_days))
        if pos >= T:
            continue
        avail = idx[pos]
        for sym in world.symbols:
            ql = float(world.quality.loc[pe, sym])
            px = float(world.close.loc[pe, sym])
            metrics = {
                "roe": 0.12 + 0.06 * ql + rng.normal(0, 0.02),
                "gross_profitability": 0.30 + 0.10 * ql + rng.normal(0, 0.03),
                "leverage": max(0.05, 0.80 - 0.25 * ql + rng.normal(0, 0.10)),
                "eps_ttm": px * (0.05 + rng.normal(0, 0.010)),   # ~constant earnings yield
                "bvps": px * (0.50 + rng.normal(0, 0.100)),       # ~constant book-to-price
                "sps": px * (0.80 + rng.normal(0, 0.150)),        # ~constant sales-to-price
            }
            for metric, value in metrics.items():
                rows.append((sym, pe, avail, metric, float(value)))
    return pd.DataFrame(rows, columns=["symbol", "period_end", "available_date", "metric", "value"])


def make_synthetic_raw_fundamentals(
    world: World,
    reporting_lag_days: int = 75,
    period_days: int = 63,
    seed: int = 0,
) -> pd.DataFrame:
    """Quarterly RAW canonical fundamentals for a synthetic World, with a reporting lag.

    Companion to ``make_synthetic_fundamentals`` that emits the CANONICAL RAW FIELDS
    (revenue, cogs, gross_profit, ... op_cash_flow, capex — the keys of
    ``providers.CANONICAL_FIELDS``) instead of pre-derived ratios. This lets the full
    factor library (research/factor_lib.py) run end-to-end offline against the same
    point-in-time machinery the paid vendors feed.

    Returns a long table [symbol, period_end, available_date, metric, value] with the
    SAME honesty property as its sibling: ``available_date`` = fiscal period end +
    ``reporting_lag_days`` (the filing is not knowable until it lands).

    Signal design (so the engine measures something honest):
      * Every raw field is driven by the latent per-name ``world.quality`` state at the
        period end. Higher quality -> higher gross profitability, ROE, operating margin,
        operating cash flow; LOWER leverage and LOWER accruals. So the QUALITY /
        PROFITABILITY factors built from these fields carry a genuine, point-in-time
        predictive signal (quality drives the next day's drift in World).
      * VALUE ratios (earnings/book/sales-to-price) are deliberately NOT planted: scale
        (revenue, equity, shares) is an independent per-name draw uncorrelated with
        quality, and price is the realised synthetic price, so E/P, B/P, S/P look like
        ~scale-free noise. On synthetic data the value factor SHOULD look unconvincing.

    Accounting is kept self-consistent within each statement:
      gross_profit = revenue - cogs;  ebit ~= gross_profit - sga - r&d (>0 by construction
      via the margin design);  net_income ~= (ebit - interest_expense) * (1 - tax);
      op_cash_flow ~= net_income + D&A-like noise (accruals wedge keyed to quality);
      total_assets > 0;  common_equity > 0;  balance-sheet pieces sum coherently.

    Does NOT modify ``make_synthetic_fundamentals``.
    """
    rng = np.random.default_rng(seed)
    idx = world.close.index
    T = len(idx)
    symbols = world.symbols
    n = len(symbols)

    # ---- per-name, time-invariant SCALE draws (uncorrelated with quality) ----
    # These set each firm's "size". Because they are independent of quality, the
    # price-ratio (value) factors derived downstream are scale-free noise, not planted.
    base_assets = rng.uniform(5e8, 5e10, n)          # total assets in $
    shares = rng.uniform(5e7, 5e8, n)                 # diluted share count
    # asset turnover (revenue / assets): a pure per-name draw, INDEPENDENT of quality,
    # so the value ratios it feeds (sales/earnings/book-to-price) stay scale-free noise.
    # Kept in a tight band so that gross_profitability = turnover * gross_margin is
    # driven mainly by the quality-loaded margin, not by turnover dispersion.
    base_turnover = rng.uniform(0.95, 1.05, n)

    rows = []
    period_ends = idx[period_days - 1::period_days]
    # Time-varying share count so the share-issuance factor has real dispersion. The
    # per-period issuance drift is drawn from a SEPARATE rng, so the main data stream
    # (and every other factor) stays byte-identical; it is INDEPENDENT of quality, so
    # "low_issuance" is honest NOISE on synthetic data — but no longer identically zero.
    _iss_rng = np.random.default_rng(seed + 12345)
    _issuance = _iss_rng.normal(0.004, 0.012, size=(len(period_ends), n))  # ~0.4%/q, can be <0
    _share_path = shares[None, :] * np.cumprod(1.0 + _issuance, axis=0)

    # ---- EXTRA raw fields for the full distress / quality screens (Piotroski,
    # Altman, Ohlson, Beneish) ----
    # These genuinely-new dollar amounts are drawn from a SEPARATE rng so the main
    # per-row draw stream above stays byte-identical (existing factors/tests rely on
    # it). The fractions are INDEPENDENT of world.quality on purpose: these screens
    # must be honest NOISE on synthetic data, not a planted edge. They are pre-drawn
    # as (period x symbol) matrices keyed by position so the values are deterministic
    # regardless of the row-emission order.
    _extra_rng = np.random.default_rng(seed + 4242)
    P = len(period_ends)
    # PP&E as a fraction of total assets; depreciation as a quarterly fraction of PP&E;
    # marketable securities as a fraction of total assets; taxes payable as a fraction
    # of current liabilities. All scale-free, quality-independent.
    _ppe_frac = np.clip(_extra_rng.normal(0.35, 0.08, size=(P, n)), 0.05, 0.80)
    _dep_frac = np.clip(_extra_rng.normal(0.06 / 4.0, 0.005, size=(P, n)), 0.005, 0.10)
    _sec_frac = np.clip(_extra_rng.normal(0.05, 0.02, size=(P, n)), 0.0, 0.30)
    _taxp_frac = np.clip(_extra_rng.normal(0.10, 0.03, size=(P, n)), 0.0, 0.40)
    for pi, pe in enumerate(period_ends):
        pos = idx.searchsorted(pe + pd.Timedelta(days=reporting_lag_days))
        if pos >= T:
            continue
        avail = idx[pos]
        for j, sym in enumerate(symbols):
            ql = float(world.quality.loc[pe, sym])  # latent quality (unit-variance-ish)

            # ----- income statement (quarter) -----
            assets = base_assets[j] * float(np.exp(rng.normal(0.0, 0.03)))
            # asset turnover is pure noise (no quality tilt) -> revenue scale carries
            # NO quality info, so value ratios remain unplanted.
            turnover_q = base_turnover[j] + rng.normal(0.0, 0.03)
            turnover_q = float(max(0.05, turnover_q))
            revenue = assets * turnover_q / 4.0  # quarterly slice of annual turnover

            # gross margin rises STRONGLY and cleanly with quality (the genuine signal).
            # Strong loading + tiny idiosyncratic noise so gross_profitability is a near-
            # monotone function of latent quality (the planted, point-in-time edge).
            gross_margin = float(np.clip(0.40 + 0.22 * ql + rng.normal(0.0, 0.005), 0.02, 0.95))
            gross_profit = revenue * gross_margin
            cogs = revenue - gross_profit  # exact self-consistency

            # opex as a share of revenue FALLS with quality -> higher operating margin
            sga = revenue * float(np.clip(0.18 - 0.05 * ql + rng.normal(0.0, 0.010), 0.02, 0.50))
            rnd = revenue * float(np.clip(0.04 + 0.01 * ql + rng.normal(0.0, 0.004), 0.0, 0.30))
            ebit = gross_profit - sga - rnd  # ~ operating income

            # ----- balance sheet (point-in-time stock) -----
            # leverage (debt/assets) FALLS with quality
            lev = float(np.clip(0.45 - 0.12 * ql + rng.normal(0.0, 0.05), 0.0, 0.90))
            total_debt = assets * lev
            current_debt = total_debt * float(np.clip(0.30 + rng.normal(0.0, 0.05), 0.05, 0.95))
            long_term_debt = total_debt - current_debt
            # non-debt liabilities (payables etc.) as a modest share of assets
            other_liab = assets * float(np.clip(0.18 + rng.normal(0.0, 0.04), 0.02, 0.50))
            total_liabilities = total_debt + other_liab
            # common equity = assets - liabilities, floored strictly positive
            common_equity = max(0.05 * assets, assets - total_liabilities)
            # keep the identity assets >= liabilities + equity-floor honest:
            total_liabilities = min(total_liabilities, assets - common_equity)

            current_assets = assets * float(np.clip(0.40 + rng.normal(0.0, 0.05), 0.05, 0.90))
            cash = current_assets * float(np.clip(0.30 + 0.05 * ql + rng.normal(0.0, 0.05), 0.02, 0.95))
            receivables = current_assets * float(np.clip(0.25 + rng.normal(0.0, 0.04), 0.02, 0.80))
            inventory = max(0.0, current_assets - cash - receivables) * float(np.clip(0.6 + rng.normal(0.0, 0.1), 0.0, 1.0))
            current_liabilities = current_debt + accounts_payable_share(rng, current_assets)
            accounts_payable = current_liabilities - current_debt
            retained_earnings = common_equity * float(np.clip(0.55 + 0.10 * ql + rng.normal(0.0, 0.05), 0.0, 0.95))

            # ----- below-EBIT income -----
            interest_expense = total_debt * (0.06 / 4.0) * float(np.clip(1.0 + rng.normal(0.0, 0.1), 0.3, 2.0))
            # A LARGE, quality-INDEPENDENT non-operating line (one-offs, gains/losses,
            # variable tax). This deliberately swamps the quality content of bottom-line
            # earnings, so the price-ratio VALUE factors (E/P, FCF yield) that use
            # net_income / op_cash_flow stay ~scale-free NOISE — not a planted edge.
            # The OPERATING ratios (gross profitability, margin) keep their clean quality
            # signal because they sit ABOVE this noisy line.
            nonop = ebit * float(rng.normal(0.0, 0.60))
            pretax = ebit - interest_expense + nonop
            net_income = pretax * (1.0 - 0.21)  # flat 21% tax wedge; can be negative

            # ----- cash flow: NI + D&A-like wedge; accruals SHRINK with quality -----
            dep = assets * (0.05 / 4.0)  # depreciation add-back, ~5% of assets annually
            # accrual wedge (NI - OCF) is smaller (more negative OCF gap) for low quality:
            accrual = revenue * float(np.clip(0.02 - 0.02 * ql + rng.normal(0.0, 0.01), -0.10, 0.10))
            op_cash_flow = net_income + dep - accrual
            capex = revenue * float(np.clip(0.06 + rng.normal(0.0, 0.01), 0.0, 0.40))

            # ----- EXTRA fields for the full screens (from the SEPARATE rng) -----
            # Net PP&E is a fraction of (non-current) asset base; depreciation is a
            # quarterly fraction of PP&E; securities and taxes_payable are small
            # scale-free balances. income_cont_ops uses net_income as the proxy
            # (income from continuing operations ~= net income when there are no
            # discontinued ops, which the synthetic world does not model).
            ppe_net = assets * float(_ppe_frac[pi, j])
            depreciation = ppe_net * float(_dep_frac[pi, j])
            securities = assets * float(_sec_frac[pi, j])
            taxes_payable = current_liabilities * float(_taxp_frac[pi, j])
            income_cont_ops = net_income  # no discontinued-ops modelling -> equal
            short_term_debt = current_debt  # canonical alias of the current-debt line

            fields = {
                "revenue": revenue,
                "cogs": cogs,
                "gross_profit": gross_profit,
                "sga": sga,
                "rnd": rnd,
                "ebit": ebit,
                "interest_expense": interest_expense,
                "net_income": net_income,
                "shares_diluted": float(_share_path[pi, j]),
                "total_assets": assets,
                "current_assets": current_assets,
                "cash": cash,
                "receivables": receivables,
                "inventory": inventory,
                "current_liabilities": current_liabilities,
                "accounts_payable": accounts_payable,
                "current_debt": current_debt,
                "long_term_debt": long_term_debt,
                "total_liabilities": total_liabilities,
                "common_equity": common_equity,
                "retained_earnings": retained_earnings,
                "op_cash_flow": op_cash_flow,
                "capex": capex,
                # extra fields for the full distress / quality screens
                "ppe_net": ppe_net,
                "depreciation": depreciation,
                "securities": securities,
                "taxes_payable": taxes_payable,
                "income_cont_ops": income_cont_ops,
                "short_term_debt": short_term_debt,
            }
            for metric, value in fields.items():
                rows.append((sym, pe, avail, metric, float(value)))
    return pd.DataFrame(rows, columns=["symbol", "period_end", "available_date", "metric", "value"])


def accounts_payable_share(rng: np.random.Generator, current_assets: float) -> float:
    """Helper: a plausible accounts-payable level as a share of current assets."""
    return current_assets * float(np.clip(0.20 + rng.normal(0.0, 0.04), 0.02, 0.60))
