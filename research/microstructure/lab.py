"""Trade-flow research and a deliberately assumption-visible quote simulator.

This is research software, not a trading system.  Historical public trade files
do not contain the BBO or queue state needed to establish realised limit-order
adverse selection.  The module therefore reports *trade-price markouts* and
labels the market-making component as a simulation.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, log, sqrt
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from research import reproducibility


class MicrostructureError(ValueError):
    """Raised when an event stream is unsuitable for chronological research."""


REQUIRED_COLUMNS = ("timestamp", "price", "quantity", "is_buyer_maker")


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def validate_trade_events(events: pd.DataFrame, *, minimum_events: int = 120) -> pd.DataFrame:
    """Return a clean, ordered trade stream or explain why it is not research-ready."""
    missing = [name for name in REQUIRED_COLUMNS if name not in events.columns]
    if missing:
        raise MicrostructureError(f"trade events are missing required columns: {', '.join(missing)}")
    clean = events.loc[:, REQUIRED_COLUMNS].copy()
    clean["timestamp"] = pd.to_datetime(clean["timestamp"], utc=True, errors="coerce")
    clean["price"] = pd.to_numeric(clean["price"], errors="coerce")
    clean["quantity"] = pd.to_numeric(clean["quantity"], errors="coerce")
    maker = clean["is_buyer_maker"]
    if maker.dtype == object:
        normalized = maker.astype(str).str.strip().str.lower()
        if not normalized.isin({"true", "false", "1", "0"}).all():
            raise MicrostructureError("is_buyer_maker must contain true/false values")
        clean["is_buyer_maker"] = normalized.isin({"true", "1"})
    else:
        clean["is_buyer_maker"] = maker.astype(bool)
    if clean.isna().any().any() or (clean["price"] <= 0).any() or (clean["quantity"] <= 0).any():
        raise MicrostructureError("timestamps, prices, and quantities must be present and strictly positive")
    if len(clean) < minimum_events:
        raise MicrostructureError(f"at least {minimum_events} ordered trade events are required")
    if clean.duplicated().any():
        raise MicrostructureError("duplicate trade events were found; de-duplicate at the source before research")
    if not clean["timestamp"].is_monotonic_increasing:
        raise MicrostructureError("events must be timestamp-ordered; sorting a mixed source would hide an audit problem")
    return clean.reset_index(drop=True)


def load_binance_trades(path_or_buffer: str | Path | object) -> pd.DataFrame:
    """Load Binance public trade or aggregate-trade CSV data into the common contract."""
    raw = pd.read_csv(path_or_buffer)
    aliases = {
        "time": "timestamp", "timestamp": "timestamp", "price": "price",
        "qty": "quantity", "quantity": "quantity", "isBuyerMaker": "is_buyer_maker",
        "is_buyer_maker": "is_buyer_maker",
    }
    selected: dict[str, pd.Series] = {}
    for source, target in aliases.items():
        if source in raw.columns and target not in selected:
            selected[target] = raw[source]
    if set(selected) != set(REQUIRED_COLUMNS):
        raise MicrostructureError("Binance CSV needs time/timestamp, price, qty/quantity, and isBuyerMaker columns")
    events = pd.DataFrame(selected)
    numeric_time = pd.to_numeric(events["timestamp"], errors="coerce")
    if numeric_time.notna().all():
        # Binance timestamps switched to microseconds in 2025. Infer only from
        # magnitude, then preserve UTC provenance in the normalized event stream.
        unit = "us" if float(numeric_time.abs().median()) >= 1e14 else "ms"
        events["timestamp"] = pd.to_datetime(numeric_time, unit=unit, utc=True)
    return validate_trade_events(events)


def synthetic_trade_events(*, n_events: int = 2_400, seed: int = 42) -> pd.DataFrame:
    """Deterministic smoke-test data, never evidence of a market effect."""
    if n_events < 120:
        raise MicrostructureError("n_events must be at least 120")
    rng = np.random.default_rng(seed)
    aggressor = rng.choice([-1.0, 1.0], size=n_events, p=[0.5, 0.5])
    persistence = pd.Series(aggressor).rolling(16, min_periods=1).mean().to_numpy()
    noise = rng.normal(0.0, 0.00035, n_events)
    log_returns = 0.00018 * persistence + noise
    price = 100.0 * np.exp(np.cumsum(log_returns))
    quantity = rng.lognormal(mean=0.0, sigma=0.55, size=n_events)
    timestamp = pd.date_range("2025-01-02T14:30:00Z", periods=n_events, freq="s")
    return pd.DataFrame({
        "timestamp": timestamp, "price": price, "quantity": quantity,
        "is_buyer_maker": aggressor < 0,
    })


def _feature_panel(events: pd.DataFrame, *, window: int, horizon_events: int) -> pd.DataFrame:
    clean = validate_trade_events(events)
    if horizon_events >= len(clean) // 4:
        raise MicrostructureError("horizon_events must leave enough observations for chronological splits")
    frame = clean.copy()
    # Buyer-maker means the seller initiated the trade, so it is negative flow.
    frame["signed_quantity"] = np.where(frame["is_buyer_maker"], -frame["quantity"], frame["quantity"])
    frame["flow_imbalance"] = frame["signed_quantity"].rolling(window, min_periods=window).sum() / frame["quantity"].rolling(window, min_periods=window).sum()
    frame["last_return"] = np.log(frame["price"]).diff()
    frame["realized_volatility"] = frame["last_return"].rolling(window, min_periods=window).std(ddof=0)
    future_price = frame["price"].shift(-horizon_events)
    frame["forward_markout"] = future_price / frame["price"] - 1.0
    frame["horizon_events"] = horizon_events
    return frame.dropna().reset_index(drop=True)


def _spearman(left: pd.Series, right: pd.Series) -> float | None:
    value = left.corr(right, method="spearman")
    return None if value is None or not np.isfinite(value) else float(value)


def _fit_slope(feature: pd.Series, target: pd.Series) -> float:
    denom = float(np.dot(feature, feature))
    return 0.0 if denom <= 1e-18 else float(np.dot(feature, target) / denom)


def _split_boundaries(n_rows: int) -> tuple[int, int]:
    train_end, validation_end = int(n_rows * 0.60), int(n_rows * 0.80)
    if train_end < 40 or validation_end - train_end < 20 or n_rows - validation_end < 20:
        raise MicrostructureError("event stream is too short for 60/20/20 chronological evaluation")
    return train_end, validation_end


def evaluate_trade_flow(events: pd.DataFrame, *, window: int = 32, horizon_events: int = 30) -> dict:
    """Evaluate flow imbalance against a momentum baseline on a held-out event block."""
    panel = _feature_panel(events, window=window, horizon_events=horizon_events)
    train_end, validation_end = _split_boundaries(len(panel))
    train, validation, holdout = panel.iloc[:train_end], panel.iloc[train_end:validation_end], panel.iloc[validation_end:]
    models = {"last_return": "last_return", "flow_imbalance": "flow_imbalance"}
    rows: list[dict] = []
    for model, column in models.items():
        slope = _fit_slope(train[column], train["forward_markout"])
        for stage, subset in (("validation", validation), ("final_holdout", holdout)):
            prediction = slope * subset[column]
            direction = np.sign(prediction)
            actual_direction = np.sign(subset["forward_markout"])
            rows.append({
                "model": model, "stage": stage, "slope": slope,
                "spearman_ic": _spearman(prediction, subset["forward_markout"]),
                "directional_accuracy": float((direction == actual_direction).mean()),
                "mean_markout_when_positive": float(subset.loc[prediction > 0, "forward_markout"].mean()) if (prediction > 0).any() else None,
                "n_events": int(len(subset)),
            })
    results = pd.DataFrame(rows)
    final = results.loc[results["stage"] == "final_holdout"].set_index("model")
    candidate_ic = final.loc["flow_imbalance", "spearman_ic"]
    baseline_ic = final.loc["last_return", "spearman_ic"]
    survives = bool(
        candidate_ic is not None and baseline_ic is not None
        and np.isfinite(candidate_ic) and np.isfinite(baseline_ic)
        and abs(float(candidate_ic)) >= abs(float(baseline_ic)) + 0.01
    )
    deciles = pd.qcut(holdout["flow_imbalance"], 10, duplicates="drop")
    calibration = holdout.assign(decile=deciles).groupby("decile", observed=True)["forward_markout"].agg(["mean", "count"]).reset_index()
    preview = panel.loc[:, ["timestamp", "price", "flow_imbalance", "forward_markout"]].iloc[::max(1, len(panel) // 240)].copy()
    preview["timestamp"] = preview["timestamp"].astype(str)
    return {
        "definition": "Forward trade-price markout after a fixed number of later trades. It is not quote-based adverse selection.",
        "splits": {"train_events": int(len(train)), "validation_events": int(len(validation)), "final_holdout_events": int(len(holdout))},
        "window_events": window, "horizon_events": horizon_events,
        "results": results.to_dict(orient="records"),
        "calibration": [{"decile": str(row["decile"]), "mean_markout": float(row["mean"]), "n_events": int(row["count"])} for _, row in calibration.iterrows()],
        "survives_baseline": survives,
        "verdict": ("Flow imbalance cleared the pre-declared final-holdout baseline margin." if survives
                    else "Flow imbalance did not clear the pre-declared final-holdout baseline margin."),
        "panel": panel,
    }


@dataclass(frozen=True)
class QuotePolicy:
    name: str
    base_half_spread_bps: float
    volatility_multiplier: float = 0.0
    flow_skew_bps: float = 0.0
    inventory_skew_bps: float = 0.0


POLICIES = (
    QuotePolicy("Fixed spread", 3.0),
    QuotePolicy("Volatility-aware", 3.0, volatility_multiplier=5_000.0),
    QuotePolicy("Inventory and flow-aware", 3.0, volatility_multiplier=5_000.0, flow_skew_bps=4.0, inventory_skew_bps=1.25),
)


def simulate_quoting(events: pd.DataFrame, *, window: int = 32, horizon_events: int = 30,
                     maker_fee_bps: float = 1.0, latency_events: int = 2, max_inventory: int = 8,
                     seed: int = 7) -> dict:
    """Compare transparent *simulated* quote policies against the same event stream."""
    if latency_events < 0 or max_inventory < 1:
        raise MicrostructureError("latency_events must be non-negative and max_inventory must be positive")
    panel = _feature_panel(events, window=window, horizon_events=horizon_events)
    rng = np.random.default_rng(seed)
    outcomes: list[dict] = []
    curves: dict[str, list[dict]] = {}
    for policy in POLICIES:
        cash, inventory, fees, fills, spread_capture, adverse = 0.0, 0, 0.0, 0, 0.0, 0.0
        equity_points: list[dict] = []
        for row in panel.iloc[:-horizon_events].itertuples(index=False):
            price = float(row.price)
            flow = float(row.flow_imbalance)
            volatility = float(row.realized_volatility)
            half_spread_bps = policy.base_half_spread_bps + policy.volatility_multiplier * volatility
            skew_bps = -policy.flow_skew_bps * flow - policy.inventory_skew_bps * inventory
            bid = price * (1.0 - (half_spread_bps - skew_bps) / 10_000.0)
            ask = price * (1.0 + (half_spread_bps + skew_bps) / 10_000.0)
            aggressive_buy = row.signed_quantity > 0
            side = -1 if aggressive_buy else 1
            flow_pressure = min(0.28, abs(flow) * 0.18)
            fill_probability = max(0.03, 0.42 - flow_pressure - 0.015 * latency_events)
            if abs(inventory + side) <= max_inventory and rng.random() < fill_probability:
                fill_price = ask if side == -1 else bid
                cash -= side * fill_price
                inventory += side
                fee = abs(fill_price) * maker_fee_bps / 10_000.0
                cash -= fee
                fees += fee
                fills += 1
                spread_capture += abs(fill_price - price)
                future = float(row.forward_markout) * price
                adverse += max(0.0, -side * future)
            equity = cash + inventory * price
            equity_points.append({"timestamp": row.timestamp.isoformat(), "equity": equity})
        final_price = float(panel.iloc[-1]["price"])
        equity = cash + inventory * final_price
        curve = pd.Series([point["equity"] for point in equity_points], dtype=float)
        returns = curve.diff().fillna(0.0)
        drawdown = curve - curve.cummax()
        outcomes.append({
            "policy": policy.name, "fills": fills, "ending_inventory": inventory,
            "simulated_pnl": float(equity), "maker_fees": float(fees),
            "gross_spread_capture": float(spread_capture), "adverse_selection_proxy": float(adverse),
            "max_drawdown": float(drawdown.min()) if not drawdown.empty else 0.0,
            "per_fill_pnl": float(equity / fills) if fills else None,
            "simulation_only": True,
        })
        curves[policy.name] = equity_points[::max(1, len(equity_points) // 300)]
    return {
        "label": "Assumption-driven discrete-time simulation, not realised exchange fills or queue-position evidence.",
        "assumptions": {"maker_fee_bps": maker_fee_bps, "latency_events": latency_events, "max_inventory": max_inventory, "seed": seed},
        "outcomes": outcomes, "curves": curves,
    }


def _bs_price(spot: float, strike: float, maturity_years: float, rate: float, volatility: float, option_type: str) -> float:
    if min(spot, strike, maturity_years, volatility) <= 0:
        raise MicrostructureError("spot, strike, maturity_years, and volatility must be positive")
    d1 = (log(spot / strike) + (rate + volatility * volatility / 2.0) * maturity_years) / (volatility * sqrt(maturity_years))
    d2 = d1 - volatility * sqrt(maturity_years)
    if option_type == "call":
        return spot * _normal_cdf(d1) - strike * exp(-rate * maturity_years) * _normal_cdf(d2)
    if option_type == "put":
        return strike * exp(-rate * maturity_years) * _normal_cdf(-d2) - spot * _normal_cdf(-d1)
    raise MicrostructureError("option_type must be call or put")


def _bs_delta(spot: float, strike: float, maturity_years: float, rate: float, volatility: float, option_type: str) -> float:
    d1 = (log(spot / strike) + (rate + volatility * volatility / 2.0) * maturity_years) / (volatility * sqrt(maturity_years))
    return _normal_cdf(d1) if option_type == "call" else _normal_cdf(d1) - 1.0


def _implied_volatility(market_price: float, spot: float, strike: float, maturity_years: float, rate: float, option_type: str) -> float | None:
    if market_price <= 0:
        return None
    low, high = 1e-4, 5.0
    if not _bs_price(spot, strike, maturity_years, rate, high, option_type) >= market_price:
        return None
    for _ in range(80):
        middle = (low + high) / 2.0
        if _bs_price(spot, strike, maturity_years, rate, middle, option_type) < market_price:
            low = middle
        else:
            high = middle
    return float((low + high) / 2.0)


def options_diagnostics(contracts: Iterable[dict], *, rate: float = 0.04) -> dict:
    """Return transparent Black-Scholes diagnostics for dated option observations."""
    rows: list[dict] = []
    for contract in contracts:
        try:
            spot, strike = float(contract["spot"]), float(contract["strike"])
            maturity = float(contract["maturity_years"])
            market_price = float(contract["market_price"])
            option_type = str(contract["option_type"]).lower()
            implied = _implied_volatility(market_price, spot, strike, maturity, rate, option_type)
            delta = _bs_delta(spot, strike, maturity, rate, implied, option_type) if implied else None
            rows.append({"contract_id": str(contract.get("contract_id", "unidentified")), "option_type": option_type,
                         "implied_volatility": implied, "delta": delta, "model_price": _bs_price(spot, strike, maturity, rate, implied, option_type) if implied else None,
                         "market_price": market_price, "status": "ok" if implied else "outside_black_scholes_bounds"})
        except (KeyError, TypeError, ValueError, MicrostructureError) as error:
            rows.append({"contract_id": str(contract.get("contract_id", "unidentified")), "status": f"invalid: {type(error).__name__}"})
    return {"rate": rate, "contracts": rows,
            "note": "Mathematical diagnostics only. Indicative or delayed option data is not execution-quality evidence."}


def _source_audit(events: pd.DataFrame, source: str) -> dict:
    gaps = events["timestamp"].diff().dt.total_seconds().dropna()
    return {
        "source": source, "n_events": int(len(events)),
        "start": events["timestamp"].iloc[0].isoformat(), "end": events["timestamp"].iloc[-1].isoformat(),
        "largest_timestamp_gap_seconds": float(gaps.max()) if not gaps.empty else 0.0,
        "trade_only": True, "quote_level_evidence": False,
        "status": "engine_validation" if source == "synthetic" else "exploratory",
        "limitations": [
            "Trade data cannot establish queue position, realised quote fills, or quote-based adverse selection.",
            "This study evaluates chronological trade-price markouts and simulation assumptions only.",
        ],
    }


def _event_fingerprint(events: pd.DataFrame) -> str:
    """Fingerprint typed event data without forcing timestamps through float()."""
    return reproducibility.fingerprint({
        "timestamp": events["timestamp"].astype(str).tolist(),
        "price": events["price"].astype(float).tolist(),
        "quantity": events["quantity"].astype(float).tolist(),
        "is_buyer_maker": events["is_buyer_maker"].astype(bool).tolist(),
    })


def run_microstructure_study(events: pd.DataFrame | None = None, *, source: str = "synthetic", seed: int = 42,
                             window: int = 32, horizon_events: int = 30, maker_fee_bps: float = 1.0,
                             latency_events: int = 2, max_inventory: int = 8) -> dict:
    """Run the reproducible first hypothesis end to end."""
    event_frame = synthetic_trade_events(seed=seed) if events is None else validate_trade_events(events)
    flow = evaluate_trade_flow(event_frame, window=window, horizon_events=horizon_events)
    simulation = simulate_quoting(event_frame, window=window, horizon_events=horizon_events,
                                  maker_fee_bps=maker_fee_bps, latency_events=latency_events,
                                  max_inventory=max_inventory, seed=seed)
    options = options_diagnostics([
        {"contract_id": "demo-call", "option_type": "call", "spot": 100, "strike": 100, "maturity_years": 30 / 365, "market_price": 3.05},
        {"contract_id": "demo-put", "option_type": "put", "spot": 100, "strike": 100, "maturity_years": 30 / 365, "market_price": 2.72},
    ])
    panel = flow.pop("panel")
    preview = panel.loc[:, ["timestamp", "price", "flow_imbalance", "forward_markout"]].iloc[::max(1, len(panel) // 240)].copy()
    preview["timestamp"] = preview["timestamp"].astype(str)
    audit = _source_audit(event_frame, source)
    request = {"source": source, "seed": seed, "window_events": window, "horizon_events": horizon_events,
               "maker_fee_bps": maker_fee_bps, "latency_events": latency_events, "max_inventory": max_inventory}
    event_digest = _event_fingerprint(event_frame)
    manifest = {
        "research_fingerprint": reproducibility.fingerprint({"request": request, "events": event_digest}),
        "data_fingerprint": event_digest,
        "request_fingerprint": reproducibility.fingerprint(request),
    }
    return {
        "research_question": "Does rolling signed trade-flow imbalance improve fixed-horizon trade-price markouts beyond a last-return baseline?",
        "protocol": {
            "hypothesis_locked_before_result": True,
            "chronological_splits": "60% training / 20% validation / 20% final holdout",
            "candidate": "rolling signed trade-flow imbalance",
            "baseline": "last trade return",
            "success_rule": "absolute final-holdout Spearman IC improves by at least 0.01 over baseline",
        },
        "data_audit": audit, "trade_flow": flow, "simulation": simulation,
        "options_diagnostics": options, "manifest": manifest,
        "disclaimer": "Research software only. No result is a trading signal, execution estimate, or investment recommendation.",
        "panel_preview": preview.to_dict(orient="records"),
    }
