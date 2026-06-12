"""Point-in-time fundamentals demo: value & quality factors through the honest engine.

Offline, deterministic, no keys:
    python examples/demo_fundamentals.py

Builds a synthetic world with a latent fundamental state, emits quarterly
fundamentals with a 75-day reporting lag (point-in-time), then backtests VALUE,
QUALITY, and a VALUE+QUALITY combo. On this synthetic data QUALITY is a genuine
(planted) point-in-time signal while VALUE is deliberately not planted — so the
engine should certify quality and stay unconvinced by value. That contrast is the
brand: the tool distinguishes a real signal from a plausible-looking non-signal.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from research import data, factors  # noqa: E402
from research import fundamentals as fd  # noqa: E402
from research.backtest import backtest  # noqa: E402
from research.walkforward import split_backtest  # noqa: E402


def scorecard(close, score, name, n_trials=20):
    w = factors.long_short_weights(score)
    res = backtest(close, w, cost_bps=5.0)
    c = res.summary(n_trials=n_trials)
    oos = split_backtest(close, w, split=0.7)
    print(f"\n--- {name} ---")
    print(f"  CAGR {c['cagr']:.2%} | Ann.Sharpe {c['ann_sharpe']:.2f} | MaxDD {c['max_drawdown']:.2%}")
    print(f"  PSR(full) {c['psr_vs_0']:.3f} | Deflated SR(full) {c['deflated_sr']:.3f} "
          f"| OOS Sharpe {oos['out_sample_sharpe']:+.2f} | overfit {oos['overfit_warning']}")
    verdict = "CERTIFIED" if (c['deflated_sr'] or 0) > 0.95 else "not certified"
    print(f"  -> {verdict} by the (full-sample) Deflated-Sharpe bar")


def main() -> None:
    symbols = [f"S{i:02d}" for i in range(20)]
    # quality_to_drift is turned up vs the default so the PLANTED quality signal is
    # strong enough to clear the certification bar — making the contrast with the
    # (unplanted) value factor explicit. Still 100% synthetic; not a real edge.
    world = data.make_synthetic_world(symbols, periods=1512, seed=7,
                                      n_events_per_symbol=1, quality_to_drift=0.0016)
    obs = data.make_synthetic_fundamentals(world, reporting_lag_days=75, seed=7)
    fund = fd.build_fundamentals(obs, world.close.index, world.symbols)
    close = world.close

    print(f"World: {len(symbols)} symbols x {len(close)} days "
          f"({close.index[0].date()} -> {close.index[-1].date()})")
    print(f"Fundamentals: {obs['metric'].nunique()} metrics, {len(obs)} point-in-time observations "
          f"(75-day reporting lag)")

    scorecard(close, fd.value_score(fund, close), "VALUE (E/P + B/P + S/P)  [not planted -> expect noise]")
    scorecard(close, fd.quality_score(fund), "QUALITY (ROE + gross profitability - leverage)  [planted signal]")
    scorecard(close, fd.value_quality_score(fund, close), "VALUE + QUALITY combo")

    print("\nReading: the engine should reward QUALITY (a real point-in-time signal here) "
          "and stay skeptical of VALUE (no premium was planted). Synthetic data is an engine "
          "test, never evidence of a real-world edge.\n")


if __name__ == "__main__":
    main()
