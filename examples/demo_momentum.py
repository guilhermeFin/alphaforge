"""End-to-end offline demo: universe -> momentum factor -> honest backtest ->
scorecard + overfit flag + fat-tail / look-ahead guards.

Runs with no API keys and no internet on synthetic data:
    python examples/demo_momentum.py
Use real tickers (needs internet, free):
    python examples/demo_momentum.py --provider yfinance

NOTE: synthetic results are an ENGINE SMOKE-TEST, not evidence of a real edge.
The whole point of this tool is to make that distinction impossible to fudge.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # repo root on path

import pandas as pd  # noqa: E402

from research import data, factors, stats_guards  # noqa: E402
from research.backtest import backtest
from research.walkforward import split_backtest, walk_forward


def _fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "YES" if v else "no"
    if isinstance(v, float):
        return f"{v:,.4f}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="synthetic", choices=["synthetic", "yfinance"])
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--n-trials", type=int, default=50,
                    help="how many strategy variants you tried (for the Deflated Sharpe haircut)")
    args = ap.parse_args()

    if args.provider == "synthetic":
        symbols = [f"S{i:02d}" for i in range(15)]
        panel = data.get_panel(symbols, provider="synthetic", periods=1512, seed=42)
    else:
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "XOM",
                   "JNJ", "PG", "KO", "WMT", "HD", "BAC", "DIS"]
        panel = data.get_panel(symbols, start="2015-01-01", provider="yfinance")

    close = panel.close
    print(f"\nUniverse: {len(close.columns)} symbols x {len(close)} days "
          f"({close.index[0].date()} -> {close.index[-1].date()}) via {args.provider}")

    # --- pipeline: 12-1 momentum -> cross-sectional z -> dollar-neutral weights ---
    score = factors.cross_sectional_zscore(factors.momentum(close, lookback=252, skip=21))
    weights = factors.long_short_weights(score, gross=1.0)
    res = backtest(close, weights, cost_bps=args.cost_bps)
    card = res.summary(n_trials=args.n_trials)

    print("\n=== Honest scorecard (long/short momentum, costs charged) ===")
    order = ["n_periods", "total_return", "cagr", "ann_vol", "ann_sharpe", "sortino",
             "max_drawdown", "calmar", "hit_rate", "avg_turnover", "skew",
             "excess_kurtosis", "jarque_bera_p", "returns_are_normal",
             "psr_vs_0", "deflated_sr"]
    for k in order:
        if k in card:
            print(f"  {k:<18} {_fmt(card[k])}")

    verdict = ("CREDIBLE (clears the multiple-testing bar)" if (card['deflated_sr'] or 0) > 0.95
               else "NOT CREDIBLE after the Deflated-Sharpe haircut -- likely noise/overfit")
    print(f"\n  -> Deflated Sharpe verdict: {verdict}")

    # --- out-of-sample / overfit ---
    print("\n=== Out-of-sample check (70/30 split) ===")
    for k, v in split_backtest(close, weights, split=0.7, cost_bps=args.cost_bps).items():
        print(f"  {k:<20} {_fmt(v)}")

    print("\n=== Walk-forward (5 folds) ===")
    for k, v in walk_forward(close, weights, n_splits=5, cost_bps=args.cost_bps).items():
        print(f"  {k:<20} {_fmt(v)}")

    # --- honesty guards ---
    print("\n=== Fat-tail report (portfolio returns) ===")
    for k, v in stats_guards.fat_tail_report(res.returns).items():
        print(f"  {k:<20} {_fmt(v)}")

    print("\n=== Look-ahead guard (signal vs single name) ===")
    name = close.columns[0]
    look = stats_guards.lookahead_warning(score[name], close[name].pct_change(fill_method=None))
    for k, v in look.items():
        print(f"  {k:<20} {_fmt(v)}")

    print("\n=== Base-rate reminder (e.g. a 'crash' signal) ===")
    br = stats_guards.base_rate_precision(base_rate=0.02, true_positive_rate=0.90, false_positive_rate=0.10)
    print(f"  A 90%-recall signal for a 2%-base-rate event has precision "
          f"{br['precision']:.1%} (lift {br['lift']:.1f}x). Recall is not precision.\n")


if __name__ == "__main__":
    main()
