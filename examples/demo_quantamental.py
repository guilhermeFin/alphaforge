"""Quantamental fusion demo: LLM-style text signal -> point-in-time panel ->
honest backtest, on its own and blended with price momentum.

Offline by default (deterministic keyword extractor, $0):
    python examples/demo_quantamental.py
Use the real Claude extractor (cached; needs ANTHROPIC_API_KEY):
    python examples/demo_quantamental.py --live --max-docs 60

HONESTY NOTE: this runs on a SYNTHETIC world where a latent fundamental state
both (a) is disclosed in the transcripts and (b) drives the NEXT day's returns.
So a working sentiment signal here is a genuine, point-in-time-correct predictive
signal -- it proves the *pipeline* transmits real information without leaking the
future. It is NOT evidence of a real-world edge. On true filings the signal will
be far weaker and noisier; that is exactly what the Deflated Sharpe is there to expose.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from research import data, factors, stats_guards  # noqa: E402
from research.backtest import backtest  # noqa: E402
from research.quantamental import (  # noqa: E402
    keyword_extractor, make_llm_extractor, make_transcripts, build_signal_panel,
)
from research.walkforward import split_backtest  # noqa: E402


def scorecard(close, weights, label, cost_bps, n_trials):
    res = backtest(close, weights, cost_bps=cost_bps)
    s = res.summary(n_trials=n_trials)
    oos = split_backtest(close, weights, split=0.7, cost_bps=cost_bps)
    print(f"\n--- {label} ---")
    print(f"  ann_sharpe     {s['ann_sharpe']:+.3f}      max_drawdown {s['max_drawdown']:+.3f}")
    print(f"  PSR vs 0       {s['psr_vs_0']:.3f}       Deflated SR  {s['deflated_sr']:.3f}  "
          f"({'CREDIBLE' if (s['deflated_sr'] or 0) > 0.95 else 'not credible'} @ {n_trials} trials)")
    print(f"  OOS sharpe     {oos['out_sample_sharpe']:+.3f}      IS->OOS degr {oos['degradation']:+.3f}"
          f"   overfit {'YES' if oos['overfit_warning'] else 'no'}"
          f"   OOS-significant {'yes' if oos['oos_significant'] else 'no'}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--max-docs", type=int, default=60, help="cap LLM calls in --live mode")
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--n-trials", type=int, default=20)
    args = ap.parse_args()

    symbols = [f"S{i:02d}" for i in range(15)]
    # quality_to_drift is turned up so the (synthetic, planted) fundamental signal is
    # genuinely strong -- enough to clear the Deflated-Sharpe bar -- so the demo can show
    # the engine CERTIFYING a real point-in-time signal while rejecting price-noise.
    world = data.make_synthetic_world(symbols, periods=1512, seed=11,
                                      n_events_per_symbol=16, quality_to_drift=0.0014)
    docs = make_transcripts(world.events, seed=11)
    close = world.close
    print(f"World: {len(symbols)} symbols x {len(close)} days, {len(docs)} transcripts "
          f"({world.regime.mean():.0%} of days bull regime)")

    if args.live:
        import anthropic
        extractor = make_llm_extractor(anthropic.Anthropic(),
                                       cache_path="cache/llm_signals.json")
        used = docs.sort_values("date").head(args.max_docs)
        print(f"LIVE: extracting {len(used)} transcripts via Claude (cached) "
              f"~${len(used) * 0.0017:.2f} first run, $0 cached")
    else:
        extractor = keyword_extractor
        used = docs
        print("OFFLINE: deterministic keyword extractor ($0)")

    # text -> point-in-time sentiment panel
    raw_sent = build_signal_panel(used, close.index, symbols, extractor, field="tone", horizon=63)
    sent_score = factors.cross_sectional_zscore(raw_sent)
    mom_score = factors.cross_sectional_zscore(factors.momentum(close, lookback=252, skip=21))
    combo_score = factors.blend(sent_score, mom_score)

    print("\n================= honest backtests (costs charged) =================")
    scorecard(close, factors.long_short_weights(sent_score), "LLM sentiment alone", args.cost_bps, args.n_trials)
    scorecard(close, factors.long_short_weights(mom_score), "Price momentum alone", args.cost_bps, args.n_trials)
    scorecard(close, factors.long_short_weights(combo_score), "QUANTAMENTAL combo (blend)", args.cost_bps, args.n_trials)

    # prove the sentiment signal is not peeking at same-day returns
    name = symbols[0]
    look = stats_guards.lookahead_warning(sent_score[name], close[name].pct_change(fill_method=None))
    print(f"\nLook-ahead guard on sentiment ({name}): same-day corr {look['corr_same_day']:+.3f} vs "
          f"next-day {look['corr_next_day']:+.3f} -> "
          f"{'SUSPICIOUS' if look['suspected_lookahead'] else 'clean (point-in-time)'}")
    print("\n(Reminder: synthetic planted signal -- this validates the pipeline, not a real edge.)\n")


if __name__ == "__main__":
    main()
