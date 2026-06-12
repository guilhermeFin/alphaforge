"""Signal-quality scorecard demo — judge the signal, not just the backtest.

Offline, deterministic:
    python examples/demo_signal_quality.py

Scores two genuine point-in-time signals from the synthetic world:
  * AI-style sentiment (from transcripts, via the deterministic keyword extractor)
  * Fundamental quality (ROE / gross profitability / low leverage)
and asks of each: is the IC significant, does it hold out-of-sample, how fast does
it decay, and is it monotonic across quantiles? Plus a re-run consistency check.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from research import data, factors, fundamentals, quantamental  # noqa: E402
from research import signal_quality as sq  # noqa: E402


def _f(v, spec="+.4f"):
    return format(v, spec) if v is not None else "n/a"


def show(name, signal, close):
    card = sq.scorecard(signal, close)
    h = card["headline_ic"]
    print(f"\n=== {name} ===")
    print(f"  IC (h=1):        mean {_f(h['mean_ic'])} | t-stat {_f(h['ic_tstat'], '+.2f')} "
          f"| IR {_f(h['ic_ir'], '+.2f')} | hit-rate {_f(h['ic_hit_rate'], '.0%')}")
    print(f"  IC decay:        " + "  ".join(f"h{k}={_f(v, '+.3f')}" for k, v in card["ic_decay"].items()))
    print(f"  out-of-sample:   IS IC {_f(card['in_sample_ic'])} -> OOS IC {_f(card['out_sample_ic'])}")
    print(f"  quantiles:       top-minus-bottom fwd return {_f(card['quantiles']['top_minus_bottom'], '+.5f')} "
          f"| monotonic={card['quantiles']['monotonic_increasing']}")
    print(f"  signal turnover: rank autocorr {card['rank_autocorr']:+.3f} (1=slow/cheap, 0=noisy/costly) "
          f"| coverage {card['coverage']:.0%}")
    print(f"  -> {card['verdict']}")


def main() -> None:
    symbols = [f"S{i:02d}" for i in range(18)]
    world = data.make_synthetic_world(symbols, periods=1512, seed=7,
                                      n_events_per_symbol=16, quality_to_drift=0.0016)
    close = world.close

    # fundamental quality signal
    obs = data.make_synthetic_fundamentals(world, seed=7)
    fund = fundamentals.build_fundamentals(obs, close.index, world.symbols)
    quality = fundamentals.quality_score(fund)

    # AI-style sentiment signal (deterministic offline extractor)
    docs = quantamental.make_transcripts(world.events, seed=7)
    sentiment = factors.cross_sectional_zscore(
        quantamental.build_signal_panel(docs, close.index, world.symbols,
                                        quantamental.keyword_extractor, field="tone", horizon=63))

    print(f"World: {len(symbols)} symbols x {len(close)} days")
    show("Fundamental QUALITY", quality, close)
    show("AI SENTIMENT (from transcripts)", sentiment, close)

    # re-run consistency (deterministic extractor -> perfectly stable; a real LLM would not be)
    reps = [[quantamental.keyword_extractor(t)["tone"] for _ in range(3)] for t in docs["text"].head(20)]
    stab = sq.extraction_stability(reps)
    print(f"\n=== Re-run stability (keyword extractor, 20 docs x 3 runs) ===")
    print(f"  mean within-item std {stab['mean_within_item_std']:.4f} | deterministic={stab['deterministic']}")
    print("  (a real LLM extractor would show std > 0 here -- that dispersion is itself a quality metric)\n")


if __name__ == "__main__":
    main()
