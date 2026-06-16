"""ML model-ladder demo: does complexity beat a linear baseline OUT-OF-SAMPLE?

Offline, deterministic, no keys:
    python examples/demo_ml.py

Builds a synthetic world + raw point-in-time fundamentals, assembles ~6 factor
features (quality/profitability are the PLANTED, roughly-linear-in-quality signal;
value/momentum are intentionally weak), then runs the whole model ladder
(lasso, elastic_net, random_forest, xgboost, lightgbm, mlp, stacking) under
LEAK-AWARE purged K-fold cross-validation (López de Prado purging + embargo over
the overlapping forward-return labels).

The honest expectation: because the planted edge is ~linear in quality, the boosted
trees / net / stack should NOT meaningfully beat the elastic-net baseline OOS — and
the demo says so plainly. Complexity isn't free; this layer exists to prove whether
it earns its keep, not to flatter it. Synthetic data is an engine test, never
evidence of a real-world edge.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from research import data  # noqa: E402
from research.fundamentals import build_fundamentals  # noqa: E402
from research.model_eval import build_feature_panel, compare_ladder  # noqa: E402


FACTORS = [
    "gross_profitability",   # planted (quality)
    "roe",                   # planted (quality)
    "operating_margin",      # planted (quality)
    "book_to_price",         # NOT planted (value ~ scale-free noise)
    "asset_growth",          # NOT planted
    "momentum_12_1",         # price factor; faint at best on this world
]


def main() -> None:
    symbols = [f"S{i:02d}" for i in range(25)]
    world = data.make_synthetic_world(symbols, periods=1512, seed=7,
                                      quality_to_drift=0.0012)
    obs = data.make_synthetic_raw_fundamentals(world, seed=7)
    fund = build_fundamentals(obs, world.close.index, world.symbols)
    close = world.close

    print(f"World: {len(symbols)} symbols x {len(close)} days "
          f"({close.index[0].date()} -> {close.index[-1].date()})")
    print(f"Features: {len(FACTORS)} factors  ->  {', '.join(FACTORS)}")

    X, y, meta = build_feature_panel(close, fund, FACTORS, horizon=21)
    print(f"Panel: {len(X):,} point-in-time (date,asset) rows, "
          f"{X.shape[1]} features, {horizon_span(meta)}-day forward-return labels")

    result = compare_ladder(X, y, meta["label_start"], meta["label_end"],
                            n_splits=6)

    print(f"\nLeak-aware leaderboard (purged {result['n_splits']}-fold CV, "
          f"embargoed; OOS = out-of-sample):")
    print(f"  {'model':<14}{'OOS rank-IC':>12}{'IC t-stat':>11}"
          f"{'LS Sharpe':>11}{'folds':>7}{'OOS preds':>11}")
    print("  " + "-" * 66)
    for r in result["leaderboard"]:
        ic = _fmt(r["oos_rank_ic"], "{:+.4f}")
        t = _fmt(r["oos_rank_ic_tstat"], "{:+.2f}")
        sh = _fmt(r["oos_long_short_sharpe"], "{:+.2f}")
        print(f"  {r['model']:<14}{ic:>12}{t:>11}{sh:>11}"
              f"{r['n_folds']:>7}{r['n_oos_predictions']:>11,}")

    print(f"\nBaseline ({result['baseline']}) OOS rank-IC: "
          f"{_fmt(result['baseline_oos_ic'], '{:+.4f}')}")
    print(f"Best nonlinear ({result['best_nonlinear']}) OOS rank-IC: "
          f"{_fmt(result['best_nonlinear_oos_ic'], '{:+.4f}')}  "
          f"(margin {_fmt(result['ic_margin_over_baseline'], '{:+.4f}')}, "
          f"need >= {result['meaningful_margin']:.3f})")
    print(f"\ncomplexity_beats_linear = {result['complexity_beats_linear']}")
    print(f"\n  {result['verdict']}\n")

    print("Reading: on this synthetic world the planted edge is roughly LINEAR in "
          "quality, so the honest outcome is that boosted trees / net / stack do not "
          "meaningfully beat the elastic-net baseline out-of-sample. Complexity isn't "
          "free — this layer is built to show that, not to hide it.\n")


def horizon_span(meta) -> int:
    return int(meta.get("horizon", 21))


def _fmt(v, spec: str) -> str:
    try:
        if v is None:
            return "n/a"
        import math
        if isinstance(v, float) and math.isnan(v):
            return "n/a"
        return spec.format(v)
    except (TypeError, ValueError):
        return "n/a"


if __name__ == "__main__":
    main()
