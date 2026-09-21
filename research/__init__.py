"""AlphaForge research engine — the crown jewel.

Pure, tested Python. Three non-negotiable honesty guards live here and must never
be weakened (see backtest.py):
  1. Signals are shifted before they trade — no look-ahead / point-in-time safe.
  2. Transaction costs are charged on turnover — no frictionless fantasy.
  3. Performance is judged with fat-tail- and multiple-testing-aware statistics —
     a high Sharpe is not believed until it survives the Probabilistic / Deflated
     Sharpe Ratio (Bailey & López de Prado), which use the return skew & kurtosis.

Correctness > cleverness. A single embarrassing false-positive backtest damages
the brand more than a missing feature.
"""

from dotenv import load_dotenv

# Local convenience only: deployment environment variables always win because
# override=False.  The ignored .env file keeps credentials out of source control.
load_dotenv(override=False)

from . import (data, factors, metrics, backtest, walkforward, stats_guards,
               fundamentals, providers, signal_quality, factor_lib, overfitting,
               trial_ledger, attribution, econometrics, macro, text_features,
               event_study, local_history, portfolio, execution, validation,
               reproducibility, paper, licensed_data, risk, statistical_rigor,
               strategy_report, benchmark, tail_risk, evidence, robustness_matrix,
               factor_diagnostics, research_protocol, protocol_runner, benchmark_suite, microstructure, temporal_stability)

__all__ = ["data", "factors", "metrics", "backtest", "walkforward", "stats_guards",
           "fundamentals", "providers", "signal_quality", "factor_lib",
           "overfitting", "trial_ledger", "attribution", "econometrics", "macro",
           "text_features", "event_study", "local_history", "portfolio", "execution",
           "validation", "reproducibility", "paper", "licensed_data", "risk",
           "statistical_rigor", "strategy_report", "benchmark", "tail_risk"]
__all__ += ["evidence", "robustness_matrix", "factor_diagnostics", "research_protocol", "protocol_runner", "benchmark_suite", "microstructure", "temporal_stability"]
__version__ = "0.0.1"
