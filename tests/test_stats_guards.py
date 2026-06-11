import numpy as np
import pandas as pd

from research import stats_guards as g


def test_base_rate_fallacy():
    # 1% event, "95% accurate" signal -> precision is low, recall is high
    out = g.base_rate_precision(base_rate=0.01, true_positive_rate=0.95, false_positive_rate=0.05)
    assert out["recall_tpr"] == 0.95
    assert out["precision"] < 0.25       # the fallacy made concrete
    assert out["lift"] > 1.0


def test_simpson_reversal_detected():
    # within each group treatment=1 beats treatment=0, but pooled it loses
    rows = []
    # group X: mostly treatment=1, modest outcomes; group Y: mostly treatment=0, high outcomes
    for _ in range(90):
        rows.append(("X", 1, 0.6))
    for _ in range(10):
        rows.append(("X", 0, 0.5))
    for _ in range(10):
        rows.append(("Y", 1, 0.95))
    for _ in range(90):
        rows.append(("Y", 0, 0.9))
    df = pd.DataFrame(rows, columns=["grp", "treat", "out"])
    res = g.simpson_reversal(df, group="grp", treatment="treat", outcome="out")
    assert res["within_group_consistent"] is True
    assert res["simpson_reversal"] is True


def test_lookahead_warning():
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.standard_normal(500))
    cheat = rets.copy()             # signal_t = return_t  (peeks)
    honest = rets.shift(1).fillna(0.0)  # signal_t = return_{t-1} (past only)
    assert g.lookahead_warning(cheat, rets)["suspected_lookahead"] is True
    assert g.lookahead_warning(honest, rets)["suspected_lookahead"] is False


def test_fat_tail_report_flags_student_t():
    rng = np.random.default_rng(0)
    heavy = rng.standard_t(3, size=2000)
    rep = g.fat_tail_report(heavy)
    assert rep["returns_are_normal"] is False
    assert rep["excess_kurtosis"] > 1.0
