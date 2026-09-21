import pandas as pd

from research.paper import LocalPaperBook, rebalance_plan
from research.reproducibility import build_manifest, runtime_provenance
from research.reproducibility import fingerprint


def _frames():
    index = pd.bdate_range("2024-01-02", periods=3)
    close = pd.DataFrame({"A": [100.0, 101.0, 102.0], "B": [100.0, 99.0, 98.0]}, index=index)
    positions = pd.DataFrame({"A": [0.0, 0.2, 0.2], "B": [0.0, -0.2, -0.2]}, index=index)
    return close, positions


def test_manifest_fingerprint_is_stable_for_same_run():
    close, positions = _frames()
    kwargs = {"request": {"factor": "momentum"}, "meta": {"start_date": "2024-01-02", "end_date": "2024-01-04"},
              "close": close, "positions": positions, "scorecard": {"cagr": 0.1}, "engine_version": "x"}
    assert build_manifest(**kwargs)["research_fingerprint"] == build_manifest(**kwargs)["research_fingerprint"]


def test_fingerprint_records_undefined_derived_diagnostics_without_failing():
    assert fingerprint({"undefined_ratio": float("nan"), "nested": [float("inf")]})


def test_manifest_captures_portable_runtime_provenance():
    close, positions = _frames()
    manifest = build_manifest(
        request={"factor": "momentum"}, meta={}, close=close, positions=positions,
        scorecard={}, engine_version="x",
    )
    runtime = manifest["runtime_provenance"]
    assert runtime["python"]["major_minor"]
    assert runtime["lockfile"]["name"] == "uv.lock"
    assert len(manifest["runtime_fingerprint"]) == 64
    assert "executable" not in str(runtime).lower()


def test_runtime_provenance_stays_json_serializable():
    assert runtime_provenance("x")["engine_version"] == "x"


def test_local_paper_book_records_historical_plan(tmp_path):
    _close, positions = _frames()
    plan = rebalance_plan(positions, capital=100_000, research_fingerprint="abc")
    book = LocalPaperBook(tmp_path / "paper.json")
    saved = book.record(plan)
    assert saved["id"]
    assert len(book.list()) == 1
    assert book.list()[0]["orders"][0]["symbol"] == "A"
