import importlib.util
from pathlib import Path


_PATH = Path(__file__).parents[1] / "scripts" / "check_coverage.py"
_SPEC = importlib.util.spec_from_file_location("check_coverage", _PATH)
assert _SPEC and _SPEC.loader
coverage_gate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(coverage_gate)


def _summary(percent: float) -> dict:
    return {
        "num_statements": 100,
        "num_branches": 0,
        "covered_lines": int(percent),
        "covered_branches": 0,
    }


def _report(*, overall: float = 90.0, module_percent: float = 100.0) -> dict:
    return {
        "totals": _summary(overall),
        "files": {
            module: {"summary": _summary(module_percent)}
            for module in coverage_gate.MODULE_MINIMUMS
        },
    }


def test_coverage_gate_accepts_the_declared_baseline():
    assert coverage_gate.coverage_failures(_report()) == []


def test_coverage_gate_rejects_an_overall_regression():
    failures = coverage_gate.coverage_failures(_report(overall=80.0))
    assert any("overall branch coverage" in failure for failure in failures)


def test_coverage_gate_rejects_a_missing_integrity_module():
    report = _report()
    report["files"].pop("research/temporal_stability.py")
    assert any("temporal_stability" in failure for failure in coverage_gate.coverage_failures(report))
