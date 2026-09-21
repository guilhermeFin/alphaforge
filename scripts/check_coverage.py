"""Enforce AlphaForge's coverage floor from coverage.py's JSON report.

The floor is a regression guard, not a proxy for research quality. Overall
coverage may evolve deliberately, while the modules that defend point-in-time
integrity and report validity retain focused branch-coverage floors.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


# Recorded from the first locked, full-suite CI run: 495 tests, 83% branch
# coverage. The aggregate floor allows one percentage point of normal movement;
# the per-module thresholds protect the integrity controls more directly.
OVERALL_MINIMUM = 82.0
MODULE_MINIMUMS = {
    "research/backtest.py": 94.0,
    "research/event_study.py": 80.0,
    "research/model_time_integrity.py": 90.0,
    "research/reproducibility.py": 70.0,
    "research/research_protocol.py": 80.0,
    "research/research_validity.py": 90.0,
    "research/strategy_report.py": 95.0,
    "research/temporal_stability.py": 90.0,
}


def _percent(summary: dict[str, Any]) -> float:
    statements = int(summary.get("num_statements", 0))
    branches = int(summary.get("num_branches", 0))
    covered_lines = int(summary.get("covered_lines", 0))
    covered_branches = int(summary.get("covered_branches", 0))
    total = statements + branches
    return 100.0 if total == 0 else 100.0 * (covered_lines + covered_branches) / total


def _find_file(files: dict[str, Any], target: str) -> dict[str, Any] | None:
    normalized_target = target.replace("\\", "/")
    for path, report in files.items():
        normalized_path = path.replace("\\", "/")
        if normalized_path == normalized_target or normalized_path.endswith(f"/{normalized_target}"):
            return report
    return None


def coverage_failures(report: dict[str, Any]) -> list[str]:
    """Return clear failures so unit tests and CI share one policy."""
    failures = []
    overall = _percent(report["totals"])
    if overall < OVERALL_MINIMUM:
        failures.append(f"overall branch coverage {overall:.1f}% is below the {OVERALL_MINIMUM:.1f}% floor")

    files = report.get("files", {})
    for module, minimum in MODULE_MINIMUMS.items():
        file_report = _find_file(files, module)
        if file_report is None:
            failures.append(f"coverage report did not include integrity module {module}")
            continue
        actual = _percent(file_report["summary"])
        if actual < minimum:
            failures.append(f"{module} branch coverage {actual:.1f}% is below the {minimum:.1f}% floor")
    return failures


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    path = Path(argv[0]) if argv else Path("coverage.json")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"Coverage gate could not read {path}: {error}", file=sys.stderr)
        return 2

    failures = coverage_failures(report)
    if failures:
        print("Coverage regression guard failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print(f"Coverage guard passed: overall branch coverage {_percent(report['totals']):.1f}%.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
