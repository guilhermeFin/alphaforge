"""Stable audit fingerprints for local, reproducible research runs."""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
# This module invokes only a resolved local Git executable with constant audit queries.
import subprocess  # nosec B404
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_RUNTIME_PACKAGES = (
    "numpy", "pandas", "scipy", "fastapi", "streamlit", "huggingface-hub",
    "scikit-learn", "xgboost", "lightgbm", "arch", "ruptures",
)


def _file_digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _git_value(*args: str) -> str | None:
    """Return repository state when Git is available, never blocking a study."""
    git_executable = shutil.which("git")
    if git_executable is None:
        return None
    try:
        completed = subprocess.run(
            [git_executable, *args], cwd=_PROJECT_ROOT, check=True,
            capture_output=True, text=True, timeout=2, shell=False,  # nosec B603
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value or None


def runtime_provenance(engine_version: str) -> dict:
    """Capture the executable environment without recording user-specific paths.

    This is operational provenance, not a research result. It lives outside the
    stable data fingerprint so exports answer how a study was produced without
    treating machine details as market evidence.
    """
    packages = {}
    for package in _RUNTIME_PACKAGES:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            continue
    lockfile = _PROJECT_ROOT / "uv.lock"
    return {
        "engine_version": engine_version,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
        },
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "lockfile": {"name": "uv.lock", "sha256": _file_digest(lockfile), "present": lockfile.exists()},
        "git": {
            "commit": _git_value("rev-parse", "HEAD"),
            "dirty": bool(_git_value("status", "--porcelain")),
        },
        "packages": packages,
    }


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _fingerprint_safe(value):
    """Make undefined *derived diagnostics* explicit before canonical JSON.

    Price inputs are validated elsewhere.  A scorecard may nevertheless contain a
    mathematically undefined ratio (for example, a zero-volatility short sample),
    and an audit fingerprint should record that as ``null`` rather than turn a
    valid study into a serialization failure.
    """
    if isinstance(value, dict):
        return {str(key): _fingerprint_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_fingerprint_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _fingerprint_safe(value.tolist())
    if isinstance(value, (float, np.floating)):
        item = float(value)
        return None if not np.isfinite(item) else item
    if isinstance(value, np.integer):
        return int(value)
    return value


def fingerprint(value) -> str:
    payload = json.dumps(_fingerprint_safe(value), sort_keys=True, separators=(",", ":"), default=_json_default, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def frame_fingerprint(frame: pd.DataFrame) -> str:
    """Digest labels and values without writing market data into the audit output."""
    clean = frame.copy().replace([np.inf, -np.inf], np.nan)
    return fingerprint({
        "columns": [str(column) for column in clean.columns], "index": [str(index) for index in clean.index],
        "values": [[None if pd.isna(value) else float(value) for value in row] for row in clean.to_numpy()],
    })


def build_manifest(*, request: dict, meta: dict, close: pd.DataFrame, positions: pd.DataFrame, scorecard: dict, engine_version: str) -> dict:
    """Return a portable manifest; its fingerprint excludes creation wall-clock time."""
    stable = {
        "request": request, "engine_version": engine_version, "data_fingerprint": frame_fingerprint(close),
        "positions_fingerprint": frame_fingerprint(positions), "scorecard": scorecard,
        "period": {"start": meta.get("start_date"), "end": meta.get("end_date")},
    }
    runtime = runtime_provenance(engine_version)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "research_fingerprint": fingerprint(stable), "request_fingerprint": fingerprint(request),
        "data_fingerprint": stable["data_fingerprint"], "positions_fingerprint": stable["positions_fingerprint"],
        "engine_version": engine_version,
        "runtime_provenance": runtime,
        "runtime_fingerprint": fingerprint(runtime),
        "data_snapshot": {"symbols": [str(symbol) for symbol in close.columns], "n_days": int(len(close)),
                          "start_date": meta.get("start_date"), "end_date": meta.get("end_date")},
    }
