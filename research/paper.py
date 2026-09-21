"""Local paper-rebalance records. No brokerage calls, credentials, or orders."""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd


def default_path() -> Path:
    configured = os.environ.get("ALPHAFORGE_PAPER_PATH")
    return Path(configured) if configured else Path("data") / "alphaforge-paper-book.json"


def rebalance_plan(positions: pd.DataFrame, *, capital: float, research_fingerprint: str) -> dict:
    """Make a final-historical-date target plan, never a current-price order."""
    if positions.empty or capital <= 0:
        raise ValueError("a non-empty portfolio and positive capital are required")
    last = positions.iloc[-1]
    orders = [
        {"symbol": str(symbol), "target_weight": float(weight), "target_notional": float(weight * capital),
         "side": "long" if weight > 0 else "short"}
        for symbol, weight in last.items() if abs(float(weight)) > 1e-10
    ]
    return {
        "as_of": str(positions.index[-1].date()) if hasattr(positions.index[-1], "date") else str(positions.index[-1]),
        "capital": float(capital), "research_fingerprint": research_fingerprint,
        "gross_exposure": float(last.abs().sum()), "net_exposure": float(last.sum()), "orders": orders,
        "note": "Historical paper-rebalance plan only. It is not a live quote, recommendation, or broker order.",
    }


class LocalPaperBook:
    """Append-only local JSON store for paper-rebalance plans."""
    _lock = threading.RLock()

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else default_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (OSError, ValueError):
            return []

    def record(self, plan: dict) -> dict:
        entry = {**plan, "id": uuid4().hex, "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        with self._lock:
            records = self._read()
            records.append(entry)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")
            temporary.replace(self.path)
        return entry

    def list(self, *, limit: int = 50) -> list[dict]:
        return list(reversed(self._read()[-limit:]))
