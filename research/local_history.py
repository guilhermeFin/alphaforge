"""Small local persistence for anonymous AlphaForge research sessions.

Authenticated workspaces already use the accounts store.  This local store keeps
the desktop app useful before sign-in: it lives under ignored ``data/`` and never
leaves the machine unless the user exports it.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def default_path() -> Path:
    configured = os.environ.get("ALPHAFORGE_HISTORY_PATH")
    return Path(configured) if configured else Path("data") / "alphaforge-workspace.db"


def _summary(kind: str, payload: dict) -> dict:
    if kind == "filing_batch":
        rows = payload.get("features", [])
        return {"documents": len(rows), "earnings_releases": sum(r.get("content_kind") == "earnings_release" for r in rows),
                "average_tone": round(sum(float(r.get("sentiment", 0)) for r in rows) / len(rows), 4) if rows else None}
    if kind == "event_study":
        return {key: payload.get(key) for key in ("n_events", "n_oos", "mse_improvement", "oos_pearson", "evidence_established")}
    if kind == "backtest":
        score = payload.get("scorecard", {})
        summary = {"factor": payload.get("meta", {}).get("factor"), "annual_return": score.get("cagr"),
                   "sharpe": score.get("ann_sharpe"), "verdict": payload.get("verdict")}
        temporal_status = (payload.get("temporal_stability") or {}).get("status")
        if temporal_status:
            summary["temporal_status"] = temporal_status
        return summary
    if kind == "portfolio_research":
        score = payload.get("scorecard", {})
        return {"factor": payload.get("meta", {}).get("factor"), "annual_return": score.get("cagr"),
                "sharpe": score.get("ann_sharpe"), "verdict": payload.get("verdict"),
                "max_name_weight": payload.get("portfolio", {}).get("max_name_weight")}
    if kind == "coverage":
        return {"companies": len(payload.get("symbols", [])), "sec_observations": sum(int(r.get("observations", 0)) for r in payload.get("sec", []))}
    if kind == "microstructure":
        flow = payload.get("trade_flow", {})
        return {"source": payload.get("data_audit", {}).get("source"),
                "events": payload.get("data_audit", {}).get("n_events"),
                "flow_survives_baseline": flow.get("survives_baseline"),
                "research_fingerprint": payload.get("manifest", {}).get("research_fingerprint")}
    if kind == "benchmark_suite":
        rows = payload.get("benchmarks") or []
        return {"benchmarks": len(rows), "provider": (payload.get("meta") or {}).get("provider"),
                "research_fingerprint": (payload.get("manifest") or {}).get("research_fingerprint")}
    return {}


class LocalResearchHistory:
    """Thread-safe, append-only saved research records for the local workspace."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else default_path()
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS research_history (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, kind TEXT NOT NULL, label TEXT NOT NULL, summary TEXT NOT NULL, payload TEXT NOT NULL)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_history_created ON research_history(created_at DESC)")

    def _connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def record(self, kind: str, label: str, payload: dict) -> str:
        now, record_id = datetime.now(timezone.utc).isoformat(timespec="seconds"), uuid4().hex
        encoded = json.dumps(payload, sort_keys=True, default=str)
        with self._lock, self._connection() as conn:
            conn.execute("INSERT INTO research_history (id, created_at, kind, label, summary, payload) VALUES (?, ?, ?, ?, ?, ?)",
                         (record_id, now, kind, label[:160], json.dumps(_summary(kind, payload), sort_keys=True), encoded))
        return record_id

    def list(self, *, limit: int = 100) -> list[dict]:
        with self._lock, self._connection() as conn:
            rows = conn.execute("SELECT id, created_at, kind, label, summary, payload FROM research_history ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [{"id": row["id"], "created_at": row["created_at"], "kind": row["kind"], "label": row["label"],
                 "summary": json.loads(row["summary"]), "payload": json.loads(row["payload"])} for row in rows]
