"""Persistent research protocol records with a one-use final holdout stage."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


STAGES = ("exploration", "validation", "final_holdout")


class ProtocolError(ValueError):
    """Raised when a study would silently reuse a declared final holdout."""


class ResearchProtocolStore:
    """Local, append-only study ledger; forks preserve the original audit trail."""

    def __init__(self, path: str | Path = "data/alphaforge-protocols.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS studies (id TEXT PRIMARY KEY, parent_id TEXT, created_at TEXT NOT NULL, hypothesis TEXT NOT NULL, boundaries TEXT NOT NULL)")
            conn.execute("CREATE TABLE IF NOT EXISTS protocol_runs (id TEXT PRIMARY KEY, study_id TEXT NOT NULL, created_at TEXT NOT NULL, stage TEXT NOT NULL, fingerprint TEXT NOT NULL, metadata TEXT NOT NULL)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_protocol_runs_study ON protocol_runs(study_id, created_at)")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def create(self, hypothesis: str, boundaries: dict[str, str]) -> dict:
        if not hypothesis.strip():
            raise ProtocolError("A research hypothesis is required before creating a protocol.")
        required = {"research_end", "validation_end", "final_holdout_end"}
        if set(boundaries) != required:
            raise ProtocolError("Protocol boundaries must define research_end, validation_end, and final_holdout_end.")
        try:
            dates = [datetime.fromisoformat(str(boundaries[key])).date() for key in ("research_end", "validation_end", "final_holdout_end")]
        except ValueError as error:
            raise ProtocolError("Protocol boundaries must be ISO dates (YYYY-MM-DD).") from error
        if not dates[0] < dates[1] < dates[2]:
            raise ProtocolError("Protocol dates must be strictly ordered: research, validation, final holdout.")
        study = {"id": uuid4().hex, "parent_id": None, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "hypothesis": hypothesis.strip(), "boundaries": boundaries}
        with self._connect() as conn:
            conn.execute("INSERT INTO studies VALUES (?, ?, ?, ?, ?)",
                         (study["id"], None, study["created_at"], study["hypothesis"], json.dumps(boundaries, sort_keys=True)))
        return study

    def record(self, study_id: str, stage: str, fingerprint: str, metadata: dict | None = None) -> dict:
        if stage not in STAGES:
            raise ProtocolError(f"stage must be one of {STAGES}")
        if not fingerprint.strip():
            raise ProtocolError("A reproducibility fingerprint is required for a protocol run.")
        with self._connect() as conn:
            exists = conn.execute("SELECT id FROM studies WHERE id = ?", (study_id,)).fetchone()
            if exists is None:
                raise ProtocolError("Study was not found.")
            if stage == "final_holdout":
                consumed = conn.execute("SELECT 1 FROM protocol_runs WHERE study_id = ? AND stage = ?", (study_id, stage)).fetchone()
                if consumed is not None:
                    raise ProtocolError("The final holdout has already been consumed. Fork the study before a new final test.")
            row = {"id": uuid4().hex, "study_id": study_id, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "stage": stage, "fingerprint": fingerprint, "metadata": metadata or {}}
            conn.execute("INSERT INTO protocol_runs VALUES (?, ?, ?, ?, ?, ?)",
                         (row["id"], row["study_id"], row["created_at"], row["stage"], row["fingerprint"], json.dumps(row["metadata"], sort_keys=True, default=str)))
        return row

    def fork(self, study_id: str, hypothesis: str | None = None) -> dict:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM studies WHERE id = ?", (study_id,)).fetchone()
        if row is None:
            raise ProtocolError("Study was not found.")
        child = self.create(hypothesis or row["hypothesis"], json.loads(row["boundaries"]))
        with self._connect() as conn:
            conn.execute("UPDATE studies SET parent_id = ? WHERE id = ?", (study_id, child["id"]))
        child["parent_id"] = study_id
        return child

    def summary(self, study_id: str) -> dict:
        with self._connect() as conn:
            study = conn.execute("SELECT * FROM studies WHERE id = ?", (study_id,)).fetchone()
            runs = conn.execute("SELECT * FROM protocol_runs WHERE study_id = ? ORDER BY created_at", (study_id,)).fetchall()
        if study is None:
            raise ProtocolError("Study was not found.")
        rows = [{"id": row["id"], "stage": row["stage"], "fingerprint": row["fingerprint"], "created_at": row["created_at"], "metadata": json.loads(row["metadata"])} for row in runs]
        return {"id": study["id"], "parent_id": study["parent_id"], "hypothesis": study["hypothesis"],
                "boundaries": json.loads(study["boundaries"]), "runs": rows,
                "final_holdout_available": not any(row["stage"] == "final_holdout" for row in rows)}
