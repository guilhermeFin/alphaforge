"""SQLite persistence for the accounts layer.

SQLite keeps the MVP dependency-free and file-portable; the schema and the strict
parameterized-query discipline are written so the same store moves to Postgres later
with minimal change (text UUID ids, ISO-8601 UTC timestamps, explicit foreign keys,
no SQLite-only column types).

Security posture baked into every connection:
  * ``foreign_keys = ON``   — referential integrity; cascade deletes power erasure
  * ``journal_mode = WAL``  — safer concurrent reads (skipped for in-memory DBs)
  * ``row_factory = Row``   — name-based access, no positional-index bugs
  * every query in store.py is PARAMETERIZED (no string-built SQL) — no SQL injection

Note: ``audit_log`` deliberately has NO foreign key on ``workspace_id`` so erasure can
delete a workspace's events in a controlled order and leave a system-chain tombstone,
rather than a cascade silently dropping them.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

SCHEMA_VERSION = 1
GENESIS_HASH = "0" * 64  # the prev_hash of the first audit event in any chain


def utcnow() -> str:
    """ISO-8601 UTC, second precision — sortable and timezone-explicit."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


_DDL = """
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);

CREATE TABLE IF NOT EXISTS users (
    id            TEXT PRIMARY KEY,
    idp_subject   TEXT UNIQUE NOT NULL,
    email_enc     TEXT NOT NULL,
    email_bidx    TEXT UNIQUE NOT NULL,
    created_at    TEXT NOT NULL,
    last_seen_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS workspaces (
    id                 TEXT PRIMARY KEY,
    name               TEXT NOT NULL,
    owner_user_id      TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    plan               TEXT NOT NULL DEFAULT 'free',
    stripe_customer_id TEXT,
    created_at         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memberships (
    workspace_id  TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    user_id       TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role          TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    PRIMARY KEY (workspace_id, user_id)
);

CREATE TABLE IF NOT EXISTS api_keys (
    id            TEXT PRIMARY KEY,
    workspace_id  TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    created_by    TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name          TEXT NOT NULL,
    prefix        TEXT NOT NULL,
    key_hash      TEXT UNIQUE NOT NULL,
    created_at    TEXT NOT NULL,
    last_used_at  TEXT,
    revoked_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);

CREATE TABLE IF NOT EXISTS runs (
    id            TEXT PRIMARY KEY,
    workspace_id  TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    user_id       TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    kind          TEXT NOT NULL,
    request_enc   TEXT NOT NULL,
    verdict       TEXT,
    summary_enc   TEXT NOT NULL,
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_ws ON runs(workspace_id, created_at);

CREATE TABLE IF NOT EXISTS audit_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL,
    workspace_id  TEXT,
    actor_user_id TEXT,
    action        TEXT NOT NULL,
    target        TEXT,
    request_id    TEXT,
    ip            TEXT,
    metadata      TEXT NOT NULL,
    prev_hash     TEXT NOT NULL,
    hash          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_ws ON audit_log(workspace_id, id);
"""


def connect(path: str) -> sqlite3.Connection:
    """Open a hardened connection. ``check_same_thread=False`` because FastAPI serves
    requests on a threadpool; the Store guards all access with a single lock."""
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        conn.execute("PRAGMA journal_mode = WAL")
    except sqlite3.OperationalError:
        pass  # :memory: does not support WAL — fine
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(_DDL)
    if conn.execute("SELECT version FROM schema_version").fetchone() is None:
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
    conn.commit()
