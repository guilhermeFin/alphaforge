"""Tenant-scoped repository over the SQLite schema.

The cardinal rule here is DENY-BY-DEFAULT ISOLATION: every workspace-scoped read takes
a ``workspace_id`` and the SQL always filters on it, so a bug or a forged id in a higher
layer still cannot return another tenant's rows (defense in depth behind authz.py).

Sensitive fields are encrypted via ``Crypto`` before they touch disk and decrypted on
the way out: emails (also blind-indexed for equality lookup) and run parameters/results
(a user's strategy ideas are their IP). API-key secrets are never stored — only a hash.

The audit log is an APPEND-ONLY HASH CHAIN per workspace: each event's hash folds in the
previous event's hash, so an event cannot be quietly altered or deleted without breaking
the chain — the same "logged, not silently mutable" honesty discipline as the trial
ledger. ``verify_audit_chain`` recomputes and detects any tampering.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from uuid import uuid4

from .crypto import Crypto
from .db import GENESIS_HASH, connect, init_db, utcnow
from .models import ApiKey, AuditEvent, Membership, Role, Run, User, Workspace


def _canonical(event: dict) -> str:
    """Deterministic JSON for hashing — sorted keys, no whitespace, str-coerced."""
    return json.dumps(event, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, default=str)


def _chain_hash(prev_hash: str, event: dict) -> str:
    return hashlib.sha256((prev_hash + _canonical(event)).encode("utf-8")).hexdigest()


class Store:
    """SQLite-backed, encrypted, tenant-scoped repository. Hold ONE instance and share
    it; all access is serialized by an internal lock (correct for the MVP scale)."""

    def __init__(self, path: str = ":memory:", crypto: Crypto | None = None) -> None:
        self._conn = connect(path)
        init_db(self._conn)
        self._crypto = crypto or Crypto()
        self._lock = threading.RLock()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------ users
    def upsert_user_by_idp(self, idp_subject: str, email: str) -> User:
        """Create the user on first sight of an IdP subject, else refresh last_seen.
        Email is stored encrypted + blind-indexed; we never store a password."""
        now = utcnow()
        bidx = self._crypto.blind_index(email)
        enc = self._crypto.encrypt(email)
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM users WHERE idp_subject = ?", (idp_subject,)).fetchone()
            if row is None:
                uid = uuid4().hex
                self._conn.execute(
                    "INSERT INTO users (id, idp_subject, email_enc, email_bidx, "
                    "created_at, last_seen_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (uid, idp_subject, enc, bidx, now, now))
            else:
                uid = row["id"]
                # refresh email (it may have changed at the IdP) + last_seen
                self._conn.execute(
                    "UPDATE users SET email_enc = ?, email_bidx = ?, last_seen_at = ? "
                    "WHERE id = ?", (enc, bidx, now, uid))
            self._conn.commit()
        return self.get_user(uid)  # type: ignore[return-value]

    def get_user(self, user_id: str) -> User | None:
        with self._lock:
            r = self._conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return self._user_from_row(r) if r else None

    def find_user_by_email(self, email: str) -> User | None:
        with self._lock:
            r = self._conn.execute(
                "SELECT * FROM users WHERE email_bidx = ?",
                (self._crypto.blind_index(email),)).fetchone()
        return self._user_from_row(r) if r else None

    def _user_from_row(self, r: sqlite3.Row) -> User:
        return User(id=r["id"], idp_subject=r["idp_subject"],
                    email=self._crypto.decrypt(r["email_enc"]),
                    created_at=r["created_at"], last_seen_at=r["last_seen_at"])

    # ------------------------------------------------------------- workspaces
    def create_workspace(self, name: str, owner_user_id: str,
                         plan: str = "free") -> Workspace:
        """Create a workspace and the owner's membership atomically."""
        now, wid = utcnow(), uuid4().hex
        with self._lock:
            self._conn.execute(
                "INSERT INTO workspaces (id, name, owner_user_id, plan, "
                "stripe_customer_id, created_at) VALUES (?, ?, ?, ?, NULL, ?)",
                (wid, name, owner_user_id, plan, now))
            self._conn.execute(
                "INSERT INTO memberships (workspace_id, user_id, role, created_at) "
                "VALUES (?, ?, ?, ?)", (wid, owner_user_id, Role.OWNER.value, now))
            self._conn.commit()
        return self.get_workspace(wid)  # type: ignore[return-value]

    def get_workspace(self, workspace_id: str) -> Workspace | None:
        with self._lock:
            r = self._conn.execute(
                "SELECT * FROM workspaces WHERE id = ?", (workspace_id,)).fetchone()
        if not r:
            return None
        return Workspace(id=r["id"], name=r["name"], owner_user_id=r["owner_user_id"],
                         plan=r["plan"], stripe_customer_id=r["stripe_customer_id"],
                         created_at=r["created_at"])

    def set_stripe_customer(self, workspace_id: str, stripe_customer_id: str) -> None:
        """Store the Stripe customer REFERENCE — never card data (PCI scope stays at Stripe)."""
        with self._lock:
            self._conn.execute("UPDATE workspaces SET stripe_customer_id = ? WHERE id = ?",
                               (stripe_customer_id, workspace_id))
            self._conn.commit()

    # ------------------------------------------------------------ memberships
    def add_member(self, workspace_id: str, user_id: str, role: Role) -> Membership:
        now = utcnow()
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO memberships (workspace_id, user_id, role, "
                "created_at) VALUES (?, ?, ?, ?)", (workspace_id, user_id, role.value, now))
            self._conn.commit()
        return Membership(workspace_id, user_id, role, now)

    def get_membership(self, workspace_id: str, user_id: str) -> Membership | None:
        with self._lock:
            r = self._conn.execute(
                "SELECT * FROM memberships WHERE workspace_id = ? AND user_id = ?",
                (workspace_id, user_id)).fetchone()
        if not r:
            return None
        return Membership(r["workspace_id"], r["user_id"], Role(r["role"]), r["created_at"])

    def list_memberships_for_user(self, user_id: str) -> list[Membership]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM memberships WHERE user_id = ? ORDER BY created_at",
                (user_id,)).fetchall()
        return [Membership(r["workspace_id"], r["user_id"], Role(r["role"]), r["created_at"])
                for r in rows]

    # --------------------------------------------------------------- api keys
    def create_api_key(self, workspace_id: str, created_by: str,
                       name: str) -> tuple[ApiKey, str]:
        """Mint a key. Returns (ApiKey metadata, SECRET). The secret is the ONLY time
        the plaintext exists — caller must surface it once and never persist it."""
        secret, prefix, key_hash = Crypto.new_api_key()
        now, kid = utcnow(), uuid4().hex
        with self._lock:
            self._conn.execute(
                "INSERT INTO api_keys (id, workspace_id, created_by, name, prefix, "
                "key_hash, created_at, last_used_at, revoked_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL)",
                (kid, workspace_id, created_by, name, prefix, key_hash, now))
            self._conn.commit()
        return ApiKey(kid, workspace_id, created_by, name, prefix, now, None, None), secret

    def list_api_keys(self, workspace_id: str) -> list[ApiKey]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM api_keys WHERE workspace_id = ? ORDER BY created_at DESC",
                (workspace_id,)).fetchall()
        return [self._apikey_from_row(r) for r in rows]

    def resolve_api_key(self, secret: str) -> ApiKey | None:
        """Look up an ACTIVE key by its secret (constant-time verify against the stored
        hash), refresh last_used, and return its metadata. None if unknown/revoked."""
        key_hash = Crypto.hash_api_key(secret)
        with self._lock:
            r = self._conn.execute(
                "SELECT * FROM api_keys WHERE key_hash = ? AND revoked_at IS NULL",
                (key_hash,)).fetchone()
            if not r:
                return None
            # constant-time re-verify (the hash column is indexed; this guards against
            # any future change to the lookup that might admit a partial match)
            if not Crypto.verify_api_key(secret, r["key_hash"]):
                return None
            self._conn.execute("UPDATE api_keys SET last_used_at = ? WHERE id = ?",
                               (utcnow(), r["id"]))
            self._conn.commit()
        return self._apikey_from_row(r)

    def revoke_api_key(self, workspace_id: str, key_id: str) -> bool:
        """Revoke a key — scoped to the workspace so one tenant can't revoke another's."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE api_keys SET revoked_at = ? WHERE id = ? AND workspace_id = ? "
                "AND revoked_at IS NULL", (utcnow(), key_id, workspace_id))
            self._conn.commit()
            return cur.rowcount > 0

    def _apikey_from_row(self, r: sqlite3.Row) -> ApiKey:
        return ApiKey(r["id"], r["workspace_id"], r["created_by"], r["name"],
                      r["prefix"], r["created_at"], r["last_used_at"], r["revoked_at"])

    # ------------------------------------------------------------------- runs
    def record_run(self, workspace_id: str, user_id: str, kind: str,
                   request: dict, verdict: str | None, summary: dict) -> Run:
        """Persist a run with its parameters/results ENCRYPTED at rest (strategy IP)."""
        now, rid = utcnow(), uuid4().hex
        with self._lock:
            self._conn.execute(
                "INSERT INTO runs (id, workspace_id, user_id, kind, request_enc, "
                "verdict, summary_enc, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (rid, workspace_id, user_id, kind,
                 self._crypto.encrypt(json.dumps(request)), verdict,
                 self._crypto.encrypt(json.dumps(summary)), now))
            self._conn.commit()
        return Run(rid, workspace_id, user_id, kind, request, verdict, summary, now)

    def list_runs(self, workspace_id: str, limit: int = 50) -> list[Run]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM runs WHERE workspace_id = ? ORDER BY created_at DESC LIMIT ?",
                (workspace_id, int(limit))).fetchall()
        return [self._run_from_row(r) for r in rows]

    def get_run(self, workspace_id: str, run_id: str) -> Run | None:
        with self._lock:
            r = self._conn.execute(
                "SELECT * FROM runs WHERE id = ? AND workspace_id = ?",
                (run_id, workspace_id)).fetchone()
        return self._run_from_row(r) if r else None

    def _run_from_row(self, r: sqlite3.Row) -> Run:
        return Run(r["id"], r["workspace_id"], r["user_id"], r["kind"],
                   json.loads(self._crypto.decrypt(r["request_enc"])), r["verdict"],
                   json.loads(self._crypto.decrypt(r["summary_enc"])), r["created_at"])

    # -------------------------------------------------------- audit (chained)
    def append_audit(self, action: str, *, workspace_id: str | None = None,
                     actor_user_id: str | None = None, target: str | None = None,
                     request_id: str | None = None, ip: str | None = None,
                     metadata: dict | None = None) -> AuditEvent:
        """Append a tamper-evident event to the workspace's hash chain (NULL workspace
        = the system chain). The hash folds in the previous event's hash."""
        ts, meta = utcnow(), metadata or {}
        with self._lock:
            if workspace_id is None:
                prev = self._conn.execute(
                    "SELECT hash FROM audit_log WHERE workspace_id IS NULL "
                    "ORDER BY id DESC LIMIT 1").fetchone()
            else:
                prev = self._conn.execute(
                    "SELECT hash FROM audit_log WHERE workspace_id = ? "
                    "ORDER BY id DESC LIMIT 1", (workspace_id,)).fetchone()
            prev_hash = prev["hash"] if prev else GENESIS_HASH
            body = {"ts": ts, "workspace_id": workspace_id, "actor_user_id": actor_user_id,
                    "action": action, "target": target, "request_id": request_id,
                    "ip": ip, "metadata": meta}
            h = _chain_hash(prev_hash, body)
            cur = self._conn.execute(
                "INSERT INTO audit_log (ts, workspace_id, actor_user_id, action, target, "
                "request_id, ip, metadata, prev_hash, hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (ts, workspace_id, actor_user_id, action, target, request_id, ip,
                 _canonical(meta), prev_hash, h))
            self._conn.commit()
            eid = int(cur.lastrowid)
        return AuditEvent(eid, ts, workspace_id, actor_user_id, action, target,
                          request_id, ip, meta, prev_hash, h)

    def list_audit(self, workspace_id: str | None, limit: int = 200) -> list[AuditEvent]:
        with self._lock:
            if workspace_id is None:
                rows = self._conn.execute(
                    "SELECT * FROM audit_log WHERE workspace_id IS NULL "
                    "ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM audit_log WHERE workspace_id = ? "
                    "ORDER BY id DESC LIMIT ?", (workspace_id, int(limit))).fetchall()
        return [self._audit_from_row(r) for r in rows]

    def verify_audit_chain(self, workspace_id: str | None) -> bool:
        """Recompute the chain in order; True iff every link's hash still matches —
        i.e. no event was altered, inserted, or removed."""
        with self._lock:
            if workspace_id is None:
                rows = self._conn.execute(
                    "SELECT * FROM audit_log WHERE workspace_id IS NULL ORDER BY id").fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM audit_log WHERE workspace_id = ? ORDER BY id",
                    (workspace_id,)).fetchall()
        prev_hash = GENESIS_HASH
        for r in rows:
            body = {"ts": r["ts"], "workspace_id": r["workspace_id"],
                    "actor_user_id": r["actor_user_id"], "action": r["action"],
                    "target": r["target"], "request_id": r["request_id"], "ip": r["ip"],
                    "metadata": json.loads(r["metadata"])}
            if r["prev_hash"] != prev_hash or _chain_hash(prev_hash, body) != r["hash"]:
                return False
            prev_hash = r["hash"]
        return True

    def _audit_from_row(self, r: sqlite3.Row) -> AuditEvent:
        return AuditEvent(int(r["id"]), r["ts"], r["workspace_id"], r["actor_user_id"],
                          r["action"], r["target"], r["request_id"], r["ip"],
                          json.loads(r["metadata"]), r["prev_hash"], r["hash"])

    # ---------------------------------------------------- GDPR / LGPD: export
    def export_workspace(self, workspace_id: str) -> dict:
        """Full, decrypted portable export of everything a workspace holds (data
        access / portability right). API-key SECRETS are NOT included — they don't
        exist in storage — only their non-secret metadata."""
        ws = self.get_workspace(workspace_id)
        if ws is None:
            return {}
        with self._lock:
            members = self._conn.execute(
                "SELECT * FROM memberships WHERE workspace_id = ?", (workspace_id,)).fetchall()
        users = []
        for m in members:
            u = self.get_user(m["user_id"])
            if u:
                users.append({"id": u.id, "email": u.email, "idp_subject": u.idp_subject,
                              "role": m["role"], "joined_at": m["created_at"]})
        return {
            "workspace": {"id": ws.id, "name": ws.name, "plan": ws.plan,
                          "stripe_customer_id": ws.stripe_customer_id,
                          "created_at": ws.created_at},
            "members": users,
            "api_keys": [{"id": k.id, "name": k.name, "prefix": k.prefix,
                          "created_at": k.created_at, "revoked_at": k.revoked_at}
                         for k in self.list_api_keys(workspace_id)],
            "runs": [{"id": r.id, "kind": r.kind, "request": r.request,
                      "verdict": r.verdict, "summary": r.summary, "created_at": r.created_at}
                     for r in self.list_runs(workspace_id, limit=100_000)],
            "audit_log": [{"ts": e.ts, "action": e.action, "target": e.target,
                           "actor_user_id": e.actor_user_id, "metadata": e.metadata}
                          for e in self.list_audit(workspace_id, limit=100_000)],
        }

    # -------------------------------------------------- GDPR / LGPD: erasure
    def delete_workspace(self, workspace_id: str) -> dict:
        """Right-to-erasure: delete the workspace and everything under it. Memberships,
        api_keys, and runs cascade via foreign keys; audit rows have no FK, so we delete
        them explicitly, then write a SYSTEM-chain tombstone recording that erasure
        happened (we keep proof an action occurred without keeping the erased data)."""
        with self._lock:
            n_runs = self._conn.execute(
                "SELECT COUNT(*) AS c FROM runs WHERE workspace_id = ?",
                (workspace_id,)).fetchone()["c"]
            n_audit = self._conn.execute(
                "SELECT COUNT(*) AS c FROM audit_log WHERE workspace_id = ?",
                (workspace_id,)).fetchone()["c"]
            self._conn.execute("DELETE FROM audit_log WHERE workspace_id = ?", (workspace_id,))
            self._conn.execute("DELETE FROM workspaces WHERE id = ?", (workspace_id,))
            self._conn.commit()
        # tombstone on the system chain (no PII, just the fact + counts)
        self.append_audit("workspace.erased", workspace_id=None, target=workspace_id,
                          metadata={"runs_deleted": int(n_runs),
                                    "audit_events_deleted": int(n_audit)})
        return {"workspace_id": workspace_id, "runs_deleted": int(n_runs),
                "audit_events_deleted": int(n_audit)}
