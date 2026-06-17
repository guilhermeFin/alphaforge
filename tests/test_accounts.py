"""The accounts/data layer's security guarantees are executable, not aspirational.

Each test maps to a control in SECURITY.md's data-protection roadmap: encryption at
rest, blind-indexed lookup, hashed API keys, per-workspace isolation (BOLA/BFLA),
tamper-evident audit logging, and GDPR/LGPD export + erasure. The API-level tests also
prove accounts stay OFF (and the anonymous compute flow intact) until enabled.
"""
import pytest

pytest.importorskip("cryptography")

from accounts import AccountService, Role, Store, resolve_principal
from accounts.authz import AuthzError, require_workspace
from accounts.crypto import Crypto

TEST_KEY = b"unit-test-master-key-not-a-real-secret"


@pytest.fixture
def store():
    s = Store(":memory:", Crypto(TEST_KEY))
    yield s
    s.close()


@pytest.fixture
def svc(store):
    return AccountService(store)


# --------------------------------------------------------------- crypto
def test_field_encryption_round_trip():
    c = Crypto(TEST_KEY)
    token = c.encrypt("alice@example.com")
    assert token != "alice@example.com"          # ciphertext, not plaintext
    assert c.decrypt(token) == "alice@example.com"


def test_wrong_key_cannot_decrypt():
    token = Crypto(TEST_KEY).encrypt("secret")
    from accounts.crypto import CryptoError
    with pytest.raises(CryptoError):
        Crypto(b"a-different-master-key").decrypt(token)


def test_blind_index_is_deterministic_and_keyed():
    c1, c2 = Crypto(TEST_KEY), Crypto(TEST_KEY)
    # same value + same key -> same index (so equality lookup works), case/space-insensitive
    assert c1.blind_index("Alice@Example.com ") == c2.blind_index("alice@example.com")
    # a different key yields a different index (rainbow tables need the key)
    assert Crypto(b"other-key").blind_index("alice@example.com") != c1.blind_index("alice@example.com")


def test_api_key_hash_and_verify():
    secret, prefix, key_hash = Crypto.new_api_key()
    assert secret.startswith("af_live_") and secret.startswith(prefix)
    assert key_hash != secret and len(key_hash) == 64       # sha256 hex, not the secret
    assert Crypto.verify_api_key(secret, key_hash) is True
    assert Crypto.verify_api_key(secret + "x", key_hash) is False


# --------------------------------------------------------------- store/service
def test_login_provisions_owner_workspace(svc):
    p = svc.login_with_identity("idp|alice", "alice@example.com")
    assert p.role == Role.OWNER and p.workspace_id and p.email == "alice@example.com"
    # logging in again is idempotent — same user, same workspace, no duplicate
    p2 = svc.login_with_identity("idp|alice", "alice@example.com")
    assert p2.workspace_id == p.workspace_id


def test_api_key_auth_resolves_principal(svc, store):
    p = svc.login_with_identity("idp|alice", "alice@example.com")
    _key, secret = svc.create_api_key(p, "cli")
    rp = resolve_principal(store, f"Bearer {secret}")
    assert rp is not None and rp.workspace_id == p.workspace_id
    assert rp.auth_method == "api_key" and rp.role == Role.OWNER
    # garbage / non-key bearer -> anonymous (None), never a partial identity
    assert resolve_principal(store, "Bearer not-a-real-key") is None
    assert resolve_principal(store, None) is None


def test_revoked_key_denies_access(svc, store):
    p = svc.login_with_identity("idp|alice", "alice@example.com")
    key, secret = svc.create_api_key(p, "cli")
    assert resolve_principal(store, f"Bearer {secret}") is not None
    assert svc.revoke_api_key(p, key.id) is True
    assert resolve_principal(store, f"Bearer {secret}") is None   # immediately dead


def test_email_and_run_are_encrypted_at_rest(svc, store):
    p = svc.login_with_identity("idp|alice", "alice@example.com")
    svc.record_run(p, "backtest", {"factor": "momentum", "idea": "my-secret-alpha"},
                   "CREDIBLE", {"sharpe": 1.4})
    # read the RAW columns straight from sqlite — they must not contain plaintext
    raw_email = store._conn.execute("SELECT email_enc FROM users").fetchone()["email_enc"]
    raw_run = store._conn.execute("SELECT request_enc FROM runs").fetchone()["request_enc"]
    assert "alice@example.com" not in raw_email
    assert "my-secret-alpha" not in raw_run and "momentum" not in raw_run
    # but the service decrypts on the way out
    assert svc.list_runs(p)[0].request["idea"] == "my-secret-alpha"


def test_tenant_isolation_runs_and_authz(svc, store):
    alice = svc.login_with_identity("idp|alice", "alice@example.com")
    bob = svc.login_with_identity("idp|bob", "bob@example.com")
    svc.record_run(alice, "backtest", {"factor": "quality"}, "CREDIBLE", {})
    # bob cannot see alice's runs...
    assert len(svc.list_runs(bob)) == 0
    assert len(svc.list_runs(alice)) == 1
    # ...and cannot be authorized into alice's workspace (BOLA defense)
    with pytest.raises(AuthzError):
        require_workspace(store, bob, alice.workspace_id, Role.VIEWER)


def test_viewer_cannot_create_keys_but_member_can_run(svc, store):
    owner = svc.login_with_identity("idp|owner", "owner@example.com")
    # add a viewer-only user to the SAME workspace
    viewer_user = store.upsert_user_by_idp("idp|viewer", "viewer@example.com")
    store.add_member(owner.workspace_id, viewer_user.id, Role.VIEWER)
    from accounts.identity import Principal
    viewer = Principal(viewer_user.id, viewer_user.email, owner.workspace_id,
                       Role.VIEWER, "session")
    # viewer may read runs (min VIEWER) but not mint keys (min ADMIN)
    assert svc.list_runs(viewer) == []
    with pytest.raises(AuthzError):
        svc.create_api_key(viewer, "nope")


# --------------------------------------------------------------- audit chain
def test_audit_chain_is_valid_and_detects_tampering(svc, store):
    p = svc.login_with_identity("idp|alice", "alice@example.com")
    svc.create_api_key(p, "k1")
    svc.record_run(p, "backtest", {"factor": "momentum"}, "CREDIBLE", {})
    assert svc.audit.verify(p.workspace_id) is True
    assert len(svc.audit.list(p.workspace_id)) >= 3   # login + apikey.create + run.create
    # tamper: silently rewrite an event's action directly in the DB
    store._conn.execute("UPDATE audit_log SET action = 'tampered' WHERE workspace_id = ? "
                        "AND id = (SELECT MIN(id) FROM audit_log WHERE workspace_id = ?)",
                        (p.workspace_id, p.workspace_id))
    store._conn.commit()
    assert svc.audit.verify(p.workspace_id) is False  # the chain catches it


# --------------------------------------------------------------- GDPR / LGPD
def test_export_then_erasure(svc, store):
    p = svc.login_with_identity("idp|alice", "alice@example.com")
    svc.create_api_key(p, "k1")
    svc.record_run(p, "backtest", {"factor": "value"}, "NOT CREDIBLE", {})
    exp = svc.export_account(p)
    assert exp["workspace"]["id"] == p.workspace_id
    assert len(exp["runs"]) == 1 and exp["members"][0]["email"] == "alice@example.com"
    # API-key SECRETS are never in an export (they don't exist in storage)
    assert all("secret" not in k and "key_hash" not in k for k in exp["api_keys"])
    # erasure removes the workspace and everything under it
    res = svc.delete_account(p)
    assert res["runs_deleted"] == 1
    assert store.get_workspace(p.workspace_id) is None
    assert store.list_runs(p.workspace_id) == []
    # a system-chain tombstone records THAT erasure happened (without the erased data)
    assert svc.audit.verify(None) is True
    assert any(e.action == "workspace.erased" for e in svc.audit.list(None))


# =================================================================== API layer
@pytest.fixture
def api(tmp_path):
    """A TestClient with accounts ENABLED against a throwaway DB. Resets the global
    singleton on teardown so the rest of the suite sees accounts OFF again."""
    from fastapi.testclient import TestClient
    import api.main as main
    from api import accounts_api
    service = AccountService(Store(str(tmp_path / "acc.db"), Crypto(TEST_KEY)))
    accounts_api.configure_for_test(service)
    try:
        yield TestClient(main.app)
    finally:
        accounts_api.reset()
        service.store.close()


def test_api_account_endpoints_require_auth(api):
    assert api.get("/account/me").status_code == 401
    assert api.get("/runs").status_code == 401
    assert api.get("/api-keys").status_code == 401


def test_api_dev_login_persists_runs(api):
    dl = api.post("/auth/dev-login", json={"email": "quant@example.com"}).json()
    assert dl["api_key"].startswith("af_live_") and dl["role"] == "owner"
    H = {"Authorization": f"Bearer {dl['api_key']}"}
    assert api.get("/account/me", headers=H).json()["email"] == "quant@example.com"
    # an authenticated backtest lands in the workspace's run history
    r = api.post("/backtest", json={"provider": "synthetic", "factor": "momentum",
                                    "periods": 300, "n_trials": 2}, headers=H)
    assert r.status_code == 200
    runs = api.get("/runs", headers=H).json()["runs"]
    assert len(runs) == 1 and runs[0]["kind"] == "backtest"


def test_api_two_workspaces_are_isolated(api):
    a = api.post("/auth/dev-login", json={"email": "a@example.com"}).json()["api_key"]
    b = api.post("/auth/dev-login", json={"email": "b@example.com"}).json()["api_key"]
    Ha, Hb = {"Authorization": f"Bearer {a}"}, {"Authorization": f"Bearer {b}"}
    r = api.post("/backtest", json={"provider": "synthetic", "factor": "momentum",
                                    "periods": 300, "n_trials": 1}, headers=Ha)
    assert r.status_code == 200
    assert len(api.get("/runs", headers=Ha).json()["runs"]) == 1
    assert len(api.get("/runs", headers=Hb).json()["runs"]) == 0  # b sees nothing of a's


def test_api_ml_compare_rejects_unknown_feature(api):
    dl = api.post("/auth/dev-login", json={"email": "q@example.com"}).json()
    H = {"Authorization": f"Bearer {dl['api_key']}"}
    r = api.post("/ml-compare", json={"provider": "synthetic",
                                      "ml_factors": ["not_a_real_factor"]}, headers=H)
    assert r.status_code == 400 and "unknown ML feature" in r.json()["detail"]


def test_api_export_and_delete(api):
    dl = api.post("/auth/dev-login", json={"email": "q@example.com"}).json()
    H = {"Authorization": f"Bearer {dl['api_key']}"}
    api.post("/backtest", json={"provider": "synthetic", "factor": "momentum",
                                "periods": 300, "n_trials": 1}, headers=H)
    exp = api.get("/account/export", headers=H).json()
    assert exp["workspace"]["id"] == dl["workspace_id"] and len(exp["runs"]) == 1
    assert api.delete("/account", headers=H).json()["runs_deleted"] == 1
    # the key is gone with the workspace -> back to 401
    assert api.get("/account/me", headers=H).status_code == 401


def test_accounts_disabled_by_default(monkeypatch):
    """With no env config the account surface is OFF (503) and the anonymous compute
    flow is completely unaffected."""
    from fastapi.testclient import TestClient
    import api.main as main
    from api import accounts_api
    monkeypatch.delenv("ALPHAFORGE_ACCOUNTS", raising=False)
    monkeypatch.delenv("ALPHAFORGE_DB_PATH", raising=False)
    accounts_api.reset()
    try:
        c = TestClient(main.app)
        assert c.get("/account/me").status_code == 503
        r = c.post("/backtest", json={"provider": "synthetic", "factor": "momentum",
                                      "periods": 300, "n_trials": 1})
        assert r.status_code == 200  # anonymous backtest still works
    finally:
        accounts_api.reset()
