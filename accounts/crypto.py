"""Field-level encryption, blind indexing, and API-key hashing — the cryptographic
primitives behind "encrypt what we keep".

Design, stated plainly:
  * ONE master key (``ALPHAFORGE_DATA_KEY``) is the only secret an operator manages.
    From it we derive (HKDF-SHA256, distinct ``info`` labels) two INDEPENDENT subkeys:
    a Fernet data-encryption key and an HMAC key for blind indexing. Rotating the
    master rotates both; the subkeys never leave this process.
  * Sensitive fields (email; the user's strategy parameters and results) are stored
    as Fernet tokens — AES-128-CBC + HMAC authentication. A stolen DB file yields
    ciphertext, not data, and any tampering is detected on decrypt.
  * BLIND INDEX: to find a user by email WITHOUT storing it in the clear, we store
    HMAC-SHA256(normalized email) as a deterministic, non-reversible lookup key.
    Equality search works; the plaintext never lands in an index. HMAC (keyed), not a
    bare hash, so the low-entropy email space can't be rainbow-tabled without the key.
  * API KEYS are bearer secrets with 256 bits of entropy: shown ONCE, then only their
    SHA-256 hash + a short display prefix are persisted. Verification is constant-time.
    A DB leak cannot recover a usable key. (A bare hash is sufficient here precisely
    because the input is high-entropy random — unlike emails.)

The master key is REQUIRED in production (fail closed). In dev/test with none set we
generate a RANDOM ephemeral key and warn — data written under it won't decrypt after a
restart, which is the correct, honest dev behaviour, never a silent hardcoded key.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import secrets

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

_log = logging.getLogger("alphaforge.accounts")

API_KEY_PREFIX = "af_live_"   # secret prefix; a live/test split can grow off this later
_API_KEY_BYTES = 32           # 256 bits of entropy in the secret
_DISPLAY_PREFIX_LEN = 16      # how much of the secret we keep for UI display (non-secret)


class CryptoError(RuntimeError):
    """Encryption/decryption failure or a missing required key."""


def _require_master_key() -> bytes:
    raw = os.environ.get("ALPHAFORGE_DATA_KEY", "").strip()
    if raw:
        return raw.encode("utf-8")
    from api.security import is_production  # local import avoids an import cycle
    if is_production():
        raise CryptoError(
            "ALPHAFORGE_DATA_KEY is required in production — refusing to store "
            "customer data without an encryption key (fail closed).")
    eph = secrets.token_urlsafe(32)
    _log.warning("ALPHAFORGE_DATA_KEY not set — using a RANDOM ephemeral dev key; "
                 "data will NOT decrypt after restart. Set the key for persistence.")
    return eph.encode("utf-8")


def _hkdf(master: bytes, info: bytes, length: int = 32) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=length, salt=None, info=info).derive(master)


class Crypto:
    """Holds the derived subkeys; offers encrypt/decrypt, blind-index, and API-key
    helpers. Construct once and share — it is stateless after init and thread-safe
    (Fernet/HMAC operations hold no mutable state)."""

    def __init__(self, master_key: bytes | None = None) -> None:
        master = master_key if master_key is not None else _require_master_key()
        enc = _hkdf(master, b"alphaforge:field-encryption:v1")
        self._index_key = _hkdf(master, b"alphaforge:blind-index:v1")
        self._fernet = Fernet(base64.urlsafe_b64encode(enc))

    # --- field encryption ---------------------------------------------------
    def encrypt(self, plaintext: str) -> str:
        return self._fernet.encrypt(plaintext.encode("utf-8")).decode("ascii")

    def decrypt(self, token: str) -> str:
        try:
            return self._fernet.decrypt(token.encode("ascii")).decode("utf-8")
        except InvalidToken as e:
            raise CryptoError("could not decrypt field (wrong key or tampered data)") from e

    # --- blind index (searchable equality over an encrypted column) ---------
    def blind_index(self, value: str) -> str:
        norm = value.strip().lower().encode("utf-8")
        return hmac.new(self._index_key, norm, hashlib.sha256).hexdigest()

    # --- API keys -----------------------------------------------------------
    @staticmethod
    def new_api_key() -> tuple[str, str, str]:
        """Return (secret, display_prefix, key_hash). The secret is shown to the user
        exactly ONCE; only (prefix, key_hash) are persisted."""
        secret = API_KEY_PREFIX + secrets.token_urlsafe(_API_KEY_BYTES)
        return secret, secret[:_DISPLAY_PREFIX_LEN], Crypto.hash_api_key(secret)

    @staticmethod
    def hash_api_key(secret: str) -> str:
        return hashlib.sha256(secret.encode("utf-8")).hexdigest()

    @staticmethod
    def verify_api_key(secret: str, key_hash: str) -> bool:
        return hmac.compare_digest(Crypto.hash_api_key(secret), key_hash)
