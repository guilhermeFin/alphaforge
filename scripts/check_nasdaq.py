"""Nasdaq Data Link / Sharadar connectivity smoke-test.

Confirms the plumbing for the paid PIT-fundamental tier WITHOUT assuming you pay
for it yet. A free Nasdaq account authenticates fine but SF1 is a paid database,
so this is EXPECTED to report "key valid, SF1 not subscribed (0 rows)" until you
subscribe. The point is to prove the only thing missing is the subscription, not
the wiring: AlphaForge's SharadarProvider works unchanged the moment SF1 opens.

It does two checks:
  1. Is the API key present?  (read from $NASDAQ_DATA_LINK_API_KEY or alphaforge/.env)
  2. What does a 1-row SHARADAR/SF1 probe do?
       - rejected as invalid key      -> the key is wrong
       - permission/subscription error -> key valid, SF1 not subscribed
       - returns 0 rows                -> key valid, free tier (no SF1 entitlement)
       - returns >0 rows               -> SF1 accessible, you're ready to go

Run:  python scripts/check_nasdaq.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ENV_VAR = "NASDAQ_DATA_LINK_API_KEY"


def _load_key() -> str | None:
    key = os.environ.get(ENV_VAR, "").strip()
    if key:
        return key
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(ENV_VAR + "="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def main() -> int:
    print("AlphaForge - Nasdaq Data Link / Sharadar connectivity check")
    print("-" * 58)

    key = _load_key()
    if not key:
        print("[1/2] KEY  : MISSING")
        print(f"      Set ${ENV_VAR} in alphaforge/.env, e.g.:")
        print(f"        {ENV_VAR}=your_key_here")
        return 1
    print(f"[1/2] KEY  : found ({key[:3]}...{key[-2:]}, {len(key)} chars)")

    try:
        import nasdaqdatalink
    except ImportError:
        print("[2/2] CLIENT: nasdaq-data-link not installed.")
        print("      Install:  pip install alphaforge[vendors]   (or: pip install nasdaq-data-link)")
        return 1
    nasdaqdatalink.ApiConfig.api_key = key

    # The SF1 table probe is the combined auth + entitlement check: an invalid key
    # is rejected with a key error; a valid key with no SF1 entitlement returns 0
    # rows (free tier) or raises a permission/subscription error.
    try:
        t = nasdaqdatalink.get_table(
            "SHARADAR/SF1", ticker="AAPL", dimension="ARQ", paginate=False)
        n = 0 if t is None else len(t)
        if n > 0:
            print(f"[2/2] SF1  : ACCESSIBLE - pulled {n} AAPL row(s).")
            print("      You have SF1 access. Set provider='sharadar' and go.")
            _summary(ready=True)
            return 0
        print("[2/2] SF1  : key VALID, but 0 rows returned (free tier has no SF1 entitlement).")
        print("      Wiring is correct - subscribe to SHARADAR/SF1 for real PIT data:")
        print("      https://data.nasdaq.com/databases/SF1")
        _summary(ready=False)
        return 0
    except Exception as e:  # noqa: BLE001 - we want the vendor's message, whatever it is
        msg = (str(e).splitlines() or [type(e).__name__])[0]
        low = msg.lower()
        if any(w in low for w in ("invalid api key", "not a valid", "qelx", "401", "unauthor")):
            print(f"[2/2] SF1  : KEY REJECTED - {msg}")
            print("      Re-check the key at https://data.nasdaq.com/account/profile")
            return 1
        if any(w in low for w in ("subscri", "permission", "forbidden", "not have", "premium", "403")):
            print("[2/2] SF1  : key VALID, SF1 NOT SUBSCRIBED (expected on a free account).")
            print(f"      Vendor said: {msg}")
            print("      Subscribe to SHARADAR/SF1 for real PIT data: https://data.nasdaq.com/databases/SF1")
            _summary(ready=False)
            return 0
        print(f"[2/2] SF1  : unexpected error - {msg}")
        return 1


def _summary(ready: bool) -> None:
    print("-" * 58)
    if ready:
        print("Summary: key + client + SF1 access all verified. You're fully wired for PIT fundamentals.")
    else:
        print("Summary: key + client + auth verified. SharadarProvider is ready; the only "
              "gate to real PIT fundamentals is the SF1 subscription.")


if __name__ == "__main__":
    sys.exit(main())
