"""Append-only local BBO capture primitives for future quote-level studies.

The recorder intentionally has no exchange credentials, network client, or broker
actions.  A small adapter can feed validated quote snapshots into this store while
the audit trail remains local and inspectable.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .lab import MicrostructureError


QUOTE_COLUMNS = ("timestamp", "symbol", "bid_price", "bid_size", "ask_price", "ask_size", "source")


def validate_quote_events(events: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in QUOTE_COLUMNS if column not in events.columns]
    if missing:
        raise MicrostructureError(f"quote events are missing required columns: {', '.join(missing)}")
    clean = events.loc[:, QUOTE_COLUMNS].copy()
    clean["timestamp"] = pd.to_datetime(clean["timestamp"], utc=True, errors="coerce")
    for column in ("bid_price", "bid_size", "ask_price", "ask_size"):
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    clean["symbol"] = clean["symbol"].astype(str).str.upper().str.strip()
    clean["source"] = clean["source"].astype(str).str.strip()
    if clean.isna().any().any() or (clean[["bid_price", "bid_size", "ask_price", "ask_size"]] <= 0).any().any():
        raise MicrostructureError("quote timestamps and bid/ask prices and sizes must be present and strictly positive")
    if (clean["bid_price"] >= clean["ask_price"]).any():
        raise MicrostructureError("crossed or locked quotes are not accepted into the local audit store")
    if not clean["timestamp"].is_monotonic_increasing:
        raise MicrostructureError("quote events must be timestamp-ordered before they enter the audit store")
    if clean.duplicated().any():
        raise MicrostructureError("duplicate quote events are not accepted into the audit store")
    return clean.reset_index(drop=True)


class LocalQuoteStore:
    """Small append-only JSONL store for locally captured quote snapshots."""

    def __init__(self, path: str | Path = "data/alphaforge-quotes.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, events: pd.DataFrame) -> int:
        clean = validate_quote_events(events)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            for record in clean.assign(timestamp=clean["timestamp"].astype(str)).to_dict(orient="records"):
                record["recorded_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        return int(len(clean))

    def read(self, *, limit: int | None = None) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=QUOTE_COLUMNS)
        rows = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if limit is not None:
            rows = rows[-max(0, limit):]
        return pd.DataFrame(rows)

    def status(self) -> dict:
        frame = self.read()
        if frame.empty:
            return {"configured": self.path.exists(), "records": 0, "quote_level_evidence": False,
                    "note": "No local quote snapshots have been captured yet."}
        clean = validate_quote_events(frame)
        return {
            "configured": True, "records": int(len(clean)), "quote_level_evidence": True,
            "start": clean["timestamp"].iloc[0].isoformat(), "end": clean["timestamp"].iloc[-1].isoformat(),
            "symbols": sorted(clean["symbol"].unique().tolist()),
            "note": "Local BBO history is available for future quote-level research; coverage remains limited to recorded sessions.",
        }
