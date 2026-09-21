"""Small Nasdaq ITCH sample parser and top-of-book reconstruction aid.

It is intentionally a correctness and data-engineering tool.  Sample data from one
venue/day is never promoted to broad market evidence.
"""
from __future__ import annotations

import struct

import pandas as pd

from .lab import MicrostructureError


def _timestamp(raw: bytes) -> int:
    return int.from_bytes(raw, byteorder="big", signed=False)


def _stock(raw: bytes) -> str:
    return raw.decode("ascii").strip()


def parse_itch_messages(payload: bytes) -> list[dict]:
    """Parse framed ITCH 5.0 Add/Cancel/Delete/Execute/Replace messages.

    Unknown message types remain explicit records, preserving sequence alignment
    rather than being discarded invisibly.
    """
    offset, records = 0, []
    while offset < len(payload):
        if offset + 2 > len(payload):
            raise MicrostructureError("truncated ITCH frame length")
        length = struct.unpack_from("!H", payload, offset)[0]
        offset += 2
        if length < 1 or offset + length > len(payload):
            raise MicrostructureError("truncated or invalid ITCH frame")
        message = payload[offset:offset + length]
        offset += length
        kind = chr(message[0])
        if kind == "A":
            _, locate, tracking, timestamp, reference, side, shares, stock, price = struct.unpack("!cHH6sQcI8sI", message)
            records.append({"type": "add", "stock_locate": locate, "tracking": tracking, "timestamp_ns": _timestamp(timestamp),
                            "reference": reference, "side": side.decode(), "shares": shares, "symbol": _stock(stock), "price": price / 10_000.0})
        elif kind == "F":
            _, locate, tracking, timestamp, reference, side, shares, stock, price, attribution = struct.unpack("!cHH6sQcI8sI4s", message)
            records.append({"type": "add", "stock_locate": locate, "tracking": tracking, "timestamp_ns": _timestamp(timestamp),
                            "reference": reference, "side": side.decode(), "shares": shares, "symbol": _stock(stock), "price": price / 10_000.0,
                            "attribution": _stock(attribution)})
        elif kind in {"E", "C"}:
            if kind == "E":
                _, locate, tracking, timestamp, reference, shares, match = struct.unpack("!cHH6sQIQ", message)
            else:
                _, locate, tracking, timestamp, reference, shares, match, _, _ = struct.unpack("!cHH6sQIQcI", message)
            records.append({"type": "execute", "stock_locate": locate, "tracking": tracking, "timestamp_ns": _timestamp(timestamp),
                            "reference": reference, "shares": shares, "match_number": match})
        elif kind == "X":
            _, locate, tracking, timestamp, reference, shares = struct.unpack("!cHH6sQI", message)
            records.append({"type": "cancel", "stock_locate": locate, "tracking": tracking, "timestamp_ns": _timestamp(timestamp),
                            "reference": reference, "shares": shares})
        elif kind == "D":
            _, locate, tracking, timestamp, reference = struct.unpack("!cHH6sQ", message)
            records.append({"type": "delete", "stock_locate": locate, "tracking": tracking, "timestamp_ns": _timestamp(timestamp), "reference": reference})
        elif kind == "U":
            _, locate, tracking, timestamp, old_reference, new_reference, shares, price = struct.unpack("!cHH6sQQII", message)
            records.append({"type": "replace", "stock_locate": locate, "tracking": tracking, "timestamp_ns": _timestamp(timestamp),
                            "reference": old_reference, "new_reference": new_reference, "shares": shares, "price": price / 10_000.0})
        else:
            records.append({"type": "unhandled", "message_type": kind, "raw_length": length})
    return records


def reconstruct_top_of_book(records: list[dict]) -> pd.DataFrame:
    """Replay visible-order messages into a top-of-book audit table."""
    orders: dict[int, dict] = {}
    snapshots: list[dict] = []
    for record in records:
        event = record["type"]
        symbol = record.get("symbol")
        if event == "add":
            orders[int(record["reference"])] = {key: record[key] for key in ("symbol", "side", "shares", "price")}
        elif event in {"execute", "cancel"} and int(record["reference"]) in orders:
            order = orders[int(record["reference"])]
            symbol = order["symbol"]
            order["shares"] -= int(record["shares"])
            if order["shares"] <= 0:
                del orders[int(record["reference"])]
        elif event == "delete":
            old = orders.pop(int(record["reference"]), None)
            symbol = old["symbol"] if old else None
        elif event == "replace" and int(record["reference"]) in orders:
            old = orders.pop(int(record["reference"]))
            symbol = old["symbol"]
            orders[int(record["new_reference"])] = {**old, "shares": int(record["shares"]), "price": float(record["price"])}
        if event == "unhandled" or "timestamp_ns" not in record:
            continue
        if not symbol:
            continue
        visible = [order for order in orders.values() if order["symbol"] == symbol]
        bids = [order["price"] for order in visible if order["side"] == "B"]
        asks = [order["price"] for order in visible if order["side"] == "S"]
        snapshots.append({"timestamp_ns": record["timestamp_ns"], "symbol": symbol,
                          "best_bid": max(bids) if bids else None, "best_ask": min(asks) if asks else None,
                          "visible_orders": len(visible), "event": event})
    return pd.DataFrame(snapshots)
