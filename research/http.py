"""Small HTTP decoding helpers for public-data adapters."""
from __future__ import annotations

import gzip
import json
import zlib


def decode_json_bytes(raw: bytes, content_encoding: str | None = None) -> dict:
    """Decode JSON responses that public APIs may transparently compress.

    Some public endpoints send gzip even when a client did not explicitly request
    it, so the gzip magic bytes are a fallback alongside the standard header.
    """
    encoding = (content_encoding or "").lower()
    if "gzip" in encoding or raw.startswith(b"\x1f\x8b"):
        raw = gzip.decompress(raw)
    elif "deflate" in encoding:
        raw = zlib.decompress(raw)
    return json.loads(raw.decode("utf-8"))
