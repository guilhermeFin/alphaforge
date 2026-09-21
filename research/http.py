"""Small HTTP decoding helpers for public-data adapters."""
from __future__ import annotations

import gzip
import io
import json
import zlib


def _bounded_gzip(raw: bytes, max_decoded_bytes: int) -> bytes:
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
        decoded = stream.read(max_decoded_bytes + 1)
    if len(decoded) > max_decoded_bytes:
        raise ValueError(f"response expands beyond the {max_decoded_bytes // (1024 * 1024)} MB research limit")
    return decoded


def _bounded_deflate(raw: bytes, max_decoded_bytes: int) -> bytes:
    stream = zlib.decompressobj()
    decoded = stream.decompress(raw, max_decoded_bytes + 1)
    if len(decoded) > max_decoded_bytes or stream.unconsumed_tail:
        raise ValueError(f"response expands beyond the {max_decoded_bytes // (1024 * 1024)} MB research limit")
    decoded += stream.flush()
    if len(decoded) > max_decoded_bytes:
        raise ValueError(f"response expands beyond the {max_decoded_bytes // (1024 * 1024)} MB research limit")
    return decoded


def decode_response_bytes(
    raw: bytes,
    content_encoding: str | None = None,
    *,
    max_decoded_bytes: int | None = None,
) -> bytes:
    """Decompress public API response bytes when needed."""
    encoding = (content_encoding or "").lower()
    if "gzip" in encoding or raw.startswith(b"\x1f\x8b"):
        return _bounded_gzip(raw, max_decoded_bytes) if max_decoded_bytes else gzip.decompress(raw)
    if "deflate" in encoding:
        return _bounded_deflate(raw, max_decoded_bytes) if max_decoded_bytes else zlib.decompress(raw)
    return raw


def decode_json_bytes(raw: bytes, content_encoding: str | None = None) -> dict:
    """Decode JSON responses that public APIs may transparently compress.

    Some public endpoints send gzip even when a client did not explicitly request
    it, so the gzip magic bytes are a fallback alongside the standard header.
    """
    return json.loads(decode_response_bytes(raw, content_encoding).decode("utf-8"))
