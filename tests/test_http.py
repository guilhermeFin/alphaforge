import gzip
import json
import zlib

from research.http import decode_json_bytes


def test_decode_json_bytes_handles_plain_and_compressed_payloads():
    payload = {"observations": [{"value": "3.2"}]}
    raw = json.dumps(payload).encode("utf-8")
    assert decode_json_bytes(raw) == payload
    assert decode_json_bytes(gzip.compress(raw), "gzip") == payload
    assert decode_json_bytes(gzip.compress(raw)) == payload  # magic-byte fallback
    assert decode_json_bytes(zlib.compress(raw), "deflate") == payload
