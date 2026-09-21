import gzip

import pytest

from research.http import decode_response_bytes


def test_bounded_gzip_decoding_accepts_a_large_but_allowed_response():
    raw = gzip.compress(b"x" * 1_024)
    assert decode_response_bytes(raw, "gzip", max_decoded_bytes=1_024) == b"x" * 1_024


def test_bounded_gzip_decoding_rejects_a_compressed_expansion_beyond_the_limit():
    raw = gzip.compress(b"x" * 1_025)
    with pytest.raises(ValueError, match="research limit"):
        decode_response_bytes(raw, "gzip", max_decoded_bytes=1_024)
