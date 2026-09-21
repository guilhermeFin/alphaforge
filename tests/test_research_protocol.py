import pytest

from research.research_protocol import ProtocolError, ResearchProtocolStore


def test_final_holdout_is_single_use_and_a_fork_starts_clean(tmp_path):
    store = ResearchProtocolStore(tmp_path / "protocol.db")
    study = store.create("Quality persists after costs", {
        "research_end": "2022-12-31", "validation_end": "2023-12-31", "final_holdout_end": "2024-12-31",
    })
    store.record(study["id"], "final_holdout", "fingerprint-1")
    with pytest.raises(ProtocolError, match="already been consumed"):
        store.record(study["id"], "final_holdout", "fingerprint-2")
    child = store.fork(study["id"])
    assert store.summary(child["id"])["final_holdout_available"]


def test_protocol_rejects_unordered_boundaries(tmp_path):
    store = ResearchProtocolStore(tmp_path / "protocol.db")
    with pytest.raises(ProtocolError, match="strictly ordered"):
        store.create("Test", {"research_end": "2024-01-01", "validation_end": "2023-01-01", "final_holdout_end": "2025-01-01"})
