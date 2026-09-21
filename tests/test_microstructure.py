import io
import struct

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.service import WorkflowError, run_microstructure_lab
from research.microstructure import (
    MicrostructureError,
    evaluate_trade_flow,
    load_binance_trades,
    options_diagnostics,
    run_microstructure_study,
    simulate_quoting,
    synthetic_trade_events,
    validate_trade_events,
    LocalQuoteStore,
    parse_itch_messages,
    reconstruct_top_of_book,
    validate_quote_events,
)
from research.local_history import LocalResearchHistory


def test_synthetic_study_is_json_safe_and_never_market_evidence():
    result = run_microstructure_study(seed=4, window=24, horizon_events=20)
    assert result["data_audit"]["status"] == "engine_validation"
    assert result["data_audit"]["quote_level_evidence"] is False
    assert "trade-price markout" in result["trade_flow"]["definition"]
    assert result["manifest"]["research_fingerprint"]
    assert len(result["simulation"]["outcomes"]) == 3
    assert all(row["simulation_only"] for row in result["simulation"]["outcomes"])


def test_event_validation_refuses_out_of_order_and_duplicate_rows():
    events = synthetic_trade_events(n_events=140)
    with pytest.raises(MicrostructureError, match="timestamp-ordered"):
        validate_trade_events(events.iloc[::-1].reset_index(drop=True))
    duplicate = pd.concat([events, events.iloc[[0]]], ignore_index=True)
    with pytest.raises(MicrostructureError, match="duplicate"):
        validate_trade_events(duplicate)


def test_binance_importer_normalizes_trade_schema_and_microsecond_time():
    source = io.StringIO(
        "time,price,qty,isBuyerMaker\n"
        + "\n".join(f"1735689600{i:06d},{100 + i * .01},1.0,{str(i % 2 == 0).lower()}" for i in range(140))
    )
    events = load_binance_trades(source)
    assert list(events.columns) == ["timestamp", "price", "quantity", "is_buyer_maker"]
    assert str(events["timestamp"].dtype).startswith("datetime64")
    assert events["timestamp"].is_monotonic_increasing


def test_flow_evaluation_uses_chronological_final_holdout_and_baseline():
    events = synthetic_trade_events(seed=12)
    result = evaluate_trade_flow(events, window=24, horizon_events=20)
    assert result["splits"]["train_events"] > result["splits"]["final_holdout_events"]
    final_models = {row["model"] for row in result["results"] if row["stage"] == "final_holdout"}
    assert final_models == {"last_return", "flow_imbalance"}
    assert result["calibration"]


def test_simulator_labels_its_assumptions_and_respects_inventory_cap():
    output = simulate_quoting(synthetic_trade_events(seed=8), window=20, horizon_events=15, max_inventory=3)
    assert "simulation" in output["label"].lower()
    assert all(abs(row["ending_inventory"]) <= 3 for row in output["outcomes"])
    assert output["assumptions"]["max_inventory"] == 3


def test_options_diagnostics_solves_iv_and_rejects_invalid_contracts():
    result = options_diagnostics([
        {"contract_id": "call", "option_type": "call", "spot": 100, "strike": 100, "maturity_years": 0.25, "market_price": 5.5},
        {"contract_id": "bad", "option_type": "put", "spot": 0, "strike": 100, "maturity_years": 0.25, "market_price": 5.5},
    ])
    assert result["contracts"][0]["status"] == "ok"
    assert 0.0 < result["contracts"][0]["implied_volatility"] < 5.0
    assert result["contracts"][1]["status"].startswith("invalid")


def test_service_and_api_enforce_bounded_events_without_paths():
    result = run_microstructure_lab({"source": "synthetic", "window_events": 20, "horizon_events": 15})
    assert result["research_question"]
    with pytest.raises(WorkflowError, match="20,000"):
        run_microstructure_lab({"source": "binance_events", "events": [{}] * 20_001})
    response = TestClient(app).post("/microstructure-study", json={"source": "synthetic", "window_events": 20, "horizon_events": 15})
    assert response.status_code == 200, response.text
    assert "NaN" not in response.text
    assert TestClient(app).post("/microstructure-study", json={"source": "synthetic", "unknown": True}).status_code == 422


def test_local_quote_store_rejects_crossed_data_and_preserves_local_audit(tmp_path):
    quotes = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=2, freq="s", tz="UTC"), "symbol": ["BTCUSDT", "BTCUSDT"],
        "bid_price": [99.0, 99.5], "bid_size": [1.0, 2.0], "ask_price": [101.0, 101.5], "ask_size": [1.5, 1.0],
        "source": ["binance_ws", "binance_ws"],
    })
    store = LocalQuoteStore(tmp_path / "quotes.jsonl")
    assert store.append(quotes) == 2
    assert store.status()["quote_level_evidence"] is True
    crossed = quotes.copy()
    crossed.loc[0, "ask_price"] = 99.0
    with pytest.raises(MicrostructureError, match="crossed"):
        validate_quote_events(crossed)


def test_itch_parser_reconstructs_top_of_book_from_framed_sample_messages():
    def frame(message: bytes) -> bytes:
        return struct.pack("!H", len(message)) + message

    timestamp = (1).to_bytes(6, "big")
    bid = struct.pack("!cHH6sQcI8sI", b"A", 1, 1, timestamp, 101, b"B", 100, b"TEST    ", 10_000)
    ask = struct.pack("!cHH6sQcI8sI", b"A", 1, 2, (2).to_bytes(6, "big"), 102, b"S", 100, b"TEST    ", 10_100)
    cancel = struct.pack("!cHH6sQI", b"X", 1, 3, (3).to_bytes(6, "big"), 102, 100)
    records = parse_itch_messages(frame(bid) + frame(ask) + frame(cancel))
    book = reconstruct_top_of_book(records)
    assert book.iloc[1]["best_bid"] == 1.0
    assert book.iloc[1]["best_ask"] == 1.01
    assert pd.isna(book.iloc[-1]["best_ask"])


def test_microstructure_study_can_join_local_research_history(tmp_path):
    study = run_microstructure_study(window=20, horizon_events=15)
    history = LocalResearchHistory(tmp_path / "history.db")
    history.record("microstructure", "synthetic study", study)
    saved = history.list()[0]
    assert saved["summary"]["source"] == "synthetic"
    assert saved["summary"]["events"] == study["data_audit"]["n_events"]
