import json

import numpy as np
import pandas as pd
import pytest

from api.service import run_filing_event_study
from research.model_time_integrity import (
    ELIGIBLE,
    ENV_FALLBACK_AVAILABLE_AT,
    ENV_FALLBACK_TRAINING_CUTOFF,
    ENV_OVERRIDES,
    HISTORICALLY_UNAVAILABLE_MODEL,
    MISSING_MODEL_METADATA,
    TimeIntegrityPolicy,
    classify_observation,
    resolve_profile,
    time_integrity_audit,
)

FINBERT = "ProsusAI/finbert"


@pytest.fixture(autouse=True)
def _clear_model_time_env(monkeypatch):
    """Operator configuration must not leak between tests."""
    for name in (ENV_OVERRIDES, ENV_FALLBACK_AVAILABLE_AT, ENV_FALLBACK_TRAINING_CUTOFF):
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------- metadata


def test_registered_finbert_dates_carry_status_source_and_rationale():
    profile = resolve_profile(FINBERT)
    available = profile.model_available_at
    cutoff = profile.training_data_cutoff

    assert available.value == pd.Timestamp("2020-12-24")
    assert available.status == "documented"
    assert "huggingface.co/ProsusAI/finbert" in available.source
    assert available.rationale and not available.is_assumption

    assert cutoff.value == pd.Timestamp("2018-12-31")
    assert cutoff.status == "inferred"
    assert "1908.10063" in cutoff.source and "1810.04805" in cutoff.source
    assert profile.metadata_source == "registry"
    assert profile.assumptions == []


def test_availability_and_training_cutoff_are_separate_concepts():
    profile = resolve_profile(FINBERT)
    assert profile.model_available_at.value != profile.training_data_cutoff.value


# ------------------------------------------------------- classification


def test_document_after_release_is_eligible_without_overlap_risk():
    verdict = classify_observation("2024-03-01", resolve_profile(FINBERT))
    assert verdict.status == ELIGIBLE
    assert verdict.training_overlap_risk is False
    assert verdict.included_in_primary is True


def test_document_on_the_release_date_is_eligible():
    verdict = classify_observation("2020-12-24", resolve_profile(FINBERT))
    assert verdict.status == ELIGIBLE


def test_document_before_release_is_historically_unavailable_not_leakage():
    verdict = classify_observation("2019-06-01", resolve_profile(FINBERT))
    assert verdict.status == HISTORICALLY_UNAVAILABLE_MODEL
    assert verdict.included_in_primary is False
    assert "historically deployable" in verdict.reason
    assert "leak" not in verdict.reason.lower()
    assert "memoriz" not in verdict.reason.lower()


def test_document_before_training_cutoff_is_flagged_as_risk_not_proof():
    verdict = classify_observation("2015-06-01", resolve_profile(FINBERT))
    assert verdict.training_overlap_risk is True
    assert "overlap risk" in verdict.reason
    assert "not evidence of memorization" in verdict.reason


def test_document_between_cutoff_and_release_has_no_overlap_risk():
    """2019 post-dates the training corpus but pre-dates the published weights."""
    verdict = classify_observation("2019-06-01", resolve_profile(FINBERT))
    assert verdict.status == HISTORICALLY_UNAVAILABLE_MODEL
    assert verdict.training_overlap_risk is False


def test_unknown_model_yields_missing_metadata_and_invents_no_date():
    profile = resolve_profile("some-vendor/unreleased-scorer")
    assert profile.model_available_at.value is None
    assert profile.metadata_source == "unknown"
    verdict = classify_observation("2024-01-02", profile)
    assert verdict.status == MISSING_MODEL_METADATA
    assert verdict.included_in_primary is False


def test_blank_model_name_is_missing_metadata():
    verdict = classify_observation("2024-01-02", resolve_profile(""))
    assert verdict.status == MISSING_MODEL_METADATA


def test_unusable_document_date_is_missing_metadata_not_eligible():
    verdict = classify_observation(None, resolve_profile(FINBERT))
    assert verdict.status == MISSING_MODEL_METADATA
    assert verdict.included_in_primary is False


# ------------------------------------------------------------- policy


def test_default_policy_excludes_unavailable_and_missing_but_keeps_overlap():
    policy = TimeIntegrityPolicy()
    assert policy.exclude_historically_unavailable is True
    assert policy.exclude_missing_metadata is True
    assert policy.exclude_training_overlap is False


def test_operator_can_readmit_unavailable_observations_explicitly():
    policy = TimeIntegrityPolicy(exclude_historically_unavailable=False)
    verdict = classify_observation("2019-06-01", resolve_profile(FINBERT), policy)
    assert verdict.status == HISTORICALLY_UNAVAILABLE_MODEL
    assert verdict.included_in_primary is True


def test_operator_can_also_exclude_overlap_risk():
    policy = TimeIntegrityPolicy(exclude_training_overlap=True, exclude_historically_unavailable=False)
    verdict = classify_observation("2015-06-01", resolve_profile(FINBERT), policy)
    assert verdict.training_overlap_risk is True
    assert verdict.included_in_primary is False


def test_disabled_policy_certifies_nothing():
    policy = TimeIntegrityPolicy(enabled=False)
    verdict = classify_observation("2024-01-02", resolve_profile(FINBERT), policy)
    assert verdict.status == ELIGIBLE
    assert verdict.included_in_primary is False
    assert "switched off" in policy.describe()


def test_policy_from_request_ignores_unknown_keys():
    policy = TimeIntegrityPolicy.from_request({"exclude_missing_metadata": False, "nonsense": 1})
    assert policy.exclude_missing_metadata is False
    assert policy.exclude_historically_unavailable is True


def test_policy_from_request_tolerates_non_mapping():
    assert TimeIntegrityPolicy.from_request(None) == TimeIntegrityPolicy()


# ---------------------------------------------------------- overrides


def test_env_override_replaces_registry_and_is_labelled_an_assumption(monkeypatch):
    monkeypatch.setenv(ENV_OVERRIDES, json.dumps({
        FINBERT: {"model_available_at": "2019-10-30", "training_data_cutoff": "2018-12-31",
                  "source": "internal note", "rationale": "Local weights were mirrored earlier."}
    }))
    profile = resolve_profile(FINBERT)
    assert profile.model_available_at.value == pd.Timestamp("2019-10-30")
    assert profile.model_available_at.status == "assumed"
    assert profile.metadata_source == "override"
    assert any("assumption" in note for note in profile.assumptions)
    assert classify_observation("2019-11-01", profile).status == ELIGIBLE


def test_malformed_override_json_falls_back_to_registry_rather_than_guessing(monkeypatch):
    monkeypatch.setenv(ENV_OVERRIDES, "{not json")
    profile = resolve_profile(FINBERT)
    assert profile.model_available_at.value == pd.Timestamp("2020-12-24")
    assert profile.metadata_source == "registry"


def test_configured_fallback_applies_only_to_unregistered_models(monkeypatch):
    monkeypatch.setenv(ENV_FALLBACK_AVAILABLE_AT, "2023-01-01")
    monkeypatch.setenv(ENV_FALLBACK_TRAINING_CUTOFF, "2022-06-30")
    fallback = resolve_profile("vendor/mystery-model")
    assert fallback.metadata_source == "fallback"
    assert fallback.model_available_at.status == "assumed"
    assert fallback.assumptions
    # The registry still wins for a model AlphaForge has researched.
    assert resolve_profile(FINBERT).metadata_source == "registry"


# --------------------------------------------------------------- audit


def _rows(dates, model=FINBERT):
    return [{"symbol": "AAA", "available_at": date, "document_id": f"doc-{i}", "model": model}
            for i, date in enumerate(dates)]


def test_audit_counts_every_bucket_separately():
    rows = _rows(["2015-06-01", "2019-06-01", "2024-01-02"]) + [
        {"symbol": "BBB", "available_at": "2024-01-02", "document_id": "doc-x", "model": "vendor/unknown"},
    ]
    audit = time_integrity_audit(rows, model=FINBERT)
    counts = audit["counts"]
    assert counts["total_observations"] == 4
    assert counts["eligible_observations"] == 1
    assert counts["historically_unavailable_model_observations"] == 2
    assert counts["training_overlap_risk_observations"] == 1
    assert counts["missing_model_metadata_observations"] == 1
    assert counts["primary_cohort_observations"] == 1
    assert counts["excluded_from_primary"] == 3
    assert audit["coverage"]["eligible_share"] == pytest.approx(0.25)


def test_audit_reports_applied_policy_and_metadata_source():
    audit = time_integrity_audit(_rows(["2024-01-02"]), model=FINBERT)
    assert audit["policy"]["exclude_historically_unavailable"] is True
    assert audit["policy"]["description"]
    assert audit["model_metadata"]["metadata_source"] == "registry"
    assert audit["model_metadata"]["model_available_at"]["status"] == "documented"
    assert audit["has_assumptions"] is False


def test_audit_surfaces_assumptions_when_metadata_is_incomplete():
    audit = time_integrity_audit(_rows(["2024-01-02"], model="vendor/unknown"))
    assert audit["has_assumptions"] is True
    assert any("cannot" in note or "No model-availability" in note for note in audit["assumptions"])


def test_audit_groups_a_mixed_batch_by_its_own_model(monkeypatch):
    monkeypatch.setenv(ENV_OVERRIDES, json.dumps({
        "vendor/late-model": {"model_available_at": "2025-01-01"}
    }))
    rows = [
        {"symbol": "AAA", "available_at": "2024-06-01", "document_id": "a", "model": FINBERT},
        {"symbol": "BBB", "available_at": "2024-06-01", "document_id": "b", "model": "vendor/late-model"},
    ]
    audit = time_integrity_audit(rows)
    by_id = {row["document_id"]: row for row in audit["observations"]}
    assert by_id["a"]["status"] == ELIGIBLE
    assert by_id["b"]["status"] == HISTORICALLY_UNAVAILABLE_MODEL
    assert sorted(audit["models_audited"]) == [FINBERT, "vendor/late-model"]


def test_audit_language_avoids_calling_overlap_leakage():
    audit = time_integrity_audit(_rows(["2015-06-01"]), model=FINBERT)
    assert "risk flag" in audit["note"]
    assert "not proof of memorization" in audit["note"]
    assert "leakage" not in json.dumps(audit).lower()


def test_empty_audit_is_safe():
    audit = time_integrity_audit([], model=FINBERT)
    assert audit["counts"]["total_observations"] == 0
    assert audit["coverage"]["eligible_share"] == 0.0


# ------------------------------------------- event-study integration


def _prices():
    index = pd.bdate_range("2014-01-02", periods=3000)
    close = pd.DataFrame({"AAA": 100.0 + np.arange(len(index)) * 0.05}, index=index)
    market = pd.Series(200.0 + 0.01 * np.arange(len(index)), index=index, name="SPY")
    return close, market


def _loader(symbols, start, end, benchmark):
    return _prices()


def _event_features(dates, model=FINBERT):
    return [{"symbol": "AAA", "available_at": date, "document_id": f"doc-{i}",
             "sentiment": round(-0.8 + i * 0.1, 3), "model": model}
            for i, date in enumerate(dates)]


def _mixed_era_dates():
    old = [str(d.date()) for d in pd.bdate_range("2015-01-05", periods=10, freq="20B")]
    new = [str(d.date()) for d in pd.bdate_range("2021-01-05", periods=12, freq="20B")]
    return old, new


def test_event_study_headline_cohort_excludes_unavailable_model_observations():
    old, new = _mixed_era_dates()
    result = run_filing_event_study(
        {"features": _event_features(old + new), "horizon": 3, "benchmark": "SPY"},
        price_loader=_loader,
    )
    coverage = result["cohort_coverage"]
    assert coverage["primary_cohort_events"] == len(new)
    assert coverage["excluded_events"] == len(old)
    assert result["usable_events"] == len(new)
    assert result["cohort"] == "model-time eligible"
    counts = result["time_integrity"]["counts"]
    assert counts["historically_unavailable_model_observations"] == len(old)


def test_event_study_keeps_all_observations_view_descriptive_and_unblended():
    old, new = _mixed_era_dates()
    result = run_filing_event_study(
        {"features": _event_features(old + new), "horizon": 3, "benchmark": "SPY"},
        price_loader=_loader,
    )
    all_view = result["all_observations_view"]
    assert all_view["usable_events"] == len(old) + len(new)
    # The descriptive view must never carry a verdict of its own.
    assert "verdict" not in all_view
    assert "evidence_established" not in all_view
    assert "not evidence of a historically deployable result" in all_view["note"]
    # The headline result is a different, smaller sample.
    assert result["usable_events"] < all_view["usable_events"]


def test_event_study_refuses_a_headline_when_nothing_is_eligible():
    old, _ = _mixed_era_dates()
    result = run_filing_event_study(
        {"features": _event_features(old), "horizon": 3, "benchmark": "SPY"},
        price_loader=_loader,
    )
    assert result["status"] == "insufficient_data"
    assert "no primary evidence cohort" in result["message"]
    assert result["usable_events"] == 0


def test_event_study_override_readmits_the_old_cohort_when_asked():
    old, new = _mixed_era_dates()
    result = run_filing_event_study(
        {"features": _event_features(old + new), "horizon": 3, "benchmark": "SPY",
         "time_integrity": {"exclude_historically_unavailable": False}},
        price_loader=_loader,
    )
    assert result["cohort_coverage"]["excluded_events"] == 0
    assert result["time_integrity"]["policy"]["exclude_historically_unavailable"] is False


def test_legacy_event_payload_without_model_is_flagged_not_silently_valid():
    """Backward compatibility: old saved results still post, but never certify."""
    _, new = _mixed_era_dates()
    features = [{key: value for key, value in row.items() if key != "model"}
                for row in _event_features(new)]
    result = run_filing_event_study(
        {"features": features, "horizon": 3, "benchmark": "SPY"},
        price_loader=_loader,
    )
    counts = result["time_integrity"]["counts"]
    assert counts["missing_model_metadata_observations"] == len(new)
    assert result["status"] == "insufficient_data"
    assert result["time_integrity"]["has_assumptions"] is True


def test_legacy_event_payload_can_be_admitted_with_an_explicit_override():
    _, new = _mixed_era_dates()
    features = [{key: value for key, value in row.items() if key != "model"}
                for row in _event_features(new)]
    result = run_filing_event_study(
        {"features": features, "horizon": 3, "benchmark": "SPY",
         "time_integrity": {"exclude_missing_metadata": False}},
        price_loader=_loader,
    )
    assert result["cohort_coverage"]["primary_cohort_events"] == len(new)
    assert result["time_integrity"]["counts"]["missing_model_metadata_observations"] == len(new)


# ----------------------------------------- document batch, exports, UI


def _batch(filing_date="2024-05-02"):
    """Run a fully stubbed filing batch, dated by the SEC fixture."""
    from research.sec_documents import SecDocumentProvider
    from research.text_features import FinBertExtractor

    base = "https://www.sec.gov/Archives/edgar/data/1234/000123456724000001/report.htm"
    release_html = (
        "<h1>Quarterly results</h1>"
        "<p>The company reported quarterly revenue of $100 million, an increase of 10 percent "
        "compared with the prior year.</p>"
    )
    primary = ('<p>Item 2.02 Results of Operations and Financial Condition.</p>'
               '<table><tr><td>99.1</td><td><a href="release.htm">Earnings release</a></td></tr></table>')

    def fetch_json(url, _headers):
        if "company_tickers" in url:
            return {"0": {"ticker": "ABC", "cik_str": 1234}}
        return {"filings": {"recent": {
            "accessionNumber": ["0001234567-24-000001"], "filingDate": [filing_date],
            "form": ["8-K"], "primaryDocument": ["report.htm"],
        }}}

    provider = SecDocumentProvider(
        user_agent="AlphaForge test@alphaforge.local", fetch_json=fetch_json,
        fetch_text=lambda url, _: primary if url == base else release_html, request_interval=0)

    class Client:
        def text_classification(self, text, model):
            return [{"label": "positive", "score": 0.7}, {"label": "negative", "score": 0.1},
                    {"label": "neutral", "score": 0.2}]

    from api.service import run_real_document_batch
    return run_real_document_batch(
        {"symbols": ["ABC"], "as_of": "2024-06-01"},
        document_provider=provider, text_extractor=FinBertExtractor(client=Client()))


def test_document_batch_attaches_a_time_integrity_audit():
    out = _batch()
    audit = out["time_integrity"]
    assert audit["model"] == FINBERT
    assert audit["counts"]["eligible_observations"] == 1
    assert audit["model_metadata"]["model_available_at"]["value"] == "2020-12-24"
    assert audit["model_metadata"]["training_data_cutoff"]["status"] == "inferred"


def test_document_batch_marks_each_feature_with_its_standing():
    record = _batch()["features"][0]
    assert record["time_integrity"]["status"] == ELIGIBLE
    assert record["time_integrity"]["included_in_primary"] is True


def test_document_batch_flags_a_pre_release_filing():
    out = _batch(filing_date="2015-05-04")
    record = out["features"][0]
    assert record["time_integrity"]["status"] == HISTORICALLY_UNAVAILABLE_MODEL
    assert record["time_integrity"]["training_overlap_risk"] is True
    assert out["time_integrity"]["counts"]["primary_cohort_observations"] == 0


def test_audit_export_is_json_serializable_and_retains_sources():
    payload = json.dumps(_batch())
    assert "2020-12-24" in payload
    assert "huggingface.co/ProsusAI/finbert" in payload
    assert "documented" in payload and "inferred" in payload


def test_api_request_model_accepts_model_and_policy_fields():
    import api.main as main
    feature = main.FilingEventFeature(symbol="AAA", available_at="2024-01-02",
                                      document_id="d1", sentiment=0.1, model=FINBERT)
    assert feature.model == FINBERT
    request = main.FilingEventStudyRequest(
        features=[feature], time_integrity={"exclude_missing_metadata": False})
    assert request.time_integrity.exclude_missing_metadata is False
    # Legacy payloads without a model still validate.
    assert main.FilingEventFeature(symbol="AAA", available_at="2024-01-02",
                                   document_id="d1", sentiment=0.1).model is None


def test_filing_research_page_renders_the_time_integrity_panel(monkeypatch):
    import pathlib

    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("ALPHAFORGE_API", "http://127.0.0.1:1")
    home = str(pathlib.Path(__file__).resolve().parent.parent / "app" / "Home.py")
    app = AppTest.from_file(home, default_timeout=180)
    app.run()
    app.session_state["real_document_batch"] = _batch(filing_date="2015-05-04")
    page = app.switch_page("pages/2_Real_Document_Batch.py").run()
    assert not page.exception
    labels = [metric.label for metric in page.metric]
    assert "Unavailable model" in labels and "Overlap risk" in labels
    assert "Missing metadata" in labels and "Eligible" in labels
    text = " ".join(str(item.value) for item in list(page.caption) + list(page.warning))
    assert "not model-time eligible" in text
    assert "risk flag" in text and "not proof of memorization" in text


def test_legacy_saved_batch_without_audit_is_flagged_in_the_ui(monkeypatch):
    """A result saved before this gate existed must not read as a silent pass."""
    import pathlib

    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("ALPHAFORGE_API", "http://127.0.0.1:1")
    home = str(pathlib.Path(__file__).resolve().parent.parent / "app" / "Home.py")
    app = AppTest.from_file(home, default_timeout=180)
    app.run()
    legacy = _batch()
    legacy.pop("time_integrity")
    for record in legacy["features"]:
        record.pop("time_integrity", None)
    app.session_state["real_document_batch"] = legacy
    page = app.switch_page("pages/2_Real_Document_Batch.py").run()
    assert not page.exception
    warnings = " ".join(str(item.value) for item in page.warning)
    assert "before AlphaForge recorded model-time integrity" in warnings
    assert "not evidence of a historically deployable result" in warnings


# ------------------------------- public-data pilot (provenance, no verdict)


def _pilot(available_at="2024-05-01T16:30:00"):
    """Run the stubbed pilot with a dated FinBERT score."""
    import tests.test_public_data_pilot as pilot
    from api import service

    request = pilot._request(with_text=True)
    request["text_document"]["available_at"] = available_at

    class Extractor:
        def extract(self, document):
            return pilot.FakeFeature(available_at=pd.Timestamp(available_at))

    return service.run_public_data_pilot(
        request, sec_provider=pilot.FakeSecProvider(),
        macro_provider=pilot.FakeMacroProvider(), text_extractor=Extractor())


def test_pilot_score_carries_model_provenance():
    integrity = _pilot()["text_feature"]["model_time_integrity"]
    metadata = integrity["model_metadata"]
    assert metadata["model_available_at"]["value"] == "2020-12-24"
    assert metadata["model_available_at"]["status"] == "documented"
    assert metadata["training_data_cutoff"]["value"] == "2018-12-31"
    assert metadata["training_data_cutoff"]["status"] == "inferred"
    assert integrity["status"] == ELIGIBLE


def test_pilot_flags_a_pre_release_document_without_certifying_anything():
    integrity = _pilot(available_at="2015-05-01T16:30:00")["text_feature"]["model_time_integrity"]
    assert integrity["status"] == HISTORICALLY_UNAVAILABLE_MODEL
    assert integrity["training_overlap_risk"] is True
    # The pilot has no cohort, so it must not imply admission or exclusion.
    assert "included_in_primary" not in integrity
    assert "makes no deployability claim" in integrity["note"]


def test_pilot_language_stays_non_certifying():
    integrity = _pilot()["text_feature"]["model_time_integrity"]
    assert "provenance only" in integrity["note"]
    assert "no observation is admitted or excluded" in integrity["note"]


def test_pilot_page_renders_model_provenance(monkeypatch):
    import pathlib

    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("ALPHAFORGE_API", "http://127.0.0.1:1")
    home = str(pathlib.Path(__file__).resolve().parent.parent / "app" / "Home.py")
    app = AppTest.from_file(home, default_timeout=180)
    app.run()
    app.session_state["public_data_pilot"] = _pilot(available_at="2015-05-01T16:30:00")
    page = app.switch_page("pages/1_Public_Data_Pilot.py").run()
    assert not page.exception
    text = " ".join(str(item.value) for item in list(page.caption) + list(page.info))
    assert "Model provenance" in text
    assert "2020-12-24" in text and "documented" in text


# ------------------------------------------------- pre-close code review


def test_partial_override_keeps_the_registered_value_it_does_not_replace():
    """A partial override must not silently disable overlap-risk flagging."""
    import os

    os.environ[ENV_OVERRIDES] = json.dumps({
        FINBERT: {"model_available_at": "2019-10-30", "rationale": "Local weights mirrored."}})
    try:
        profile = resolve_profile(FINBERT)
        assert profile.model_available_at.value == pd.Timestamp("2019-10-30")
        assert profile.model_available_at.status == "assumed"
        # The untouched field keeps the registry value AND its own status.
        assert profile.training_data_cutoff.value == pd.Timestamp("2018-12-31")
        assert profile.training_data_cutoff.status == "inferred"
        assert classify_observation("2015-06-01", profile).training_overlap_risk is True
    finally:
        os.environ.pop(ENV_OVERRIDES, None)


def test_cited_dates_match_their_stated_sources():
    """Every rendered date must be traceable to the citation shown beside it."""
    profile = resolve_profile(FINBERT)
    available = profile.model_available_at
    cutoff = profile.training_data_cutoff
    # Availability is the Hugging Face Hub upload; the citation is that repo.
    assert available.source == "https://huggingface.co/ProsusAI/finbert/commits/main"
    assert "2020-12-24" in available.rationale
    # The rejected earlier candidates are named so a reader can audit the choice.
    assert "2019-10-30" in available.rationale and "1908.10063" in available.rationale
    # The cutoff cites both the FinBERT paper and the BERT paper it builds on.
    assert "1908.10063" in cutoff.source and "1810.04805" in cutoff.source
    assert "2008" in cutoff.rationale and "2010" in cutoff.rationale
    assert "Malo et al. 2014" in cutoff.rationale
    assert "not published" in cutoff.rationale  # why it is inferred, not documented


def test_inferred_is_never_reported_as_an_assumption():
    profile = resolve_profile(FINBERT)
    assert profile.training_data_cutoff.status == "inferred"
    assert profile.training_data_cutoff.is_assumption is False
    assert profile.assumptions == []
    payload = profile.to_dict()
    assert payload["training_data_cutoff"]["is_assumption"] is False
    assert payload["has_assumptions"] is False


def test_assumed_is_never_reported_as_documented_or_inferred():
    import os

    os.environ[ENV_FALLBACK_AVAILABLE_AT] = "2023-01-01"
    try:
        profile = resolve_profile("vendor/mystery")
        assert profile.model_available_at.status == "assumed"
        assert profile.model_available_at.is_assumption is True
        assert profile.to_dict()["has_assumptions"] is True
    finally:
        os.environ.pop(ENV_FALLBACK_AVAILABLE_AT, None)


def test_primary_cohort_events_never_include_an_excluded_document():
    """Strongest anti-merge check: inspect the rows the headline actually used."""
    old, new = _mixed_era_dates()
    result = run_filing_event_study(
        {"features": _event_features(old + new), "horizon": 3, "benchmark": "SPY"},
        price_loader=_loader,
    )
    excluded_ids = {row["document_id"] for row in result["time_integrity"]["observations"]
                    if not row["included_in_primary"]}
    headline_ids = {row["document_id"] for row in result.get("events", [])}
    assert headline_ids and not (headline_ids & excluded_ids)
    # The descriptive view does cover them, and stays separate.
    all_ids = {row["document_id"] for row in result["all_observations_view"].get("events", [])}
    assert excluded_ids <= all_ids


def test_descriptive_view_cannot_be_mistaken_for_the_conclusion():
    old, new = _mixed_era_dates()
    result = run_filing_event_study(
        {"features": _event_features(old + new), "horizon": 3, "benchmark": "SPY"},
        price_loader=_loader,
    )
    all_view = result["all_observations_view"]
    assert "verdict" not in all_view and "evidence_established" not in all_view
    assert all_view["label"] == "All observations (descriptive only)"
    # It is namespaced, so a consumer reading the top level cannot pick it up.
    assert result["usable_events"] != all_view["usable_events"]
    assert result["cohort"] == "model-time eligible"


def test_ui_labels_avoid_jargon_and_state_the_consequence(monkeypatch):
    """A non-technical reader must get plain words and a stated consequence."""
    import pathlib

    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("ALPHAFORGE_API", "http://127.0.0.1:1")
    home = str(pathlib.Path(__file__).resolve().parent.parent / "app" / "Home.py")
    app = AppTest.from_file(home, default_timeout=180)
    app.run()
    app.session_state["real_document_batch"] = _batch(filing_date="2015-05-04")
    page = app.switch_page("pages/2_Real_Document_Batch.py").run()
    assert not page.exception

    labels = [metric.label for metric in page.metric]
    # Plain-English metric names, not internal status slugs.
    assert "Unavailable model" in labels and "Missing metadata" in labels
    assert not any("_" in label for label in labels)

    body = " ".join(str(item.value) for item in
                    list(page.caption) + list(page.warning) + list(page.info))
    # The consequence is stated, not just the flag.
    assert "cannot support a historically deployable study" in body
    assert "risk flag and not proof of memorization" in body
    # Internal identifiers must not leak into the reader-facing copy.
    for slug in ("historically_unavailable_model", "missing_model_metadata", "included_in_primary"):
        assert slug not in body
