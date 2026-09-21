"""Model-time integrity: was the scoring model itself available point-in-time?

AlphaForge is already strict about *document* availability: a ``TextDocument``
cannot be built without an ``available_at`` timestamp.  It was silent about
*model* availability.  Scoring a 2015 filing with weights first published in
2020 produces a number no researcher could have computed in 2015, so the
resulting study cannot be described as historically deployable.

This module separates two distinct concerns and refuses to conflate them:

``model_available_at``
    The earliest defensible date the model could have been used in a live
    research workflow.  Observations before it are *historically
    unavailable-model observations*.  They cannot support a historically
    deployable study.  This is a statement about workflow feasibility, not
    about the model's behaviour.

``training_data_cutoff``
    The latest date represented in the model's pretraining or fine-tuning
    data, recorded only where documented or reasonably supportable.
    Observations before it carry *training-data overlap risk*.  That is a risk
    flag, not proof of memorization: overlap means the outcome period may be
    represented in the training corpus, not that the model recalled it.

Every stored date carries its source, a ``documented``/``inferred``/``assumed``
status, and a rationale, so a reader can audit the claim rather than trust it.

Motivation: Benhenda, "Look-Ahead-Bench: a Standardized Benchmark of Look-ahead
Bias in Point-in-Time LLMs for Finance" (arXiv:2601.13770), which measures
look-ahead bias in financial language models through performance decay across
temporally distinct regimes.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from typing import Any, Iterable, Literal

import pandas as pd

FactStatus = Literal["documented", "inferred", "assumed"]
MetadataSource = Literal["registry", "override", "fallback", "unknown"]

ELIGIBLE = "eligible"
HISTORICALLY_UNAVAILABLE_MODEL = "historically_unavailable_model"
MISSING_MODEL_METADATA = "missing_model_metadata"

#: Human-readable wording used in the UI and exports.  Deliberately avoids the
#: words "leakage" and "memorization" for the overlap risk flag.
STATUS_LABELS: dict[str, str] = {
    ELIGIBLE: "Eligible",
    HISTORICALLY_UNAVAILABLE_MODEL: "Historically unavailable model",
    MISSING_MODEL_METADATA: "Missing model metadata",
}

OVERLAP_RISK_LABEL = "Training-data overlap risk"

ENV_OVERRIDES = "ALPHAFORGE_MODEL_TIME_OVERRIDES"
ENV_FALLBACK_AVAILABLE_AT = "ALPHAFORGE_MODEL_TIME_FALLBACK_AVAILABLE_AT"
ENV_FALLBACK_TRAINING_CUTOFF = "ALPHAFORGE_MODEL_TIME_FALLBACK_TRAINING_CUTOFF"


def _as_date(value: Any) -> pd.Timestamp | None:
    """Normalize a date-like value to a date-only timestamp, or None."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(stamp):
        return None
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp.normalize()


@dataclass(frozen=True)
class ModelFact:
    """One dated claim about a model, with the evidence behind it."""

    value: pd.Timestamp | None
    source: str
    status: FactStatus
    rationale: str

    def __post_init__(self) -> None:
        if self.status not in ("documented", "inferred", "assumed"):
            raise ValueError("model fact status must be documented, inferred, or assumed")
        object.__setattr__(self, "value", _as_date(self.value))

    @property
    def is_assumption(self) -> bool:
        """True when the value is not grounded in a primary source."""
        return self.status == "assumed"

    def to_dict(self) -> dict:
        return {
            "value": None if self.value is None else str(self.value.date()),
            "source": self.source,
            "status": self.status,
            "rationale": self.rationale,
            "is_assumption": self.is_assumption,
        }


@dataclass(frozen=True)
class ModelTimeProfile:
    """Everything known about when a scoring model became usable."""

    model: str
    model_available_at: ModelFact
    training_data_cutoff: ModelFact
    metadata_source: MetadataSource = "registry"

    @property
    def has_availability(self) -> bool:
        return self.model_available_at.value is not None

    @property
    def assumptions(self) -> list[str]:
        """Plain-language list of every value a reader should not treat as fact."""
        notes: list[str] = []
        if self.model_available_at.is_assumption and self.model_available_at.value is not None:
            notes.append(
                f"Model availability date for {self.model} "
                f"({self.model_available_at.value.date()}) is an operator assumption, not a documented fact."
            )
        if self.training_data_cutoff.is_assumption and self.training_data_cutoff.value is not None:
            notes.append(
                f"Training-data cutoff for {self.model} "
                f"({self.training_data_cutoff.value.date()}) is an operator assumption, not a documented fact."
            )
        if not self.has_availability:
            notes.append(
                f"No model-availability date is registered for {self.model}; "
                "no observation scored by it can be certified as historically deployable."
            )
        return notes

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "model_available_at": self.model_available_at.to_dict(),
            "training_data_cutoff": self.training_data_cutoff.to_dict(),
            "metadata_source": self.metadata_source,
            "has_assumptions": bool(self.assumptions),
            "assumptions": self.assumptions,
        }


# --------------------------------------------------------------------------
# Registry of models AlphaForge actually invokes.
# --------------------------------------------------------------------------

FINBERT_PROFILE = ModelTimeProfile(
    model="ProsusAI/finbert",
    model_available_at=ModelFact(
        value="2020-12-24",
        source="https://huggingface.co/ProsusAI/finbert/commits/main",
        status="documented",
        rationale=(
            "The ProsusAI/finbert weights were first published to the Hugging Face Hub on 2020-12-24 "
            "(the repository commit history shows 'initial commit' and 'First version of finbert', both "
            "dated 2020-12-24). AlphaForge scores documents through the Hugging Face inference API, so "
            "this is the earliest date this pipeline could have run. The reference implementation "
            "appeared on GitHub on 2019-10-30 and the paper (arXiv:1908.10063) on 2019-08-27; neither "
            "made these Hub-hosted weights retrievable, so the Hub date is both the accurate date for "
            "this workflow and the more conservative of the candidates."
        ),
    ),
    training_data_cutoff=ModelFact(
        value="2018-12-31",
        source=(
            "https://arxiv.org/abs/1908.10063 (FinBERT, sections 3.1 and 4.2); "
            "https://arxiv.org/abs/1810.04805 (BERT)"
        ),
        status="inferred",
        rationale=(
            "FinBERT further-pretrains BERT (Devlin et al. 2018; BookCorpus and English Wikipedia) on "
            "Reuters TRC2-financial, which the FinBERT paper describes as articles 'published by Reuters "
            "between 2008 and 2010', then fine-tunes on the Financial PhraseBank (Malo et al. 2014). The "
            "newest component is BERT's English Wikipedia snapshot, which precedes BERT's October 2018 "
            "publication. No component is documented to contain text after 2018, so 2018-12-31 is a "
            "conservative upper bound. The exact Wikipedia dump date is not published by its authors, so "
            "this value is inferred from the release timeline rather than documented."
        ),
    ),
)

MODEL_REGISTRY: dict[str, ModelTimeProfile] = {
    FINBERT_PROFILE.model.lower(): FINBERT_PROFILE,
}


def _unknown_profile(model: str) -> ModelTimeProfile:
    """The default fallback: no invented dates, so nothing is certified valid."""
    return ModelTimeProfile(
        model=model,
        model_available_at=ModelFact(
            value=None,
            source="",
            status="assumed",
            rationale=(
                f"{model} is not in AlphaForge's model-time registry and no operator fallback was "
                "configured. No availability date is invented, so every observation is reported as "
                "missing model metadata rather than silently treated as eligible."
            ),
        ),
        training_data_cutoff=ModelFact(
            value=None,
            source="",
            status="assumed",
            rationale=f"No training-data cutoff is registered or configured for {model}.",
        ),
        metadata_source="unknown",
    )


def _configured_fallback(model: str) -> ModelTimeProfile | None:
    """Build a profile from operator-supplied fallback dates, if any are set."""
    available = _as_date(os.environ.get(ENV_FALLBACK_AVAILABLE_AT, "").strip() or None)
    cutoff = _as_date(os.environ.get(ENV_FALLBACK_TRAINING_CUTOFF, "").strip() or None)
    if available is None and cutoff is None:
        return None
    return ModelTimeProfile(
        model=model,
        model_available_at=ModelFact(
            value=available,
            source=f"operator configuration ({ENV_FALLBACK_AVAILABLE_AT})",
            status="assumed",
            rationale=(
                f"{model} is not in AlphaForge's model-time registry. This availability date is an "
                f"operator assumption supplied through {ENV_FALLBACK_AVAILABLE_AT}. It is not sourced "
                "from the model's documentation and is labelled as an assumption wherever it is shown."
            ),
        ),
        training_data_cutoff=ModelFact(
            value=cutoff,
            source=f"operator configuration ({ENV_FALLBACK_TRAINING_CUTOFF})",
            status="assumed",
            rationale=(
                f"Training-data cutoff for {model} supplied through {ENV_FALLBACK_TRAINING_CUTOFF}. "
                "It is an operator assumption, not a documented model fact."
            ),
        ),
        metadata_source="fallback",
    )


def _env_overrides() -> dict[str, dict]:
    """Parse the override environment variable; malformed JSON is ignored, not guessed."""
    raw = os.environ.get(ENV_OVERRIDES, "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(key).lower(): value for key, value in parsed.items() if isinstance(value, dict)}


def _override_profile(model: str, payload: dict, base: ModelTimeProfile | None) -> ModelTimeProfile:
    """Apply an operator override field by field, recorded as an assumption.

    Only the fields the payload actually supplies are replaced. A partial
    override must not silently discard a registered value: dropping the
    training-data cutoff would turn off overlap-risk flagging without saying so.
    """
    source = str(payload.get("source") or f"operator override ({ENV_OVERRIDES})")
    rationale = str(
        payload.get("rationale")
        or "Operator-supplied override of the registered model-time metadata."
    )
    fallback = base or _unknown_profile(model)

    def pick(field: str, registered: ModelFact) -> ModelFact:
        if field not in payload:
            return registered
        return ModelFact(value=payload.get(field), source=source, status="assumed",
                         rationale=rationale)

    return ModelTimeProfile(
        model=model,
        model_available_at=pick("model_available_at", fallback.model_available_at),
        training_data_cutoff=pick("training_data_cutoff", fallback.training_data_cutoff),
        metadata_source="override",
    )


def resolve_profile(model: str | None) -> ModelTimeProfile:
    """Find the model-time profile for a scoring model.

    Resolution order is override, registry, operator fallback, then the unknown
    profile.  An override outranks the registry so an operator can correct a
    value, field by field; each field it actually supplies is recorded as an
    assumption, and the fields it omits keep their registered value and status.
    """
    name = (model or "").strip()
    if not name:
        return _unknown_profile("unnamed model")
    key = name.lower()
    registered = (
        replace(MODEL_REGISTRY[key], model=name) if key in MODEL_REGISTRY
        else _configured_fallback(name)
    )
    overrides = _env_overrides()
    if key in overrides:
        return _override_profile(name, overrides[key], registered)
    return registered or _unknown_profile(name)


# --------------------------------------------------------------------------
# Policy and classification
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TimeIntegrityPolicy:
    """Which observation classes stay in the primary evidence cohort.

    The defaults are the conservative reading: an observation whose model did
    not yet exist, or whose model provenance is unknown, cannot support a
    historically deployable claim.  Training-data overlap is reported as a risk
    flag and does not exclude by default, because overlap is not proof that the
    model reproduced a remembered outcome.
    """

    exclude_historically_unavailable: bool = True
    exclude_missing_metadata: bool = True
    exclude_training_overlap: bool = False
    enabled: bool = True

    @classmethod
    def from_request(cls, payload: Any) -> "TimeIntegrityPolicy":
        """Build a policy from an API payload fragment; unknown keys are ignored."""
        if not isinstance(payload, dict):
            return cls()
        fields = {
            "exclude_historically_unavailable",
            "exclude_missing_metadata",
            "exclude_training_overlap",
            "enabled",
        }
        kwargs = {key: bool(payload[key]) for key in fields if key in payload}
        return cls(**kwargs)

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "exclude_historically_unavailable": self.exclude_historically_unavailable,
            "exclude_missing_metadata": self.exclude_missing_metadata,
            "exclude_training_overlap": self.exclude_training_overlap,
            "description": self.describe(),
        }

    def describe(self) -> str:
        if not self.enabled:
            return (
                "Model-time integrity screening is switched off. Every observation is reported in a "
                "single descriptive cohort and none of it is certified as historically deployable."
            )
        excluded = []
        if self.exclude_historically_unavailable:
            excluded.append("historically unavailable-model observations")
        if self.exclude_missing_metadata:
            excluded.append("observations with missing model metadata")
        if self.exclude_training_overlap:
            excluded.append("observations carrying training-data overlap risk")
        if not excluded:
            return "No observation class is excluded from the primary evidence cohort."
        return "Excluded from the primary evidence cohort: " + "; ".join(excluded) + "."


@dataclass(frozen=True)
class ObservationVerdict:
    """The model-time standing of one scored document."""

    status: str
    training_overlap_risk: bool
    reason: str
    included_in_primary: bool

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "status_label": STATUS_LABELS.get(self.status, self.status),
            "training_overlap_risk": self.training_overlap_risk,
            "reason": self.reason,
            "included_in_primary": self.included_in_primary,
        }


def classify_observation(
    available_at: Any,
    profile: ModelTimeProfile,
    policy: TimeIntegrityPolicy | None = None,
) -> ObservationVerdict:
    """Classify one document against a model's availability and training window."""
    policy = policy or TimeIntegrityPolicy()
    stamp = _as_date(available_at)
    cutoff = profile.training_data_cutoff.value
    overlap = bool(stamp is not None and cutoff is not None and stamp <= cutoff)

    if stamp is None:
        status = MISSING_MODEL_METADATA
        reason = (
            "The document has no usable availability date, so its model-time standing cannot be "
            "established."
        )
    elif not profile.has_availability:
        status = MISSING_MODEL_METADATA
        reason = (
            f"No model-availability date is registered for {profile.model}, so this observation cannot "
            "be certified as historically deployable."
        )
    elif stamp < profile.model_available_at.value:
        status = HISTORICALLY_UNAVAILABLE_MODEL
        reason = (
            f"The document was available on {stamp.date()}, before {profile.model} could be used in a "
            f"live workflow ({profile.model_available_at.value.date()}). It cannot support a "
            "historically deployable study."
        )
    else:
        status = ELIGIBLE
        reason = (
            f"The document was available on {stamp.date()}, on or after {profile.model} became usable "
            f"({profile.model_available_at.value.date()})."
        )

    if overlap:
        reason += (
            f" It also predates the model's training-data cutoff ({cutoff.date()}), so it carries "
            "training-data overlap risk. That is a risk flag, not evidence of memorization."
        )

    if not policy.enabled:
        included = False
    else:
        included = status == ELIGIBLE
        if status == HISTORICALLY_UNAVAILABLE_MODEL and not policy.exclude_historically_unavailable:
            included = True
        if status == MISSING_MODEL_METADATA and not policy.exclude_missing_metadata:
            included = True
        if overlap and policy.exclude_training_overlap:
            included = False
    return ObservationVerdict(status, overlap, reason, included)


def time_integrity_audit(
    observations: Iterable[dict],
    *,
    model: str | None = None,
    policy: TimeIntegrityPolicy | None = None,
) -> dict:
    """Audit a batch of scored documents for model-time integrity.

    Each observation needs ``available_at``; ``document_id``, ``symbol`` and
    ``model`` are used when present.  Observations are grouped per model, so a
    mixed batch is never audited against a single model's dates.
    """
    policy = policy or TimeIntegrityPolicy()
    rows = list(observations)
    default_model = (model or "").strip()

    records: list[dict] = []
    profiles: dict[str, ModelTimeProfile] = {}
    for item in rows:
        row_model = str(item.get("model") or default_model or "").strip()
        key = row_model.lower()
        if key not in profiles:
            profiles[key] = resolve_profile(row_model)
        profile = profiles[key]
        verdict = classify_observation(item.get("available_at"), profile, policy)
        stamp = _as_date(item.get("available_at"))
        records.append({
            "document_id": str(item.get("document_id", "")),
            "symbol": str(item.get("symbol", "")),
            "available_at": None if stamp is None else str(stamp.date()),
            "model": row_model or None,
            **verdict.to_dict(),
        })

    total = len(records)
    eligible = sum(r["status"] == ELIGIBLE for r in records)
    unavailable = sum(r["status"] == HISTORICALLY_UNAVAILABLE_MODEL for r in records)
    missing = sum(r["status"] == MISSING_MODEL_METADATA for r in records)
    overlap = sum(bool(r["training_overlap_risk"]) for r in records)
    included = sum(bool(r["included_in_primary"]) for r in records)

    primary = None
    if default_model and default_model.lower() in profiles:
        primary = profiles[default_model.lower()]
    elif profiles:
        primary = next(iter(profiles.values()))
    model_metadata = (
        primary.to_dict() if primary
        else _unknown_profile(default_model or "unnamed model").to_dict()
    )

    assumptions: list[str] = []
    for profile in profiles.values():
        assumptions.extend(profile.assumptions)
    assumptions = list(dict.fromkeys(assumptions))

    return {
        "model": (primary.model if primary else default_model) or None,
        "models_audited": sorted({r["model"] for r in records if r["model"]}),
        "policy": policy.to_dict(),
        "model_metadata": model_metadata,
        "all_model_metadata": [profile.to_dict() for profile in profiles.values()],
        "has_assumptions": bool(assumptions),
        "assumptions": assumptions,
        "counts": {
            "total_observations": total,
            "eligible_observations": eligible,
            "historically_unavailable_model_observations": unavailable,
            "training_overlap_risk_observations": overlap,
            "missing_model_metadata_observations": missing,
            "primary_cohort_observations": included,
            "excluded_from_primary": total - included,
        },
        "coverage": {
            "eligible_share": (eligible / total) if total else 0.0,
            "primary_cohort_share": (included / total) if total else 0.0,
        },
        "observations": records,
        "note": (
            "Observations before the model's availability date are historically unavailable-model "
            "observations: no researcher could have produced these scores at that time, so they cannot "
            "support a historically deployable study. Observations before the training-data cutoff "
            "carry training-data overlap risk, which is a risk flag and not proof of memorization."
        ),
    }
