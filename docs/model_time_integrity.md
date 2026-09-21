# Model-Time Integrity

## The gap this closes

AlphaForge was already strict about *document* availability. `TextDocument`
refuses to construct without an `available_at` timestamp, filing studies enter
on the next trading session, and the audit export keeps the exact model input.

None of that says anything about the **model**. Scoring a 2015 filing with
FinBERT produces a number no researcher could have computed in 2015, because
the weights did not exist yet. A study built on such scores may be interesting,
but it cannot be described as historically deployable. Before this guard,
AlphaForge would report it with the same headline verdict as a study built
entirely on scores that were reproducible at the time.

The motivating reference is Benhenda, *Look-Ahead-Bench: a Standardized
Benchmark of Look-ahead Bias in Point-in-Time LLMs for Finance*
([arXiv:2601.13770](https://arxiv.org/abs/2601.13770), submitted 20 Jan 2026),
which measures look-ahead bias in financial language models through performance
decay across temporally distinct market regimes rather than through question
answering. Our guard is narrower than that benchmark: it is a workflow-feasibility
screen, not a measurement of a model's memorization.

## Two concepts, deliberately not merged

| Concept | Question it answers | What a violation means |
| --- | --- | --- |
| `model_available_at` | Could this model have been used in a live research workflow on this date? | **Historically unavailable-model observation.** Cannot support a historically deployable study. Excluded from the primary evidence cohort by default. |
| `training_data_cutoff` | Is this document's period represented in the model's training corpus? | **Training-data overlap risk.** A risk flag, not proof of memorization. Reported, not excluded by default. |

These are different claims with different strengths of evidence, so they are
stored and reported separately. The first is a hard statement about what a
researcher could have done. The second is a caution: overlap means the outcome
period *may* be represented in the training data, not that the model recalled
any specific outcome.

The code avoids the words "leakage" and "memorization" when describing overlap,
because neither is established by a date comparison.

## Every date carries its evidence

Each value is a `ModelFact` with four fields:

- `value` — the date, or `None` when nothing defensible is known
- `source` — a URL or citation
- `status` — `documented`, `inferred`, or `assumed`
- `rationale` — why this value, in prose

## Registered values for `ProsusAI/finbert`

**`model_available_at` = 2020-12-24 — `documented`**

Source: <https://huggingface.co/ProsusAI/finbert/commits/main>

The repository's commit history shows `initial commit` and
`First version of finbert`, both dated 2020-12-24. AlphaForge scores documents
through the Hugging Face inference API, so this is the earliest date this
pipeline could have run.

Two earlier dates exist and were rejected, for the same reason in both cases —
neither made these Hub-hosted weights retrievable:

- 2019-08-27, the FinBERT paper ([arXiv:1908.10063](https://arxiv.org/abs/1908.10063))
- 2019-10-30, creation of the `ProsusAI/finBERT` GitHub repository (which
  publishes no tagged releases)

The Hub date is therefore both the accurate date for this workflow and the more
conservative of the candidates. An operator running local weights obtained
another way should override it (see below) — that override is recorded as an
assumption.

**`training_data_cutoff` = 2018-12-31 — `inferred`**

Sources: [arXiv:1908.10063](https://arxiv.org/abs/1908.10063) sections 3.1 and
4.2; [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)

FinBERT is built from three layers, and the cutoff is bounded by the newest one:

| Layer | Dates | Evidence |
| --- | --- | --- |
| Fine-tuning: Financial PhraseBank | Malo et al. 2014 | FinBERT paper §4.2.2 |
| Further pretraining: Reuters TRC2-financial | articles "published by Reuters between 2008 and 2010" | FinBERT paper §4.2.1, quoting the corpus description |
| Base: BERT (BookCorpus + English Wikipedia) | snapshot predating BERT's October 2018 publication | FinBERT paper §3.1; BERT paper |

The newest component is BERT's English Wikipedia snapshot. No component is
documented to contain text after 2018, so 2018-12-31 is a conservative upper
bound. **The exact Wikipedia dump date is not published by BERT's authors**,
which is why this value is `inferred` rather than `documented`. It is an upper
bound chosen to over-flag rather than under-flag.

Note the ordering: the cutoff (2018-12-31) precedes availability (2020-12-24).
A 2019 filing is therefore *historically unavailable-model* but carries **no**
overlap risk. The two checks are genuinely independent.

## Observation classes

| Class | Meaning | In primary cohort by default |
| --- | --- | --- |
| `eligible` | Filed on or after the model became usable | Yes |
| `historically_unavailable_model` | Filed before the model existed | No |
| `missing_model_metadata` | No registered model-time metadata, or no usable document date | No |
| Training-overlap risk | Orthogonal flag, can attach to any class | Does not exclude |

## Configuration

Defaults are the conservative reading. All of it is overridable, and every
override is recorded as an assumption in the audit and labelled in the UI.

**Per request** (API or in-process), under `time_integrity`:

```json
{
  "enabled": true,
  "exclude_historically_unavailable": true,
  "exclude_missing_metadata": true,
  "exclude_training_overlap": false
}
```

**Environment variables:**

| Variable | Effect |
| --- | --- |
| `ALPHAFORGE_MODEL_TIME_OVERRIDES` | JSON map of model name to `{model_available_at, training_data_cutoff, source, rationale}`. Outranks the registry, field by field. Malformed JSON is ignored, never guessed. |
| `ALPHAFORGE_MODEL_TIME_FALLBACK_AVAILABLE_AT` | Availability date for models not in the registry. Recorded as `assumed`. |
| `ALPHAFORGE_MODEL_TIME_FALLBACK_TRAINING_CUTOFF` | Training cutoff for models not in the registry. Recorded as `assumed`. |

**Overrides apply per field.** Supplying only `model_available_at` replaces that
value and leaves the registered `training_data_cutoff` intact, with its own
`inferred` status. This matters: a whole-profile override would silently drop the
cutoff and switch off overlap-risk flagging without saying so. Only the fields you
actually supply become `assumed`.

**The default fallback invents nothing.** An unregistered model with no
configured fallback resolves to a profile whose dates are `None`, so every
observation is reported as `missing_model_metadata`. Silence is never read as a
pass.

## What the audit reports

`time_integrity_audit` returns eligible, historically unavailable-model,
training-overlap-risk, and missing-model-metadata counts; the applied policy
with a prose description; the model metadata with its source and status; a
per-document table with a reason for each verdict; and an `assumptions` list
that is non-empty whenever any value is not grounded in a primary source.

A mixed batch is grouped by each row's own model, so one model's dates are
never applied to another's scores.

## Effect on event studies

The headline result covers the primary cohort only. A separate
`all_observations_view` describes the full sample, with its `verdict` and
`evidence_established` fields **removed** so it cannot be read as a conclusion.
The two are never blended. `cohort_coverage` reports how many observations were
held back, and the UI states the coverage consequence plainly: a smaller cohort
is a weaker test, not a worse result.

When nothing survives the screen, the study returns `insufficient_data` and
says why, rather than reporting a result from ineligible observations.

## Backward compatibility

Event-study payloads without a `model` field still validate and still run; their
observations are classified `missing_model_metadata` and held out of the primary
cohort. A filing batch saved before this guard existed has no `time_integrity`
key; the UI shows an explicit notice for it rather than rendering nothing.

Either case can be admitted deliberately with
`{"time_integrity": {"exclude_missing_metadata": false}}`. That is a research
decision the operator records, not a default.

## Limitations

1. **This is a feasibility screen, not a bias measurement.** It shows that a
   score could not have been produced at the time. It does not measure whether
   the model's output was actually influenced by future information. Measuring
   that needs something closer to Look-Ahead-Bench's alpha-decay protocol.
2. **The training cutoff is inferred.** BERT's Wikipedia dump date is not
   published. 2018-12-31 is a conservative upper bound, not a documented fact.
3. **Availability is pipeline-specific.** 2020-12-24 is when these weights
   reached the Hugging Face Hub. A researcher with the weights by another route
   had a different date; that is what the override exists for.
4. **Eligibility is not evidence.** A fully eligible cohort clears this one
   check. It says nothing about sample size, multiple testing, cost realism, or
   whether document tone predicts returns at all.
5. **Only the scoring model is covered.** Other models that could carry the same
   problem are screened only once registered.

## A note on QuantMind

[LLMQuant/quant-mind](https://github.com/LLMQuant/quant-mind) (MIT) was
evaluated during this work as a possible source of components. It is a
document-ingestion and retrieval framework — arXiv/RSS fetching, PDF and HTML
parsing, typed knowledge cards, SQLite persistence, chunking and RAG retrieval.
It contains no backtesting, validation, portfolio, or cost machinery, its
`Factor` type is an explicit stub, its SEC/filings flow is on the roadmap rather
than shipped, and its own evaluation benchmarks are stated as in design with no
published results.

It was therefore **not integrated**. It remains a candidate for a future
literature-ingestion and RAG utility, which AlphaForge has no equivalent of
today. It is not a candidate for any part of the evidence engine, and adopting
it would not make any AlphaForge result harder to fake.
