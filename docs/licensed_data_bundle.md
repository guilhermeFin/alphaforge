# Licensed Data Bundle

AlphaForge accepts a local, vendor-neutral export rather than coupling a research
workflow to a single paid-data API. The source can be Sharadar, Compustat,
FactSet, Bloomberg, a data warehouse, or another dataset the user is licensed to
use. AlphaForge reads the files locally and returns derived research results; it
does not upload or redistribute the vendor data.

Set `ALPHAFORGE_LICENSED_DATA_PATH` to the bundle directory. The Data Connections
workspace reports readiness without showing the path, raw data, or any secret.

## Required manifest

`manifest.json` is required and must use schema version 1:

```json
{
  "schema_version": 1,
  "provider_name": "Example licensed vendor",
  "license_acknowledged": true,
  "prices_adjusted_for_corporate_actions": true,
  "point_in_time_fundamentals": true,
  "survivorship_free_universe": true,
  "includes_delisted_securities": true
}
```

These fields are attestations recorded with the research run. They do not replace
the customer's own vendor documentation or license terms.

## Price and volume file

`prices.csv` is required for price factors and portfolio research. It has one row
per date and symbol:

```csv
date,symbol,close,volume
2020-01-02,AAPL,74.36,135480400
2020-01-02,MSFT,160.62,22622100
```

`date` is the trading date, `symbol` is the historical security identifier used
by the bundle, `close` must be positive, and `volume` must be non-negative.
Duplicate date/symbol rows and non-finite values are rejected.

## Historical eligibility file

When the manifest declares a survivorship-free universe, `universe.csv` is
required. It makes membership at each historical date explicit:

```csv
date,symbol,eligible
2020-01-02,AAPL,true
2020-01-02,MSFT,true
```

Ineligible or missing entries are excluded from that date's cross-section. This
prevents a current list of companies from being silently used as history.

## Fundamental observations

`fundamentals.csv` is required only for fundamental factors. It has the canonical
long schema:

```csv
symbol,period_end,available_date,metric,value
AAPL,2020-03-28,2020-05-01,revenue,58313000000
AAPL,2020-03-28,2020-05-01,net_income,11249000000
AAPL,2020-03-28,2020-05-01,shares_diluted,4320000000
```

`available_date` is the earliest date on which the value was public. It must not
precede `period_end`. Metrics must use AlphaForge's canonical raw field names,
such as `revenue`, `cogs`, `net_income`, `shares_diluted`, `total_assets`,
`common_equity`, `op_cash_flow`, and `capex`; see `research/providers.py` for the
complete canonical field map.

Fundamental research remains blocked unless `point_in_time_fundamentals` is
explicitly true. A restated dataset can still support price-factor research, but
it cannot be represented as an as-reported fundamental backtest.

## Built-in adapters

The repository also contains key-gated adapters for SEC EDGAR, SimFin, and
Sharadar SF1. Sharadar's as-reported `datekey` can provide the fundamental side
of a bundle. A complete study still needs historically eligible price/volume and
universe data, so export those inputs together and preserve the manifest.
