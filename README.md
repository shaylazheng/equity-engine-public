# Equity Alpha Engine — public edition

An offline, long-only equity research engine with point-in-time data contracts, residualized signals, attribution ledgers, multiple-testing-aware evaluation, portfolio construction, transaction costs, tax lots, and trade-list generation.

## Run

Install Python 3.13 and uv, then:

```sh
uv sync
uv run python scripts/demo.py
uv run pytest
```

The demo generates a deterministic synthetic panel of 120 firms (seed 77), runs the signal/risk/backtest pipeline, and writes equity, score, and trade CSVs to `output/`. All demo observations are generated locally; no network is used by the engine. Synthetic returns illustrate mechanics and do not establish investment performance.

## Explore

Open `mock/index.html` for the existing interactive explorer. It is an illustrative static presentation with recorded research summaries; changing its controls does not run the Python engine. The executable synthetic example above is the reproducible workflow.

## Architecture

- `qe-core`: three-date point-in-time contract, synthetic generator, attribution ledger.
- `qe-data`: pure identifier, delisting, and fundamental transforms.
- `qe-signals`, `qe-risk`, `qe-combine`: signals, factor residualization, attributable composites.
- `qe-eval`: purged validation, trial registry, Deflated Sharpe, SPA, Reality Check, PBO.
- `qe-backtest`, `qe-tax`: cost-aware portfolio construction and lot accounting.
- `qe-live`: offline trade lists only.
- `qe-report`: self-contained analytical tearsheets.

This edition preserves the computational code and tests while using generated examples. It is maintained separately from the private research workspace. Changes are exported selectively; the private workspace is never mirrored automatically.
