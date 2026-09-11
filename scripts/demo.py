from __future__ import annotations

import pandas as pd
from qe_backtest.costs import CostModel
from qe_backtest.engine import BacktestConfig, run_backtest
from qe_combine.composite import combine
from qe_core.synthetic import generate
from qe_risk.model import fit_cross_sectional
from qe_risk.neutralize import neutralize
from qe_signals.accruals import Accruals
from qe_signals.normalize import cross_sectional_percentile, size_buckets
from qe_signals.reference import EarningsYield, Momentum12_1
from qe_tax.engine import TaxEngine, TaxProfile

# Synthetic fundamentals are annual, so yoy_lag=1 (real fundq is quarterly).
SIGNALS = [Momentum12_1(), EarningsYield(), Accruals("cash_flow", yoy_lag=1)]
META = {s.name: {"family": s.family, "tier": s.tier} for s in SIGNALS}


def pipeline():
    """Run the entire engine once and hand the artifacts to every test."""
    synth = generate(n_firms=120, start="2015-01-02", end="2018-12-31", seed=77)

    view_all = synth.panel.as_of("2018-12-31")
    prices = view_all.series("prc")
    caps = view_all.series("mktcap")
    returns = view_all.series("ret")

    # --- risk model, then neutralize signals against it -------------------
    risk = fit_cross_sectional(returns, synth.betas, mktcap=caps)

    # Rebalance monthly, starting once momentum has a year of history.
    rebalance_dates = prices.index[260::21]
    assert len(rebalance_dates) > 10, "not enough rebalance dates to be meaningful"

    scores: dict[pd.Timestamp, pd.Series] = {}
    ledgers: dict[pd.Timestamp, object] = {}

    for date in rebalance_dates:
        view = synth.panel.as_of(date)
        buckets = size_buckets(caps.loc[:date].iloc[-1])

        normalized = {}
        for sig in SIGNALS:
            raw = sig.compute(view)
            if raw.empty or raw.notna().sum() < 30:
                continue
            resid = neutralize(raw, synth.betas.reindex(raw.index))
            normalized[sig.name] = cross_sectional_percentile(resid, size_bucket=buckets)

        if len(normalized) < 2:
            continue

        ledger = combine(normalized, META, date)
        ledgers[date] = ledger
        scores[date] = ledger.totals().xs(date, level="date")

    score_panel = pd.DataFrame(scores).T.reindex(columns=prices.columns)

    # --- backtest with an illustrative income ----------------------------
    tax = TaxEngine(TaxProfile(ordinary_income=20_000, portfolio_value=100_000))
    result = run_backtest(
        score_panel,
        prices,
        mktcap=caps,
        config=BacktestConfig(n_positions=15, rebalance_every=21, min_price=1.0),
        cost_model=CostModel(),
        tax_engine=tax,
    )

    return {
        "synth": synth, "prices": prices, "risk": risk, "scores": score_panel,
        "ledgers": ledgers, "result": result, "tax": tax,
        "last_date": max(ledgers),
    }



if __name__ == "__main__":
    import json
    from pathlib import Path
    artifacts = pipeline()
    output = Path("output")
    output.mkdir(exist_ok=True)
    artifacts["result"].equity.to_csv(output / "synthetic-equity.csv")
    artifacts["result"].trades.to_csv(output / "synthetic-trades.csv", index=False)
    artifacts["scores"].to_csv(output / "synthetic-scores.csv")
    print(json.dumps({"dataset": "synthetic", "seed": 77, "firms": 120,
                      "trades": len(artifacts["result"].trades),
                      "output": str(output)}, indent=2))
