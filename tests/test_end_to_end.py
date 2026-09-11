"""The whole stack, on the synthetic panel, with no synthetic dataset connection.

Unit tests verify each package against its own contract. This verifies the
contracts *match* — that a signal's output is shaped the way the risk model
expects, that neutralized signals are what the combiner wants, that a ledger's
totals are what the backtester scores on. Interface drift between packages is
invisible to unit tests and is the failure mode a nine-package workspace actually
suffers from.

It is also the `qe backtest --config configs/smoke.yaml` acceptance criterion in
spirit: the full pipeline runs offline, in seconds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from qe_backtest.costs import CostModel
from qe_backtest.engine import BacktestConfig, run_backtest
from qe_backtest.optimize import (
    OptimizerConfig,
    RiskInputs,
    optimize_weights,
    scores_to_alpha,
)
from qe_combine.composite import combine
from qe_core.synthetic import generate
from qe_eval.registry import TrialRegistry
from qe_eval.stats import deflated_sharpe
from qe_live.tradelist import generate_trade_list
from qe_report.tearsheet import TearsheetInputs, render_tearsheet
from qe_risk.covariance import build_factor_covariance
from qe_risk.model import fit_cross_sectional
from qe_risk.neutralize import neutralize
from qe_signals.accruals import Accruals
from qe_signals.normalize import cross_sectional_percentile, size_buckets
from qe_signals.reference import EarningsYield, Momentum12_1
from qe_tax.engine import TaxEngine, TaxProfile

# Synthetic fundamentals are annual, so yoy_lag=1 (real fundq is quarterly).
SIGNALS = [Momentum12_1(), EarningsYield(), Accruals("cash_flow", yoy_lag=1)]
META = {s.name: {"family": s.family, "tier": s.tier} for s in SIGNALS}


@pytest.fixture(scope="module")
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

    # --- backtest, taxed at a student's income ----------------------------
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


# -- the pipeline runs at all ---------------------------------------------


def test_every_stage_produced_output(pipeline):
    assert not pipeline["scores"].empty
    assert len(pipeline["ledgers"]) > 5
    assert len(pipeline["result"].equity) == len(pipeline["prices"])


def test_scores_cover_a_real_cross_section(pipeline):
    counts = pipeline["scores"].notna().sum(axis=1)
    assert counts.min() > 30, "some rebalance date scored almost nothing"


def test_the_backtest_actually_traded(pipeline):
    trades = pipeline["result"].trades
    assert len(trades) > 20
    assert trades["notional"].abs().sum() > 0


# -- the invariants survive composition -----------------------------------


def test_ledgers_still_reconcile_after_the_full_pipeline(pipeline):
    """Additivity has to hold on real pipeline output, not just synthetic fixtures."""
    for ledger in pipeline["ledgers"].values():
        ledger.reconcile(ledger.totals())


def test_no_flag_or_risk_tier_reached_any_score(pipeline):
    for ledger in pipeline["ledgers"].values():
        assert set(ledger.frame["tier"].unique()) <= {"core", "exploratory"}


def test_the_portfolio_stayed_long_only(pipeline):
    weights = pipeline["result"].weights
    assert (weights >= -1e-12).all().all()
    for shares in pipeline["result"].lots.holdings().values():
        assert shares >= 0


def test_signals_were_neutral_to_risk_at_score_time(pipeline):
    """If neutralization silently stopped working, scores would track beta."""
    last = pipeline["last_date"]
    scores = pipeline["scores"].loc[last].dropna()
    betas = pipeline["synth"].betas["market"].reindex(scores.index)
    assert abs(np.corrcoef(scores, betas)[0, 1]) < 0.35


# -- the tax layer is actually engaged ------------------------------------


def test_high_turnover_forfeits_the_zero_percent_band(pipeline):
    """A real finding rather than a quirk, and worth stating plainly.

    The 0% band applies only to *long-term* gains. Rebalancing monthly means
    almost nothing is held the 365 days needed to qualify, so the gains are
    short-term — ordinary income, taxed at the student's 10-12% marginal rate
    with no free band at all. A large headroom is worth nothing to a strategy
    that never holds anything for a year.

    This is precisely the interaction a flat tax assumption cannot express, and
    it argues for holding-period-aware rebalancing in Phase 8.
    """
    result = pipeline["result"]
    short, long = result.lots.realized_gains()

    assert abs(short) > abs(long), "monthly rebalancing should realize mostly short-term gains"
    assert result.realized_tax.max() > 0, "short-term gains were somehow untaxed"


def test_holding_longer_cuts_the_effective_tax_rate(pipeline):
    """The other side of the same coin, measured properly.

    Holding past 365 days shifts gains into the long-term schedule where the 0%
    band lives. It does not drive the bill to zero — residual short-term gains
    are still ordinary income, and they *also* consume long-term headroom — so
    the honest claim is about the effective rate, not about reaching zero.
    """
    student = TaxEngine(TaxProfile(ordinary_income=20_000))
    slow = run_backtest(
        pipeline["scores"], pipeline["prices"],
        config=BacktestConfig(n_positions=15, rebalance_every=400, min_price=1.0),
        tax_engine=student,
    )

    def effective_rate(result) -> float:
        short, long = result.lots.realized_gains()
        total = short + long
        return float(result.realized_tax.max() / total) if total > 0 else 0.0

    fast_rate = effective_rate(pipeline["result"])
    slow_rate = effective_rate(slow)

    s_short, s_long = slow.lots.realized_gains()
    assert abs(s_long) > abs(s_short), "long holding periods did not produce long-term gains"
    assert slow_rate < fast_rate, (
        f"holding longer did not reduce the effective tax rate "
        f"({slow_rate:.3f} vs {fast_rate:.3f})"
    )


def test_the_same_run_is_taxed_at_a_high_income(pipeline):
    high = TaxEngine(TaxProfile(ordinary_income=700_000))
    taxed = run_backtest(
        pipeline["scores"], pipeline["prices"],
        config=BacktestConfig(n_positions=15, rebalance_every=21, min_price=1.0),
        tax_engine=high,
    )
    assert taxed.realized_tax.max() > 0, (
        "identical trades produced no tax at the top bracket — the tax layer is not wired in"
    )


def test_costs_were_charged(pipeline):
    assert pipeline["result"].costs.sum() > 0


# -- downstream artifacts -------------------------------------------------


def test_a_trade_list_can_be_generated_from_the_final_state(pipeline):
    result = pipeline["result"]
    last = pipeline["prices"].index[-1]
    equity = result.equity.iloc[-1]

    target = pipeline["scores"].dropna(how="all").iloc[-1].nlargest(10)
    target = pd.Series(0.08, index=target.index)

    tl = generate_trade_list(
        target, pipeline["prices"].loc[last], equity,
        book=result.lots, as_of=last, tax_engine=pipeline["tax"],
    )
    assert tl.headroom is not None
    assert not tl.rows.empty


def test_a_tearsheet_renders_from_pipeline_output(pipeline, tmp_path):
    last = pipeline["last_date"]
    html = render_tearsheet(
        TearsheetInputs(
            title="End-to-end smoke",
            equity=pipeline["result"].equity,
            ledger=pipeline["ledgers"][last],
            exposures=pipeline["synth"].betas,
            notes=("Synthetic panel; no synthetic dataset connection.",),
        )
    )
    (tmp_path / "tearsheet.html").write_text(html)

    assert "http" not in html
    assert "= sum of" in html
    for sig in META:
        assert sig in html


def test_a_deflated_sharpe_can_be_computed_with_a_real_trial_count(pipeline):
    """Closes the loop: the registry supplies both the count and the variance."""
    rets = pipeline["result"].returns
    with TrialRegistry() as reg:
        for i, name in enumerate(META):
            reg.record(
                signals=[name], universe="synthetic", start="2015-01-02", end="2018-12-31",
                sr_period=float(0.01 * (i + 1)), n_obs=len(rets),
            )
        inputs = reg.deflation_inputs()

    out = deflated_sharpe(rets, 252, inputs["n_trials"], inputs["var_sr_trials"])
    assert out["var_sr_source"] == "registry"
    assert 0.0 <= out["dsr"] <= 1.0


# -- the guarantee that matters most --------------------------------------


def test_the_pipeline_never_saw_the_future(pipeline):
    """Recompute an early rebalance and confirm it differs from a later one.

    If a signal had somehow captured the full panel, these would match.
    """
    synth = pipeline["synth"]
    early, late = pd.Timestamp("2016-06-30"), pd.Timestamp("2018-06-29")

    sig = Momentum12_1()
    a = sig.compute(synth.panel.as_of(early)).dropna()
    b = sig.compute(synth.panel.as_of(late)).dropna()
    common = a.index.intersection(b.index)

    assert len(common) > 50
    assert not np.allclose(a[common], b[common])


# -- the optimizer path, end to end ---------------------------------------


def test_the_optimizer_runs_on_the_whole_stack(pipeline):
    """Phase 8 through the real pipeline, not a synthetic fixture.

    Everything upstream is the same run every other test uses -- planted betas,
    neutralized signals, a combined ledger. Only construction changes. This is
    the test most likely to catch an interface mismatch between the risk model
    and the optimizer, because unit tests on either side build their own inputs
    and so can agree with each other while disagreeing with reality.
    """
    risk_result = pipeline["risk"]
    betas = pipeline["synth"].betas
    factor_cov = build_factor_covariance(risk_result.factor_returns, annualize=None)
    specific_var = risk_result.specific_returns.pow(2).mean()

    inputs = RiskInputs(
        exposures=betas.reindex(columns=factor_cov.columns).dropna(how="all"),
        factor_cov=factor_cov,
        specific_var=specific_var,
    )

    result = run_backtest(
        pipeline["scores"],
        pipeline["prices"],
        mktcap=pipeline["prices"] * 0 + 1e9,
        config=BacktestConfig(n_positions=15, rebalance_every=21, min_price=1.0),
        cost_model=CostModel(),
        tax_engine=pipeline["tax"],
        risk=lambda _date: inputs,
        optimizer=OptimizerConfig(max_weight=0.15, risk_aversion=5.0),
    )

    assert not result.trades.empty, "the optimizer path executed zero trades"
    assert (result.weights.astype(float) >= -1e-12).all().all()
    assert result.equity.notna().all()


def test_the_optimizer_book_differs_from_equal_weight(pipeline):
    """If it matched, the risk model would not be reaching the decision."""
    risk_result = pipeline["risk"]
    factor_cov = build_factor_covariance(risk_result.factor_returns, annualize=None)
    inputs = RiskInputs(
        exposures=pipeline["synth"].betas.reindex(columns=factor_cov.columns),
        factor_cov=factor_cov,
        specific_var=risk_result.specific_returns.pow(2).mean(),
    )
    cfg = BacktestConfig(n_positions=15, rebalance_every=21, min_price=1.0)
    flat = run_backtest(pipeline["scores"], pipeline["prices"], config=cfg)
    opt = run_backtest(
        pipeline["scores"], pipeline["prices"], config=cfg,
        risk=lambda _date: inputs,
        optimizer=OptimizerConfig(max_weight=0.15, risk_aversion=5.0),
    )
    a = flat.weights.iloc[-1].astype(float)
    b = opt.weights.iloc[-1].astype(float).reindex(a.index).fillna(0.0)
    assert not np.allclose(a.to_numpy(), b.to_numpy(), atol=1e-3)


def test_a_trade_list_can_be_generated_from_an_optimized_book(pipeline):
    """The optimizer has to reach the thing this project actually emits."""
    risk_result = pipeline["risk"]
    factor_cov = build_factor_covariance(risk_result.factor_returns, annualize=None)
    inputs = RiskInputs(
        exposures=pipeline["synth"].betas.reindex(columns=factor_cov.columns),
        factor_cov=factor_cov,
        specific_var=risk_result.specific_returns.pow(2).mean(),
    )
    last = pipeline["last_date"]
    scores = pipeline["scores"].loc[last].dropna()
    alpha = scores_to_alpha(
        scores, ic=0.03, specific_var=inputs.specific_var.reindex(scores.index)
    )
    opt = optimize_weights(alpha, inputs, config=OptimizerConfig(max_weight=0.15))

    tl = generate_trade_list(
        opt.weights,
        pipeline["prices"].loc[last].reindex(opt.weights.index),
        equity=100_000.0,
        book=pipeline["result"].lots,
        as_of=last,
        tax_engine=pipeline["tax"],
    )
    assert not tl.rows.empty
    assert not tl.buys.empty, "an optimized book with no holdings must produce buys"
    assert opt.risk_decomposition().sum() == pytest.approx(1.0)
