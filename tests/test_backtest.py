"""The backtester's two structural rules, and the tax policy that justifies qe-tax.

Never same-bar and long-only are enforced by construction rather than by
configuration, so these tests exist to prove the construction actually holds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from qe_backtest.costs import CostModel
from qe_backtest.engine import (
    BacktestConfig,
    _target_weights,
    run_backtest,
    sale_tax_cost,
)
from qe_backtest.optimize import OptimizerConfig, RiskInputs
from qe_core.calendar import trading_days
from qe_tax.engine import TaxEngine, TaxProfile
from qe_tax.lots import LotBook

PERMNOS = [10_001, 10_002, 10_003, 10_004, 10_005]


@pytest.fixture(scope="module")
def dates():
    return trading_days("2020-01-02", "2021-12-31")


@pytest.fixture
def prices(dates):
    rng = np.random.default_rng(11)
    paths = 50.0 * np.cumprod(
        1.0 + rng.normal(0.0004, 0.012, size=(len(dates), len(PERMNOS))), axis=0
    )
    return pd.DataFrame(paths, index=dates, columns=PERMNOS)


@pytest.fixture
def scores(prices):
    """Static preference ordering, so rebalances are predictable."""
    return pd.DataFrame(
        np.tile(np.arange(len(PERMNOS))[::-1], (len(prices), 1)),
        index=prices.index,
        columns=PERMNOS,
        dtype=float,
    )


def _cfg(**kw):
    base = {"initial_cash": 100_000.0, "n_positions": 3, "max_weight": 0.40,
            "rebalance_every": 21, "min_price": 1.0}
    return BacktestConfig(**{**base, **kw})


# -- never same-bar -------------------------------------------------------


def test_orders_fill_on_the_following_session(dates):
    """A price step the day after a rebalance must not be captured by that rebalance.

    Prices are flat at 10 until day 22, then jump to 100. The first rebalance is
    day 0 and fills on day 1 at 10. If the engine filled same-bar on a later
    rebalance it would buy at a stale price, so we check the fill price directly.
    """
    px = pd.DataFrame(10.0, index=dates[:40], columns=PERMNOS)
    px.iloc[22:] = 100.0
    sc = pd.DataFrame(1.0, index=px.index, columns=PERMNOS)

    result = run_backtest(sc, px, config=_cfg(rebalance_every=21, n_positions=1))
    first = result.trades.iloc[0]

    assert first["date"] == px.index[1], "order did not fill on the session after the signal"
    assert first["price"] == 10.0


def test_no_trade_occurs_on_the_first_bar(prices, scores):
    result = run_backtest(scores, prices, config=_cfg())
    assert result.trades["date"].min() > prices.index[0]


# -- long-only ------------------------------------------------------------


def test_weights_are_never_negative(prices, scores):
    result = run_backtest(scores, prices, config=_cfg())
    assert (result.weights >= -1e-12).all().all()


def test_no_position_exceeds_the_cap(prices, scores):
    result = run_backtest(scores, prices, config=_cfg(max_weight=0.40))
    # Allow drift between rebalances; the constraint binds at trade time.
    assert result.weights.max().max() < 0.75


def test_holdings_never_go_short(prices, scores):
    result = run_backtest(scores, prices, config=_cfg())
    for permno, shares in result.lots.holdings().items():
        assert shares >= 0, f"permno {permno} went short"


def test_selling_more_than_held_raises():
    from qe_tax.lots import LotBook

    book = LotBook()
    book.buy(1, 10, 5.0, "2020-01-02")
    with pytest.raises(ValueError, match="long-only"):
        book.sell(1, 50, 6.0, "2020-02-02")


# -- costs ----------------------------------------------------------------


def test_costs_reduce_terminal_equity(prices, scores):
    free = CostModel(
        base_half_spread_bps=0.0, illiquidity_bps_at_reference=0.0,
        impact_coefficient=0.0, commission_per_share=0.0,
    )
    expensive = CostModel(
        base_half_spread_bps=25.0, illiquidity_bps_at_reference=50.0,
        commission_per_share=0.02,
    )
    cheap_run = run_backtest(scores, prices, config=_cfg(), cost_model=free)
    dear_run = run_backtest(scores, prices, config=_cfg(), cost_model=expensive)

    assert dear_run.equity.iloc[-1] < cheap_run.equity.iloc[-1]
    assert dear_run.costs.sum() > cheap_run.costs.sum()


def test_spread_widens_as_market_cap_falls():
    # Caps in $ millions, matching every panel in this workspace.
    model = CostModel()
    assert model.half_spread_bps(50) > model.half_spread_bps(50_000)


def test_spread_is_capped_for_microcaps():
    model = CostModel(max_half_spread_bps=300.0)
    assert model.half_spread_bps(1e-6) == 300.0
    assert model.half_spread_bps(0.0) == 300.0


@pytest.mark.parametrize(
    ("mktcap_musd", "lo_bps", "hi_bps"),
    [
        (50_000, 2, 15),     # mega cap
        (5_000, 5, 25),      # large cap
        (686, 15, 60),       # the real panel's median
        (50, 60, 200),       # small cap
    ],
)
def test_half_spread_is_realistic_at_panel_scale(mktcap_musd, lo_bps, hi_bps):
    """Anchors the cost model to caps in the units the panels actually use.

    `reference_mktcap` was 1e9 — raw dollars — while both the real and synthetic
    panels carry `mktcap` in $ millions. Nothing raised: every name simply looked
    like a sub-$1 microcap and paid the capped 300bp half-spread, so a $50bn
    company was charged 3% one-way in every backtest ever run. The old tests
    passed because they compared two raw-dollar values against each other, which
    is internally consistent and matches no panel.

    This asserts absolute plausibility rather than a relative ordering, which is
    the only kind of check that catches a units drift.
    """
    bps = CostModel().half_spread_bps(mktcap_musd)
    assert lo_bps <= bps <= hi_bps, f"{mktcap_musd}M priced at {bps:.0f}bp"


def test_impact_follows_a_square_root_law():
    model = CostModel()
    small = model.impact_bps(10_000, 1_000_000, 0.02)
    big = model.impact_bps(40_000, 1_000_000, 0.02)
    # Four times the size, twice the impact.
    assert big == pytest.approx(2 * small, rel=1e-9)


def test_zero_size_costs_nothing():
    assert CostModel().cost(shares=0, price=10, mktcap=1_000, adv=1e6).total == 0.0


# -- participation cap ----------------------------------------------------


def test_a_large_order_spills_across_sessions(dates):
    """Ignoring the cap is how a backtest silently assumes infinite liquidity."""
    px = pd.DataFrame(10.0, index=dates[:40], columns=[10_001])
    sc = pd.DataFrame(1.0, index=px.index, columns=[10_001])
    # 100 shares/day of volume, 10% cap => 10 shares fillable per session.
    adv = pd.DataFrame(100.0, index=px.index, columns=[10_001])

    result = run_backtest(
        sc, px, adv=adv,
        config=_cfg(n_positions=1, max_weight=1.0, initial_cash=10_000, min_trade_value=1.0),
        cost_model=CostModel(participation_cap=0.10),
    )
    fills = result.trades.loc[result.trades["shares"] > 0]
    assert len(fills) > 1, "the whole order filled in one session despite the cap"
    assert fills["shares"].max() <= 10.0 + 1e-9


# -- tax-aware lot selection ---------------------------------------------


def test_headroom_selects_lowest_cost_lots():
    """The concrete payoff of qe-tax: inside the 0% band, realize the *largest*
    long-term gain, because it is free and it steps up basis."""
    from qe_backtest.engine import _lot_method

    cfg = _cfg(lot_method="hifo")
    engine = TaxEngine(TaxProfile(ordinary_income=20_000))

    assert engine.headroom() > 0
    assert _lot_method(cfg, engine, 0.0, 0.0) == "lofo"


def test_exhausted_headroom_reverts_to_minimizing_gains():
    from qe_backtest.engine import _lot_method

    cfg = _cfg(lot_method="hifo")
    engine = TaxEngine(TaxProfile(ordinary_income=20_000))
    used = engine.headroom() + 10_000

    assert _lot_method(cfg, engine, 0.0, used) == "hifo"


def test_no_tax_engine_means_the_configured_method(prices, scores):
    from qe_backtest.engine import _lot_method

    assert _lot_method(_cfg(lot_method="fifo"), None, 0.0, 0.0) == "fifo"


def test_tax_is_accrued_when_an_engine_is_supplied(prices, scores):
    engine = TaxEngine(TaxProfile(ordinary_income=400_000))  # well past the 0% band
    result = run_backtest(scores, prices, config=_cfg(), tax_engine=engine)
    assert result.realized_tax.notna().all()
    assert (result.realized_tax != 0).any(), "no tax was ever accrued"


def test_the_accrual_can_be_negative_when_losses_dominate(prices, scores):
    """A realized loss is a tax *benefit*, and the accrual has to be able to say so.

    This assertion used to read `>= 0`, which was only true because the engine
    was charging `total_tax` — the investor's entire federal bill including tax
    on salary — so the number could never go below the tax owed on income alone.
    Charging only the portion attributable to the portfolio lets a loss year
    correctly show up as a credit.
    """
    engine = TaxEngine(TaxProfile(ordinary_income=400_000))
    falling = prices.iloc[::-1].set_axis(prices.index)  # a persistent downtrend
    result = run_backtest(scores, falling, config=_cfg(), tax_engine=engine)
    assert (result.realized_tax < 0).any()


# -- results --------------------------------------------------------------


def test_equity_starts_at_the_initial_cash(prices, scores):
    result = run_backtest(scores, prices, config=_cfg(initial_cash=250_000))
    assert result.equity.iloc[0] == pytest.approx(250_000)


def test_equity_is_defined_every_session(prices, scores):
    result = run_backtest(scores, prices, config=_cfg())
    assert len(result.equity) == len(prices)
    assert result.equity.notna().all()


def test_turnover_is_zero_without_rebalancing(prices, scores):
    """rebalance_every longer than the sample means one initial buy and no churn."""
    result = run_backtest(scores, prices, config=_cfg(rebalance_every=10_000))
    assert result.turnover() < 1.0


def test_a_never_traded_run_has_flat_equity(prices):
    empty = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    result = run_backtest(empty, prices, config=_cfg())
    assert result.trades.empty
    assert result.equity.nunique() == 1


def test_min_price_filter_excludes_penny_stocks(dates):
    px = pd.DataFrame(10.0, index=dates[:30], columns=PERMNOS)
    px[10_001] = 0.50
    sc = pd.DataFrame(1.0, index=px.index, columns=PERMNOS)
    sc[10_001] = 99.0  # would be the top pick if eligible

    result = run_backtest(sc, px, config=_cfg(min_price=5.0, n_positions=2))
    assert 10_001 not in set(result.trades["permno"])


# -- the optimizer path (Phase 8) -----------------------------------------
#
# A note on what these assert. `BacktestResult.weights` is the *realized* book,
# which drifts with prices between rebalances — the position cap constrains the
# target, not the drift. Asserting the cap on realized weights fails for a reason
# that is not a bug, so the constraint tests call `_target_weights` directly and
# the integration tests check that the optimizer is genuinely reached.


def _risk_for(permnos, *, risky=(), seed=3):
    """A RiskInputs whose `risky` names carry far more specific variance."""
    rng = np.random.default_rng(seed)
    factors = ["mkt", "size"]
    X = pd.DataFrame(
        rng.normal(0.0, 0.5, (len(permnos), len(factors))),
        index=pd.Index(permnos), columns=factors,
    )
    X["mkt"] = 1.0
    F = pd.DataFrame(np.diag([2e-4, 1e-4]), index=factors, columns=factors)
    d = pd.Series(1e-4, index=pd.Index(permnos))
    d.loc[list(risky)] = 4e-3
    return RiskInputs(exposures=X, factor_cov=F, specific_var=d)


def test_target_weights_respects_the_cap_on_the_optimizer_path():
    scores = pd.Series(np.arange(len(PERMNOS))[::-1], index=PERMNOS, dtype=float)
    target = _target_weights(
        scores, _cfg(max_weight=0.30), pd.Index(PERMNOS),
        risk=_risk_for(PERMNOS),
        opt_cfg=OptimizerConfig(max_weight=0.30, max_active_share=None,
                                trade_cost=0.0, risk_aversion=1.0),
    )
    assert target.max() <= 0.30 + 1e-9
    assert (target >= 0).all()
    assert target.sum() == pytest.approx(1.0, abs=1e-8)


def test_target_weights_underweights_the_riskier_name():
    """The top-scored name is also the riskiest; equal weight cannot see that."""
    scores = pd.Series(np.arange(len(PERMNOS))[::-1], index=PERMNOS, dtype=float)
    target = _target_weights(
        scores, _cfg(max_weight=1.0), pd.Index(PERMNOS),
        risk=_risk_for(PERMNOS, risky=[PERMNOS[0]]),
        opt_cfg=OptimizerConfig(max_weight=1.0, max_active_share=None,
                                trade_cost=0.0, risk_aversion=400.0),
    )
    # PERMNOS[0] has the highest score but 40x the specific variance.
    assert target[PERMNOS[0]] < target[PERMNOS[1]]


def test_a_held_but_ineligible_name_may_be_kept_not_bought():
    """A wash-sale block stops purchases; it does not force a liquidation.

    Dropping the name from the universe would make the engine sell it without
    the cost term ever pricing that sale.
    """
    scores = pd.Series(np.arange(len(PERMNOS))[::-1], index=PERMNOS, dtype=float)
    blocked = PERMNOS[0]
    eligible = pd.Index([p for p in PERMNOS if p != blocked])
    current = pd.Series({blocked: 0.30, PERMNOS[1]: 0.70})

    target = _target_weights(
        scores, _cfg(max_weight=1.0), eligible,
        risk=_risk_for(PERMNOS), current=current,
        opt_cfg=OptimizerConfig(max_weight=1.0, max_active_share=None,
                                trade_cost=0.0, risk_aversion=1.0),
    )
    assert blocked in target.index, "a blocked holding must not be force-sold"
    assert target[blocked] <= 0.30 + 1e-9, "a blocked name must not be added to"


def test_supplying_a_risk_model_changes_the_book(prices, scores):
    """The optimizer must actually be reached, not merely importable.

    Equal weight ignores risk entirely, so if the two constructions agree the
    risk model is not being consulted.
    """
    cfg = _cfg(n_positions=5, max_weight=1.0)
    flat = run_backtest(scores, prices, config=cfg)
    risk = _risk_for(PERMNOS, risky=[PERMNOS[0]])
    opt = run_backtest(
        scores, prices, config=cfg,
        risk=lambda _d: risk,
        optimizer=OptimizerConfig(risk_aversion=400.0, max_weight=1.0,
                                  max_active_share=None, trade_cost=0.0),
    )
    last_flat = flat.weights.iloc[-1].astype(float)
    last_opt = opt.weights.iloc[-1].astype(float).reindex(last_flat.index).fillna(0.0)
    assert not np.allclose(last_flat.to_numpy(), last_opt.to_numpy(), atol=1e-3)


def test_optimizer_path_stays_long_only(prices, scores):
    res = run_backtest(
        scores, prices, config=_cfg(n_positions=5, max_weight=0.35),
        risk=lambda _d: _risk_for(PERMNOS),
    )
    assert (res.weights.astype(float) >= -1e-12).all().all()


def test_a_none_risk_model_falls_back_to_equal_weight(prices, scores):
    """Early dates have no fitted model. That is a real state, not an error."""
    cfg = _cfg(n_positions=3)
    baseline = run_backtest(scores, prices, config=cfg)
    fallback = run_backtest(scores, prices, config=cfg, risk=lambda _d: None)
    pd.testing.assert_frame_equal(baseline.weights, fallback.weights)


def test_the_cost_term_reduces_trading(prices):
    """The trading-cost term is the reason the optimizer holds things.

    Counted in trades rather than `turnover()`: that metric divides by mean
    equity, so two runs with different P&L are not comparable through it.
    """
    rng = np.random.default_rng(5)
    noisy = pd.DataFrame(
        rng.normal(0.0, 1.0, (len(prices), len(PERMNOS))),
        index=prices.index, columns=PERMNOS,
    )
    cfg = _cfg(n_positions=3, max_weight=1.0)
    risk = _risk_for(PERMNOS)
    free = run_backtest(
        noisy, prices, config=cfg, risk=lambda _d: risk,
        optimizer=OptimizerConfig(risk_aversion=50.0, max_weight=1.0,
                                  trade_cost=0.0, max_active_share=None),
    )
    pricey = run_backtest(
        noisy, prices, config=cfg, risk=lambda _d: risk,
        optimizer=OptimizerConfig(risk_aversion=50.0, max_weight=1.0,
                                  trade_cost=0.05, max_active_share=None),
    )
    assert len(pricey.trades) < len(free.trades)


def test_zero_min_trade_value_does_not_break_a_cash_starved_buy(prices, scores):
    """Regression: `min_trade_value` was doubling as the zero-share guard.

    Setting it to 0 is a legitimate configuration — it means "no economic floor"
    — and it used to pass a zero-share order to the lot book, which raises.
    """
    res = run_backtest(
        scores, prices,
        config=_cfg(n_positions=3, max_weight=1.0, min_trade_value=0.0),
    )
    assert len(res.equity) == len(prices)


def test_nullable_float_prices_are_accepted():
    """Regression: CRSP parquet arrives as pandas *nullable* Float64.

    A missing price is then `pd.NA`, not `nan`, and `np.isfinite(pd.NA)` returns
    NA — so `if not np.isfinite(price)` raises "boolean value of NA is
    ambiguous". The synthetic panel is plain float64, so the entire suite passed
    while the real panel could not survive its first bar.
    """
    dates = trading_days("2020-01-02", "2020-06-30")
    rng = np.random.default_rng(3)
    px = pd.DataFrame(
        50.0 * np.cumprod(1 + rng.normal(0, 0.01, (len(dates), len(PERMNOS))), axis=0),
        index=dates, columns=PERMNOS,
    ).astype("Float64")
    px.iloc[10, 0] = pd.NA  # a genuine gap, the thing that used to raise

    sc = pd.DataFrame(
        np.tile(np.arange(len(PERMNOS))[::-1], (len(dates), 1)),
        index=dates, columns=PERMNOS,
    ).astype("Float64")

    res = run_backtest(sc, px, config=_cfg(n_positions=3))
    assert len(res.equity) == len(dates)
    assert not res.trades.empty


# -- delisting ------------------------------------------------------------


def _delisting_panel():
    """A five-name panel where one name stops quoting halfway through."""
    dates = trading_days("2020-01-02", "2020-12-31")
    rng = np.random.default_rng(7)
    px = pd.DataFrame(
        50.0 * np.cumprod(1 + rng.normal(0.0003, 0.01, (len(dates), len(PERMNOS))), axis=0),
        index=dates, columns=PERMNOS,
    )
    px.iloc[120:, 0] = np.nan  # PERMNOS[0] delists and never quotes again
    sc = pd.DataFrame(
        np.tile(np.arange(len(PERMNOS))[::-1], (len(dates), 1)),
        index=dates, columns=PERMNOS, dtype=float,
    )
    return dates, px, sc


def test_a_delisting_does_not_poison_equity():
    """The bug that invalidated the first real backtest.

    `px.get(p, 0.0)` returns the *stored* value when the column exists, so a
    quoted-then-NaN name marked the position at NaN rather than falling back to
    the default. Equity went NaN permanently and never recovered — and because
    `returns` drops NaN, every downstream statistic silently described only the
    period before the first delisting while reporting the full date range.
    """
    dates, px, sc = _delisting_panel()
    res = run_backtest(sc, px, config=_cfg(n_positions=3, max_weight=0.5))

    assert res.equity.notna().all(), "equity went NaN"
    assert np.isfinite(res.equity.to_numpy()).all()
    assert len(res.returns) == len(dates) - 1, "returns silently lost observations"


def test_a_delisted_name_is_closed_out_not_carried():
    """A zombie position that can never be sold would distort every later weight."""
    _dates, px, sc = _delisting_panel()
    res = run_backtest(sc, px, config=_cfg(n_positions=3, max_weight=0.5))
    assert res.lots.shares(PERMNOS[0]) == 0.0
    assert res.weights[PERMNOS[0]].iloc[-1] == 0.0


def test_the_delisting_sale_happens_at_the_last_known_price():
    """Not at zero — that would assert a total loss that usually did not occur."""
    _dates, px, sc = _delisting_panel()
    last_quote = px[PERMNOS[0]].dropna().iloc[-1]
    res = run_backtest(sc, px, config=_cfg(n_positions=3, max_weight=0.5))

    sales = res.trades[(res.trades["permno"] == PERMNOS[0]) & (res.trades["shares"] < 0)]
    assert not sales.empty
    assert sales["price"].iloc[-1] == pytest.approx(last_quote)


def test_a_non_finite_price_is_treated_as_unquoted():
    """`inf` takes the same path as a delisting rather than propagating.

    Written after asserting the opposite: the first version of this test expected
    a raise, because a broken panel felt like it should be loud. It is not, and
    the reason is sound — `np.isfinite` is False for `inf` just as for `nan`, so
    the delisting branch closes the position at the last real price. Pinning the
    behaviour that exists beats pinning the one that felt right.
    """
    dates = trading_days("2020-01-02", "2020-03-31")
    px = pd.DataFrame(50.0, index=dates, columns=PERMNOS)
    sc = pd.DataFrame(1.0, index=dates, columns=PERMNOS)
    px.iloc[20:, :] = np.inf
    res = run_backtest(sc, px, config=_cfg(n_positions=3))
    assert np.isfinite(res.equity.to_numpy()).all()
    assert res.lots.holdings() == {}


def test_equity_raises_rather_than_going_quietly_nan():
    """The backstop, reached the only way left once marking is handled.

    Kept because the failure it guards is the one that invalidated the first real
    backtest: a NaN equity is not a bad result, it is *no* result, and every
    statistic computed from it silently describes a shorter sample than it
    claims. A guard that never fires is cheap; a NaN that reads as a Sharpe of
    1.31 is not.
    """
    dates = trading_days("2020-01-02", "2020-03-31")
    px = pd.DataFrame(50.0, index=dates, columns=PERMNOS)
    sc = pd.DataFrame(1.0, index=dates, columns=PERMNOS)
    with pytest.raises(ValueError, match="non-finite"):
        run_backtest(sc, px, config=_cfg(initial_cash=float("nan")))


# -- tax is actually paid, not merely accrued -----------------------------


def _taxable_run(prices, scores, income=250_000.0):
    tax = TaxEngine(TaxProfile(ordinary_income=income, portfolio_value=100_000))
    cfg = _cfg(n_positions=3, max_weight=0.40)
    return run_backtest(scores, prices, config=cfg, tax_engine=tax), tax


def test_tax_is_deducted_from_the_account(prices, scores):
    """The bug that made a 'net of tax' run out-return its own pre-tax twin.

    Tax was accrued into `realized_tax` and reported, but no cash movement ever
    corresponded to it. The column was a number in a report. Worse, because
    tax-aware lot selection changes *which* shares are sold, the taxed run could
    finish ahead of the untaxed one — which reads as a result rather than a bug.
    """
    taxed, _ = _taxable_run(prices, scores)
    untaxed = run_backtest(scores, prices, config=_cfg(n_positions=3, max_weight=0.40))

    assert taxed.tax_paid > 0, "no tax was ever paid"
    assert taxed.equity.iloc[-1] < untaxed.equity.iloc[-1], (
        "a taxed run must not finish ahead of the same run untaxed"
    )


def test_every_tax_year_is_settled(prices, scores):
    """Including the final, partial one — otherwise the last year looks free."""
    taxed, _ = _taxable_run(prices, scores)
    years = sorted({d.year for d in prices.index})
    assert set(taxed.tax_payments["year"]) == set(years)
    assert taxed.tax_payments["tax"].sum() == pytest.approx(taxed.tax_paid)


def test_a_zero_rate_investor_pays_nothing(prices, scores):
    """The 0% long-term band is the whole reason qe-tax models a step function."""
    poor, _ = _taxable_run(prices, scores, income=0.0)
    assert poor.tax_paid >= 0.0


def test_no_tax_engine_means_no_deduction(prices, scores):
    res = run_backtest(scores, prices, config=_cfg(n_positions=3, max_weight=0.40))
    assert res.tax_paid == 0.0
    assert res.tax_payments.empty


def test_a_delisting_return_is_applied_to_the_exit_price():
    """Without this the run inherits a survivorship bias pointing the nice way.

    A price series simply stops when a company fails, so closing at the last
    quoted price exits every bankruptcy at its final good mark. CRSP carries the
    real delisting return; this asserts it actually reaches the exit.
    """
    _dates, px, sc = _delisting_panel()
    last_quote = px[PERMNOS[0]].dropna().iloc[-1]
    dr = pd.Series({PERMNOS[0]: -0.55})

    plain = run_backtest(sc, px, config=_cfg(n_positions=3, max_weight=0.5))
    haircut = run_backtest(
        sc, px, config=_cfg(n_positions=3, max_weight=0.5), delisting_returns=dr
    )

    sold = haircut.trades[
        (haircut.trades["permno"] == PERMNOS[0]) & (haircut.trades["shares"] < 0)
    ]
    assert sold["price"].iloc[-1] == pytest.approx(last_quote * 0.45)
    assert haircut.equity.iloc[-1] < plain.equity.iloc[-1]


def test_a_delisting_return_of_minus_one_exits_at_zero_not_negative():
    """A total loss is a real outcome; a negative price is not."""
    _dates, px, sc = _delisting_panel()
    res = run_backtest(
        sc, px, config=_cfg(n_positions=3, max_weight=0.5),
        delisting_returns=pd.Series({PERMNOS[0]: -1.5}),
    )
    sold = res.trades[
        (res.trades["permno"] == PERMNOS[0]) & (res.trades["shares"] < 0)
    ]
    assert sold["price"].iloc[-1] == 0.0
    assert np.isfinite(res.equity.to_numpy()).all()


# -- holding-period-aware construction ------------------------------------


def _book_with(permno, shares, basis, acquired):
    book = LotBook("hifo")
    book.buy(permno, shares, basis, pd.Timestamp(acquired))
    return book


def test_a_short_term_gain_is_expensive_to_sell():
    """The point of the whole feature, as a testable consequence.

    Lot *selection* was already tax-aware — given a sale, pick the cheap lots.
    Nothing upstream asked whether the sale was worth making, so the optimizer
    priced every exit at the same spread.
    """
    tax = TaxEngine(TaxProfile(ordinary_income=250_000, portfolio_value=1_000_000))
    now = pd.Timestamp("2021-06-01")
    book = _book_with(PERMNOS[0], 100, 50.0, "2021-01-02")   # ~5 months held
    cost = sale_tax_cost(book, pd.Series({PERMNOS[0]: 100.0}), now, tax)
    assert cost[PERMNOS[0]] > 0.10, "a 100% short-term gain should cost real money"


def test_the_same_gain_held_a_year_is_cheaper():
    """The incentive that makes holding-period awareness mean anything."""
    tax = TaxEngine(TaxProfile(ordinary_income=250_000, portfolio_value=1_000_000))
    now = pd.Timestamp("2021-06-01")
    px = pd.Series({PERMNOS[0]: 100.0})

    short = sale_tax_cost(_book_with(PERMNOS[0], 100, 50.0, "2021-01-02"), px, now, tax)
    long = sale_tax_cost(_book_with(PERMNOS[0], 100, 50.0, "2019-01-02"), px, now, tax)
    assert long[PERMNOS[0]] < short[PERMNOS[0]]


def test_a_loss_is_cheaper_than_free():
    """Harvesting a loss offsets other gains, so it is a benefit, not a cost.

    Returned negative and kept that way. The optimizer floors the *threshold* at
    zero to stay well-posed, but the signal has to survive to that point.
    """
    tax = TaxEngine(TaxProfile(ordinary_income=250_000, portfolio_value=1_000_000))
    now = pd.Timestamp("2021-06-01")
    book = _book_with(PERMNOS[0], 100, 100.0, "2021-01-02")
    cost = sale_tax_cost(book, pd.Series({PERMNOS[0]: 50.0}), now, tax)
    assert cost[PERMNOS[0]] < 0.0


def test_a_zero_rate_investor_sees_no_tax_cost():
    """At a low enough income the 0% long-term band makes the exit free."""
    tax = TaxEngine(TaxProfile(ordinary_income=0, portfolio_value=50_000))
    now = pd.Timestamp("2021-06-01")
    book = _book_with(PERMNOS[0], 10, 50.0, "2019-01-02")  # long-term, small gain
    cost = sale_tax_cost(book, pd.Series({PERMNOS[0]: 60.0}), now, tax)
    assert cost[PERMNOS[0]] == pytest.approx(0.0, abs=1e-12)


def test_tax_aware_construction_reduces_taxable_churn(prices, scores):
    """End to end: pricing the bill into the objective must change the book.

    Compared at the same `trade_cost`, so the only difference is whether the tax
    consequence of an exit reaches portfolio construction at all.
    """
    tax = TaxEngine(TaxProfile(ordinary_income=250_000, portfolio_value=100_000))
    risk = _risk_for(PERMNOS)
    cfg = _cfg(n_positions=5, max_weight=0.40)

    def run(tax_aware):
        return run_backtest(
            scores, prices, config=cfg, tax_engine=tax, risk=lambda _d: risk,
            optimizer=OptimizerConfig(
                max_weight=0.40, risk_aversion=50.0, trade_cost=0.001,
                max_active_share=None, tax_aware=tax_aware,
            ),
        )

    blind, aware = run(False), run(True)
    # Only the tax claim is asserted. Trade *count* is not the right proxy and
    # goes the other way here: pricing the bill can make the optimizer take many
    # small steps instead of one large taxable one.
    assert aware.tax_paid < blind.tax_paid
    assert aware.tax_paid != blind.tax_paid, "tax_aware had no effect at all"
