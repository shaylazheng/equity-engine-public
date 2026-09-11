"""Signal library: correct computation, PIT safety, and honest normalization.

Note what these do *not* claim. On synthetic data, fundamentals do not predict
returns — the generator drives returns from planted factors, not from firm
health. So these test that signals compute what they say they compute and cannot
see the future. Whether they *predict* is the replication gate's job, and that
needs real data.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from qe_core.synthetic import generate
from qe_signals.accruals import Accruals
from qe_signals.base import REGISTRY, BaseSignal, SignalRegistry
from qe_signals.normalize import (
    apply_materiality,
    cross_sectional_percentile,
    dual_baseline,
    own_history_z,
    size_buckets,
)
from qe_signals.reference import EarningsYield, Momentum12_1


@pytest.fixture(scope="module")
def synth():
    return generate(n_firms=80, start="2015-01-02", end="2018-12-31", seed=31)


@pytest.fixture(scope="module")
def view(synth):
    return synth.panel.as_of("2018-12-31")


# -- the registry ---------------------------------------------------------


def test_reference_signals_are_registered():
    assert {"momentum_12_1", "earnings_yield", "accruals"} <= set(REGISTRY.names())


def test_all_registered_signals_pass_the_contract():
    from qe_core.signal import validate_signal

    for sig in REGISTRY:
        validate_signal(sig)


def test_registry_rejects_a_duplicate_name():
    reg = SignalRegistry()
    reg.register(Momentum12_1())
    with pytest.raises(ValueError, match="already registered"):
        reg.register(Momentum12_1())


def test_registry_rejects_an_inconsistent_signal():
    class Broken(Momentum12_1):
        name = "broken"
        evidence = "flag"   # flag evidence on a core tier
        tier = "core"

    with pytest.raises(Exception, match="tier='flag'"):
        SignalRegistry().register(Broken())


def test_scored_and_flag_partitions_are_disjoint():
    scored = {s.name for s in REGISTRY.scored()}
    flags = REGISTRY.flag_names()
    assert not scored & flags


# -- momentum -------------------------------------------------------------


def test_momentum_matches_a_hand_computed_return(view):
    got = Momentum12_1().compute(view)
    prices = view.series("prc")
    permno = got.dropna().index[0]

    start = prices[permno].iloc[-253]
    end = prices[permno].iloc[-22]
    assert got.loc[permno] == pytest.approx(end / start - 1.0)


def test_momentum_skips_the_most_recent_month(view):
    """The skip matters: including the last month mixes in short-term reversal."""
    prices = view.series("prc")
    got = Momentum12_1().compute(view)
    permno = got.dropna().index[0]

    naive = prices[permno].iloc[-1] / prices[permno].iloc[-253] - 1.0
    assert got.loc[permno] != pytest.approx(naive)


def test_momentum_needs_a_year_of_history(synth):
    early = synth.panel.as_of("2015-03-02")
    assert Momentum12_1().compute(early).empty


def test_momentum_covers_the_cross_section(view):
    got = Momentum12_1().compute(view)
    assert got.notna().sum() > 70


# -- earnings yield -------------------------------------------------------


def test_earnings_yield_is_income_over_market_cap(view):
    got = EarningsYield().compute(view)
    wide = view.pivot(["ni", "mktcap"])
    permno = got.dropna().index[0]
    assert got.loc[permno] == pytest.approx(
        wide.loc[permno, "ni"] / wide.loc[permno, "mktcap"]
    )


def test_earnings_yield_is_quarterly_native():
    """Recomputing a filing-driven number daily repeats it ~60x a quarter."""
    assert EarningsYield().native_freq == "quarterly"


# -- accruals -------------------------------------------------------------


def test_cash_flow_accruals_is_the_earnings_cash_wedge(view):
    """Synthetic fundamentals are ANNUAL, so the year-over-year lag is 1, not 4.

    A lag of 4 would difference against four years ago and still return a
    number — which is why the lag is an explicit parameter rather than a
    constant."""
    got = Accruals("cash_flow", yoy_lag=1).compute(view)
    wide = view.pivot(["ni", "oancf"])
    assets = view.series("at")
    permno = got.dropna().index[0]

    expected = (wide.loc[permno, "ni"] - wide.loc[permno, "oancf"]) / assets[permno].tail(2).mean()
    assert got.loc[permno] == pytest.approx(expected)


def test_balance_sheet_accruals_computes(view):
    """Sloan's own construction, and the package default."""
    got = Accruals("balance_sheet", yoy_lag=1).compute(view).dropna()
    assert len(got) > 50, "balance-sheet arm produced almost nothing"
    assert got.abs().median() < 1.0, "accruals should be a small fraction of assets"


def test_the_two_constructions_agree_in_sign(view, synth):
    """The claim that justifies treating them as ONE signal rather than two.

    Both measure the same underlying wedge, so they should rank firms similarly.
    If they disagreed, they would be two signals and would cost two DSR trials."""
    bs = Accruals("balance_sheet", yoy_lag=1).compute(view)
    cf = Accruals("cash_flow", yoy_lag=1).compute(view)
    common = bs.dropna().index.intersection(cf.dropna().index)
    assert len(common) > 50
    assert np.corrcoef(bs[common], cf[common])[0, 1] > 0.3


def test_an_invalid_lag_is_rejected():
    with pytest.raises(ValueError, match="yoy_lag"):
        Accruals(yoy_lag=0)


def test_accruals_expects_a_negative_sign():
    """High accruals predict underperformance — getting this backwards is common."""
    assert Accruals().expected_sign == -1


def test_accruals_tracks_the_planted_wedge(synth, view):
    """The generator widens the accrual wedge as health falls, so the computed
    signal must correlate negatively with health — otherwise it is not measuring
    what it claims."""
    got = Accruals("cash_flow", yoy_lag=1).compute(view).dropna()
    health = synth.health.reindex(got.index)
    assert np.corrcoef(got, health)[0, 1] < -0.3


# -- the property that matters most --------------------------------------


@pytest.mark.parametrize(
    "signal", [Momentum12_1(), EarningsYield(), Accruals("cash_flow", yoy_lag=1)]
)
def test_no_signal_can_see_the_future(synth, signal):
    """Every signal runs off a view, and a view cannot hold future data."""
    early = synth.panel.as_of("2017-06-30")
    late = synth.panel.as_of("2018-12-31")

    got_early = signal.compute(early)
    got_late = signal.compute(late)
    if got_early.empty or got_late.empty:
        pytest.skip("insufficient history for this signal at the early date")

    common = got_early.dropna().index.intersection(got_late.dropna().index)
    assert len(common) > 0
    # Values must differ — if they matched, the "early" computation was using
    # data it should not have had.
    assert not np.allclose(got_early[common], got_late[common])


def test_a_misconfigured_signal_fails_loudly(synth):
    """A concept that never exists anywhere is a wiring bug, and must raise."""

    class NeedsMissing(Momentum12_1):
        name = "needs_missing"
        requires: ClassVar[list[str]] = ["a_concept_that_does_not_exist"]

    with pytest.raises(KeyError, match="never contains required concept"):
        NeedsMissing().validate_against_panel(synth.panel)


def test_a_signal_whose_inputs_are_not_yet_knowable_stays_quiet(synth):
    """Distinct from misconfiguration. Fundamentals do not exist before the first
    filing lands — that is the PIT layer working, not a fault — so the signal
    returns nothing instead of blowing up the opening months of every backtest.
    """
    early = synth.panel.as_of("2015-06-30")
    assert not EarningsYield().has_inputs(early)
    assert EarningsYield().compute(early).empty

    # ...and the same signal is properly configured against the panel as a whole.
    EarningsYield().validate_against_panel(synth.panel)


def test_registry_validates_every_signal_against_a_panel(synth):
    REGISTRY.validate_requirements(synth.panel)


# -- normalization --------------------------------------------------------


def test_percentile_is_centred_and_bounded():
    s = pd.Series(np.arange(100, dtype=float))
    out = cross_sectional_percentile(s)
    assert out.min() >= -1.0 and out.max() <= 1.0
    assert abs(out.median()) < 0.05


def test_percentile_is_robust_to_an_outlier():
    """Why the cross-sectional leg is a percentile and not a z-score."""
    normal = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    with_outlier = pd.Series([1.0, 2.0, 3.0, 4.0, 1e6])

    a = cross_sectional_percentile(normal)
    b = cross_sectional_percentile(with_outlier)
    assert np.allclose(a.to_numpy(), b.to_numpy()), "one blow-up moved everyone's score"


def test_percentile_respects_peer_groups():
    s = pd.Series([1.0, 2.0, 3.0, 10.0, 20.0, 30.0], index=range(6))
    sector = pd.Series(["a", "a", "a", "b", "b", "b"], index=range(6))
    out = cross_sectional_percentile(s, sector=sector, min_group=3)
    # Top of each sector should score alike despite very different raw values.
    assert out.loc[2] == pytest.approx(out.loc[5])


def test_tiny_groups_fall_back_to_the_whole_cross_section():
    s = pd.Series(np.arange(20, dtype=float))
    sector = pd.Series(["solo"] * 2 + ["big"] * 18)
    out = cross_sectional_percentile(s, sector=sector, min_group=10)
    assert out.notna().all()


def test_size_buckets_split_the_cross_section():
    caps = pd.Series(np.exp(np.linspace(10, 20, 100)))
    buckets = size_buckets(caps, n=5)
    assert buckets.nunique() == 5


def test_own_history_z_scores_against_the_name_itself():
    idx = pd.date_range("2015-01-01", periods=400, freq="B")
    history = pd.DataFrame({1: np.r_[np.zeros(399), 10.0]}, index=idx)
    z = own_history_z(history, min_periods=252)
    assert z.loc[1] > 3.0


def test_own_history_z_requires_enough_history():
    idx = pd.date_range("2015-01-01", periods=50, freq="B")
    history = pd.DataFrame({1: np.arange(50, dtype=float)}, index=idx)
    assert own_history_z(history, min_periods=252).isna().all()


def test_dual_baseline_blends_both_legs():
    idx = pd.date_range("2015-01-01", periods=400, freq="B")
    rng = np.random.default_rng(2)
    history = pd.DataFrame(rng.normal(0, 1, size=(400, 5)), index=idx, columns=range(5))
    signal = history.iloc[-1]

    both = dual_baseline(signal, history=history)
    cross_only = dual_baseline(signal, history=None)
    assert not np.allclose(both.dropna().to_numpy(), cross_only.dropna().to_numpy())


def test_dual_baseline_falls_back_when_history_is_absent():
    signal = pd.Series([1.0, 2.0, 3.0])
    assert dual_baseline(signal, history=None).notna().all()


def test_materiality_floors_apply_to_raw_units():
    """Threshold on the normalized statistic, floor on the raw value — two scales,
    deliberately not blended."""
    raw = pd.Series([0.001, 5.0, -8.0])
    normalized = pd.Series([0.9, 0.5, -0.7])
    out = apply_materiality(normalized, raw, {"min_abs": 1.0})
    assert pd.isna(out.iloc[0]), "a statistically extreme but trivially small reading fired"
    assert out.iloc[1] == 0.5
    assert out.iloc[2] == -0.7


def test_materiality_bounds_clip_absurd_ratios():
    raw = pd.Series([0.05, 500.0])
    normalized = pd.Series([0.3, 0.99])
    out = apply_materiality(normalized, raw, {"max_value": 10.0})
    assert out.iloc[0] == 0.3
    assert pd.isna(out.iloc[1])


# -- base class -----------------------------------------------------------


def test_base_signal_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseSignal()  # type: ignore[abstract]
