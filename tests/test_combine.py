"""Signal combination — the additivity guarantee above all.

Exact decomposition is the hard requirement that made the linear composite the
default over a gradient-boosted alternative. If `total == sum(contribution)` can
break, that decision bought nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from qe_combine.composite import combine, orthogonalize_signals, shrink_weights
from qe_core.ledger import FlagLeakError

DATE = pd.Timestamp("2020-06-30")

META = {
    "momentum_12_1": {"family": "momentum", "tier": "core"},
    "earnings_yield": {"family": "value", "tier": "core"},
    "accruals": {"family": "quality", "tier": "exploratory"},
}


@pytest.fixture
def signals():
    rng = np.random.default_rng(7)
    permnos = pd.Index(range(10_001, 10_051), name="permno")
    return {
        name: pd.Series(rng.normal(0, 1, len(permnos)), index=permnos, name=name)
        for name in META
    }


# -- the invariant --------------------------------------------------------


def test_totals_equal_the_sum_of_contributions(signals):
    ledger = combine(signals, META, DATE)
    by_hand = ledger.frame.groupby(["permno", "date"])["contribution"].sum()
    pd.testing.assert_series_equal(ledger.totals(), by_hand, check_names=False)


def test_the_ledger_reconciles_with_itself(signals):
    ledger = combine(signals, META, DATE)
    ledger.reconcile(ledger.totals())  # must not raise


def test_additivity_holds_to_floating_point(signals):
    """Not 'close enough' — the linear composite should be exact."""
    ledger = combine(signals, META, DATE)
    totals = ledger.totals()
    for (permno, date), total in totals.items():
        parts = ledger.by_signal(int(permno), date)["contribution"].sum()
        assert abs(total - parts) < 1e-12


def test_every_signal_contributes_a_row(signals):
    ledger = combine(signals, META, DATE, orthogonalize=False)
    assert set(ledger.frame["signal"].unique()) == set(META)


def test_contribution_is_weight_times_z(signals):
    ledger = combine(signals, META, DATE, orthogonalize=False)
    f = ledger.frame
    assert np.allclose(f["contribution"], f["weight"] * f["z_value"])


# -- flags must never reach a score --------------------------------------


def test_a_flag_tier_signal_is_rejected(signals):
    meta = {**META, "restatement_8k": {"family": "event", "tier": "flag"}}
    signals = {**signals, "restatement_8k": signals["accruals"].rename("restatement_8k")}
    with pytest.raises(FlagLeakError, match="non-scoring tier"):
        combine(signals, meta, DATE)


def test_a_risk_tier_signal_is_rejected(signals):
    meta = {**META, "beta": {"family": "risk", "tier": "risk"}}
    signals = {**signals, "beta": signals["accruals"].rename("beta")}
    with pytest.raises(FlagLeakError, match="non-scoring tier"):
        combine(signals, meta, DATE)


def test_missing_metadata_fails_loudly(signals):
    with pytest.raises(KeyError, match="no metadata"):
        combine(signals, {"momentum_12_1": META["momentum_12_1"]}, DATE)


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="no signals"):
        combine({}, META, DATE)


def test_all_empty_signals_is_rejected():
    empty = {"momentum_12_1": pd.Series(dtype=float)}
    with pytest.raises(ValueError, match="every signal was empty"):
        combine(empty, META, DATE)


# -- weight shrinkage -----------------------------------------------------


def test_full_shrinkage_is_exactly_equal_weight():
    out = shrink_weights({"a": 10.0, "b": 1.0, "c": 0.1}, shrinkage=1.0)
    assert list(out.values()) == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_no_shrinkage_preserves_the_ordering():
    out = shrink_weights({"a": 10.0, "b": 1.0, "c": 0.1}, shrinkage=0.0)
    assert out["a"] > out["b"] > out["c"]
    assert sum(out.values()) == pytest.approx(1.0)


def test_partial_shrinkage_moves_toward_equal_weight():
    raw = shrink_weights({"a": 10.0, "b": 1.0}, shrinkage=0.0)
    half = shrink_weights({"a": 10.0, "b": 1.0}, shrinkage=0.5)
    assert abs(half["a"] - 0.5) < abs(raw["a"] - 0.5)


def test_zero_weights_fall_back_to_equal():
    out = shrink_weights({"a": 0.0, "b": 0.0}, shrinkage=0.0)
    assert out == {"a": 0.5, "b": 0.5}


def test_shrinkage_bounds_are_enforced():
    with pytest.raises(ValueError, match="shrinkage"):
        shrink_weights({"a": 1.0}, shrinkage=1.5)


def test_weights_reach_the_ledger(signals):
    ledger = combine(signals, META, DATE, weights={"momentum_12_1": 3.0,
                                                   "earnings_yield": 1.0,
                                                   "accruals": 1.0}, shrinkage=0.0)
    w = ledger.frame.groupby("signal")["weight"].first()
    assert w["momentum_12_1"] > w["earnings_yield"]


# -- orthogonalization ----------------------------------------------------


def test_the_first_signal_is_left_untouched():
    rng = np.random.default_rng(3)
    df = pd.DataFrame(rng.normal(0, 1, size=(80, 3)), columns=["a", "b", "c"])
    out = orthogonalize_signals(df)
    pd.testing.assert_series_equal(out["a"], df["a"], check_dtype=False)


def test_later_signals_become_orthogonal_to_earlier_ones():
    rng = np.random.default_rng(4)
    a = pd.Series(rng.normal(0, 1, 200))
    b = 0.8 * a + 0.2 * pd.Series(rng.normal(0, 1, 200))
    df = pd.DataFrame({"a": a, "b": b})

    assert abs(np.corrcoef(df["a"], df["b"])[0, 1]) > 0.8
    out = orthogonalize_signals(df)
    assert abs(np.corrcoef(out["a"], out["b"])[0, 1]) < 1e-10


def test_order_changes_the_result():
    """Ordering is a real choice — whatever comes first keeps its full variance."""
    rng = np.random.default_rng(5)
    a = pd.Series(rng.normal(0, 1, 200))
    b = 0.7 * a + 0.3 * pd.Series(rng.normal(0, 1, 200))
    df = pd.DataFrame({"a": a, "b": b})

    ab = orthogonalize_signals(df, order=["a", "b"])
    ba = orthogonalize_signals(df, order=["b", "a"])
    assert not np.allclose(ab["a"].to_numpy(), ba["a"].to_numpy())


def test_unknown_signal_in_the_order_fails():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(KeyError, match="unknown signals"):
        orthogonalize_signals(df, order=["a", "nope"])


def test_orthogonalization_changes_the_combined_score(signals):
    plain = combine(signals, META, DATE, orthogonalize=False).totals()
    ortho = combine(signals, META, DATE, orthogonalize=True).totals()
    assert not np.allclose(plain.to_numpy(), ortho.to_numpy())


# -- the views the ledger exists for --------------------------------------


def test_family_and_tier_views_both_reconcile(signals):
    ledger = combine(signals, META, DATE)
    fam = ledger.by_family().groupby(level=["permno", "date"]).sum()
    tier = ledger.by_tier().groupby(level=["permno", "date"]).sum()
    pd.testing.assert_series_equal(fam, ledger.totals(), check_names=False)
    pd.testing.assert_series_equal(tier, ledger.totals(), check_names=False)


def test_tier_view_separates_core_from_exploratory(signals):
    ledger = combine(signals, META, DATE)
    tiers = set(ledger.frame["tier"].unique())
    assert tiers == {"core", "exploratory"}
