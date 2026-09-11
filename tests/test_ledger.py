"""The score must never exist apart from its constituents."""

from __future__ import annotations

import pandas as pd
import pytest
from qe_core.ledger import AttributionLedger, FlagLeakError, LedgerError

D1 = pd.Timestamp("2020-06-30")
D2 = pd.Timestamp("2020-07-01")


def _row(permno, signal, contribution, *, family="quality", tier="core", date=D1, weight=0.25):
    return {
        "permno": permno,
        "date": date,
        "signal": signal,
        "family": family,
        "tier": tier,
        "weight": weight,
        "z_value": contribution / weight if weight else 0.0,
        "contribution": contribution,
    }


def _ledger(rows):
    return AttributionLedger(pd.DataFrame(rows))


@pytest.fixture
def led():
    return _ledger(
        [
            _row(1, "insider_cluster_buy", 1.4, family="insider"),
            _row(1, "gross_profitability", 0.9, family="quality"),
            _row(1, "momentum_12_1", -0.2, family="momentum", tier="exploratory"),
            _row(2, "insider_cluster_buy", 0.3, family="insider"),
            _row(2, "gross_profitability", 0.1, family="quality"),
            _row(2, "momentum_12_1", 0.6, family="momentum", tier="exploratory"),
        ]
    )


# -- the invariant --------------------------------------------------------


def test_total_is_the_sum_of_constituents(led):
    totals = led.totals()
    assert totals.loc[(1, D1)] == pytest.approx(2.1)
    assert totals.loc[(2, D1)] == pytest.approx(1.0)


def test_reconcile_accepts_agreement(led):
    led.reconcile(led.totals())


def test_reconcile_rejects_disagreement(led):
    tampered = led.totals().copy()
    tampered.loc[(1, D1)] += 0.01
    with pytest.raises(LedgerError, match="does not reconcile"):
        led.reconcile(tampered)


def test_reconcile_rejects_missing_keys(led):
    with pytest.raises(LedgerError, match="only one of"):
        led.reconcile(led.totals().drop(index=(2, D1)))


def test_reconcile_reports_the_worst_offender(led):
    tampered = led.totals().copy()
    tampered.loc[(2, D1)] += 0.5
    with pytest.raises(LedgerError) as exc:
        led.reconcile(tampered)
    assert "0.5" in str(exc.value)


# -- the views ------------------------------------------------------------


def test_by_signal_explains_one_name(led):
    detail = led.by_signal(1, D1)
    assert list(detail["signal"]) == [
        "insider_cluster_buy",
        "gross_profitability",
        "momentum_12_1",
    ]
    assert detail["contribution"].sum() == pytest.approx(2.1)


def test_by_family_and_by_tier_both_reconcile(led):
    fam = led.by_family().groupby(level=["permno", "date"]).sum()
    pd.testing.assert_series_equal(fam, led.totals(), check_names=False)

    tier = led.by_tier().groupby(level=["permno", "date"]).sum()
    pd.testing.assert_series_equal(tier, led.totals(), check_names=False)

    by_tier = led.by_tier()
    assert by_tier.loc[(1, D1, "exploratory")] == pytest.approx(-0.2)
    assert by_tier.loc[(1, D1, "core")] == pytest.approx(2.3)


def test_multiple_dates_are_kept_separate():
    led = _ledger([_row(1, "accruals", 1.0), _row(1, "accruals", 2.0, date=D2)])
    totals = led.totals()
    assert totals.loc[(1, D1)] == 1.0
    assert totals.loc[(1, D2)] == 2.0


# -- flags must never reach the composite ---------------------------------


def test_flag_tier_in_the_ledger_is_rejected():
    with pytest.raises(FlagLeakError, match="non-scoring tier"):
        _ledger([_row(1, "restatement_8k", 1.0, family="event", tier="flag")])


def test_risk_tier_in_the_ledger_is_rejected():
    """A risk exposure is already priced; it informs neutralization, not the score."""
    with pytest.raises(FlagLeakError, match="non-scoring tier"):
        _ledger([_row(1, "beta", 0.5, family="risk", tier="risk")])


def test_assert_no_flags_catches_a_renamed_flag(led):
    """Tier is the structural gate; this catches a flag that slipped in as 'core'."""
    led.assert_no_flags({"restatement_8k"})
    with pytest.raises(FlagLeakError, match="momentum_12_1"):
        led.assert_no_flags({"momentum_12_1"})


# -- malformed ledgers ----------------------------------------------------


def test_duplicate_signal_rows_are_rejected():
    with pytest.raises(LedgerError, match="duplicate"):
        _ledger([_row(1, "accruals", 1.0), _row(1, "accruals", 2.0)])


def test_null_contribution_is_rejected():
    rows = [_row(1, "accruals", 1.0)]
    df = pd.DataFrame(rows)
    df.loc[0, "contribution"] = None
    with pytest.raises(LedgerError, match="nulls"):
        AttributionLedger(df)


def test_missing_column_is_rejected():
    df = pd.DataFrame([_row(1, "accruals", 1.0)]).drop(columns=["family"])
    with pytest.raises(LedgerError, match="missing required column"):
        AttributionLedger(df)
