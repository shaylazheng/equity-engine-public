"""Tests for the fundamental signal library.

Every signal here returns a number for almost any input, so the tests check the
properties a *wrong* implementation would violate: that lags are fiscal quarters
rather than rows, that a non-positive scaler is treated as unknown rather than as
a large negative, and that each signal responds to its own input in the direction
its name claims.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from qe_signals.fundamentals import (
    FUNDAMENTAL_SPECS,
    compute_all,
    compute_fundamental,
)
from qe_signals.price import long_term_reversal, short_term_reversal

SPECS = {s.name: s for s in FUNDAMENTAL_SPECS}


def frame(n_quarters=12, permnos=(1, 2), **overrides):
    """A tidy quarterly panel with sane defaults, overridable per column."""
    rows = []
    for p in permnos:
        for q in range(n_quarters):
            rows.append({
                "permno": p,
                "datadate": pd.Timestamp("2010-03-31") + pd.DateOffset(months=3 * q),
                "atq": 1000.0, "ceqq": 400.0, "ibq": 50.0, "revtq": 800.0,
                "cheq": 100.0, "ltq": 600.0, "dlcq": 50.0, "actq": 300.0,
                "lctq": 150.0, "cshoq": 100.0,
            })
    f = pd.DataFrame(rows)
    for k, v in overrides.items():
        f[k] = v
    return f


class TestScaling:
    def test_negative_book_equity_is_unknown_not_worst_in_class(self):
        """The trap this guards: a profit over negative equity is a large
        negative ratio that sorts as the worst name in the cross-section, when
        the honest answer is that the ratio is meaningless for that firm."""
        f = frame()
        f.loc[f.index[:4], "ceqq"] = -400.0
        roe = compute_fundamental(f, SPECS["roe"])
        assert roe.iloc[:4].isna().all()
        assert roe.iloc[4:].notna().any()

    def test_zero_assets_does_not_produce_infinity(self):
        f = frame()
        f.loc[f.index[0], "atq"] = 0.0
        out = compute_all(f)
        assert np.isfinite(out.to_numpy(dtype=float)[~pd.isna(out.to_numpy())]).all()

    def test_index_is_preserved(self):
        """Results are assigned back alongside the frame; a reset index here
        would silently misalign every firm."""
        f = frame()
        f.index = pd.Index(range(100, 100 + len(f)))
        out = compute_all(f)
        assert out.index.equals(f.index)

    def test_unsorted_input_gives_the_same_answer(self):
        """The lag is taken on a sorted copy, so callers must not have to sort."""
        f = frame()
        shuffled = f.sample(frac=1.0, random_state=0)
        a = compute_all(f).sort_index()
        b = compute_all(shuffled).sort_index()
        pd.testing.assert_frame_equal(a, b)


class TestLagsAreFiscalQuarters:
    def test_year_over_year_reaches_four_quarters_back(self):
        f = frame(n_quarters=12, permnos=(1,))
        f["atq"] = [1000.0] * 4 + [1200.0] * 8   # +20% in quarter 5 vs quarter 1
        ag = compute_fundamental(f, SPECS["asset_growth"])
        assert ag.iloc[:4].isna().all(), "no four-quarter history yet"
        assert ag.iloc[4] == pytest.approx(0.20)

    def test_the_lag_does_not_cross_firms(self):
        """Two firms interleaved must not borrow each other's history."""
        f = frame(n_quarters=6, permnos=(1, 2))
        f.loc[f["permno"] == 1, "atq"] = 1000.0
        f.loc[f["permno"] == 2, "atq"] = 5000.0
        ag = compute_fundamental(f, SPECS["asset_growth"])
        assert ag.dropna().eq(0.0).all(), "growth should be zero within each firm"

    def test_a_quarterly_seasonal_is_not_read_as_growth(self):
        """The reason the lag is four and not one: most balance-sheet items are
        seasonal, and a quarter-over-quarter delta measures the season."""
        f = frame(n_quarters=12, permnos=(1,))
        f["revtq"] = [500.0, 800.0, 600.0, 1100.0] * 3   # a strong seasonal, no trend
        sg = compute_fundamental(f, SPECS["sales_growth"])
        assert sg.dropna().abs().max() == pytest.approx(0.0, abs=1e-12)


class TestDirection:
    @pytest.mark.parametrize(
        ("name", "col", "raise_it"),
        [
            ("roa", "ibq", True),
            ("roe", "ibq", True),
            ("asset_turnover", "revtq", True),
            ("cash_to_assets", "cheq", True),
            ("leverage", "ltq", True),
            ("current_ratio", "actq", True),
        ],
    )
    def test_signal_moves_with_its_own_input(self, name, col, raise_it):
        base = compute_fundamental(frame(), SPECS[name]).dropna()
        bumped = compute_fundamental(
            frame(**{col: 2.0 * frame()[col]}), SPECS[name]
        ).dropna()
        assert (bumped > base).all() if raise_it else (bumped < base).all()

    def test_issuance_is_positive_when_shares_grow(self):
        f = frame(n_quarters=8, permnos=(1,))
        f["cshoq"] = [100.0] * 4 + [110.0] * 4
        iss = compute_fundamental(f, SPECS["net_share_issuance"]).dropna()
        assert (iss > 0).all()

    def test_buybacks_read_negative(self):
        f = frame(n_quarters=8, permnos=(1,))
        f["cshoq"] = [100.0] * 4 + [90.0] * 4
        iss = compute_fundamental(f, SPECS["net_share_issuance"]).dropna()
        assert (iss < 0).all()

    def test_sue_is_undefined_when_earnings_never_move(self):
        """Not zero — *undefined*, and the distinction matters.

        SUE divides the surprise by the trailing volatility of surprises. With
        perfectly flat earnings that volatility is zero, so the ratio has no
        value; returning 0.0 would rank such a firm in the middle of the
        cross-section as though it had been measured. The first draft of this
        test asserted `max() == 0 or result is empty`, which passes either way
        and therefore asserted nothing.
        """
        sue = compute_fundamental(frame(n_quarters=16, permnos=(1,)), SPECS["sue"])
        assert sue.isna().all()

    def test_sue_is_positive_on_a_genuine_surprise(self):
        f = frame(n_quarters=16, permnos=(1,))
        ib = [50.0] * 12 + [80.0] * 4
        f["ibq"] = [v + 0.5 * (i % 3) for i, v in enumerate(ib)]  # some variation to scale by
        sue = compute_fundamental(f, SPECS["sue"]).dropna()
        assert sue.iloc[-1] > 0


class TestSpecsThemselves:
    def test_every_spec_has_a_sign_and_a_citation(self):
        for s in FUNDAMENTAL_SPECS:
            assert s.expected_sign in (-1, 1), s.name
            assert s.citation and "(" in s.citation, s.name
            assert s.rationale, s.name

    def test_names_are_unique(self):
        names = [s.name for s in FUNDAMENTAL_SPECS]
        assert len(names) == len(set(names))

    def test_compute_all_returns_one_column_per_spec(self):
        out = compute_all(frame())
        assert list(out.columns) == [s.name for s in FUNDAMENTAL_SPECS]


class TestReversal:
    def test_short_term_reversal_negates_last_month(self):
        m = pd.DataFrame({"ret": [0.10, -0.05]})
        out = short_term_reversal(m)
        assert out.tolist() == pytest.approx([-0.10, 0.05])

    def test_long_term_reversal_skips_the_momentum_window(self):
        """The window stops at 13 months so it cannot overlap 12-1 momentum.

        Overlapping would measure the difference between two opposite-signed
        effects rather than either one.
        """
        n = 72
        m = pd.DataFrame({
            "permno": [1] * n,
            "ym": pd.period_range("2010-01", periods=n, freq="M"),
            "ret": [0.0] * n,
        })
        # A single +50% month 6 months ago must NOT enter the 60..13 window.
        m.loc[n - 6, "ret"] = 0.5
        out = long_term_reversal(m).dropna()
        assert out.iloc[-1] == pytest.approx(0.0, abs=1e-12)

    def test_long_term_reversal_captures_the_distant_past(self):
        n = 72
        m = pd.DataFrame({
            "permno": [1] * n,
            "ym": pd.period_range("2010-01", periods=n, freq="M"),
            "ret": [0.0] * n,
        })
        m.loc[n - 30, "ret"] = 0.5          # inside 60..13
        out = long_term_reversal(m).dropna()
        assert out.iloc[-1] == pytest.approx(-0.5, abs=1e-9)

    def test_long_term_reversal_needs_a_full_history(self):
        n = 40
        m = pd.DataFrame({
            "permno": [1] * n,
            "ym": pd.period_range("2010-01", periods=n, freq="M"),
            "ret": [0.01] * n,
        })
        assert long_term_reversal(m).isna().all()
