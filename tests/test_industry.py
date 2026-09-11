"""Tests for FF12 classification and the sum-to-zero industry constraint.

The bugs these guard against are the ones that do not raise: a SIC boundary off
by one puts a bank in `Manuf` and the fit still runs; an unconstrained industry
block quietly redefines what the market factor means.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from qe_risk.industry import (
    FF12,
    INDUSTRIES,
    constrain_industries,
    ff12_from_sic,
    industry_coverage,
    industry_dummies_ff12,
)


class TestClassification:
    @pytest.mark.parametrize(
        ("sic", "expected"),
        [
            (2011, "NoDur"),   # meat packing
            (1311, "Enrgy"),   # crude petroleum
            (2834, "Hlth"),    # pharmaceutical preparations
            (3674, "BusEq"),   # semiconductors
            (4813, "Telcm"),   # telephone communications
            (4911, "Utils"),   # electric services
            (5411, "Shops"),   # grocery stores
            (6021, "Money"),   # national commercial banks
            (3312, "Manuf"),   # blast furnaces and steel
            (2510, "Durbl"),   # household furniture
            (2821, "Chems"),   # plastics materials
            (7011, "Other"),   # hotels — genuinely unclassified in FF12
        ],
    )
    def test_known_codes(self, sic: int, expected: str) -> None:
        assert ff12_from_sic(pd.Series([sic])).iloc[0] == expected

    def test_boundaries_are_inclusive(self) -> None:
        # A half-open range would silently misfile every firm on the edge.
        for _label, ranges in FF12:
            for lo, hi in ranges:
                got = ff12_from_sic(pd.Series([lo, hi]))
                assert (got != "Other").all(), f"{lo}-{hi} endpoints fell through"

    def test_ranges_do_not_overlap(self) -> None:
        # First-match-wins hides overlaps rather than reporting them, so a later
        # definition silently loses names. Check the definitions directly.
        seen: dict[int, str] = {}
        for label, ranges in FF12:
            for lo, hi in ranges:
                for code in range(lo, hi + 1):
                    assert code not in seen, (
                        f"SIC {code} claimed by both {seen.get(code)} and {label}"
                    )
                    seen[code] = label

    def test_missing_and_junk_become_other(self) -> None:
        got = ff12_from_sic(pd.Series([np.nan, None, "not a code", 0]))
        assert (got == "Other").all()

    def test_index_is_preserved(self) -> None:
        # The result is concatenated against the panel by position-free
        # alignment; a reset index here would scramble every assignment.
        idx = pd.Index([7, 3, 99], name="row")
        got = ff12_from_sic(pd.Series([2011, 6021, 1311], index=idx))
        assert got.index.equals(idx)

    def test_coverage_shares_sum_to_one(self) -> None:
        cov = industry_coverage(ff12_from_sic(pd.Series([2011, 6021, 6022, 9999])))
        assert cov["share"].sum() == pytest.approx(1.0)
        assert cov.loc["Money", "names"] == 2


class TestDummies:
    def test_drop_leaves_eleven_and_avoids_singularity(self) -> None:
        sic = pd.Series([2011, 6021, 1311, 3674, 9999])
        d = industry_dummies_ff12(sic)
        assert d.shape[1] == len(INDUSTRIES) - 1
        assert "ind_Other" not in d.columns

    def test_undropped_dummies_sum_to_one(self) -> None:
        d = industry_dummies_ff12(pd.Series([2011, 6021, 9999]), drop=None)
        assert d.shape[1] == len(INDUSTRIES)
        assert d.sum(axis=1).eq(1.0).all()

    def test_undropped_block_is_singular_with_an_intercept(self) -> None:
        # This is *why* one of the two treatments is mandatory. Twelve dummies
        # plus a column of ones is exactly rank-deficient, and lstsq will happily
        # return a minimum-norm answer rather than complain.
        d = industry_dummies_ff12(pd.Series([2011, 6021, 1311, 3674, 9999]), drop=None)
        X = np.column_stack([np.ones(len(d)), d.to_numpy()])
        assert np.linalg.matrix_rank(X) < X.shape[1]

    def test_all_twelve_columns_exist_even_when_absent_from_data(self) -> None:
        # A month with no energy names must not silently change the factor list.
        d = industry_dummies_ff12(pd.Series([6021, 6022]), drop=None)
        assert list(d.columns) == [f"ind_{c}" for c in INDUSTRIES]
        assert d["ind_Enrgy"].eq(0.0).all()


class TestConstraint:
    #: One firm per FF12 industry, so a 12-column design is identified.
    SICS = (2011, 2510, 1311, 2821, 2834, 3674, 4813, 4911, 5411, 6021, 3312, 7011)

    @classmethod
    def _panel(cls) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        n = len(cls.SICS)
        sic = pd.Series(cls.SICS * 2)
        date = pd.Series(pd.to_datetime(["2020-01-31"] * n + ["2020-02-29"] * n))
        rng = np.random.default_rng(0)
        weight = pd.Series(
            np.concatenate([rng.uniform(5, 50, n), rng.uniform(5, 50, n)])
        )
        return industry_dummies_ff12(sic, drop=None), date, weight

    def test_constrained_block_is_full_rank_with_an_intercept(self) -> None:
        """The complement of `test_undropped_block_is_singular_with_an_intercept`.

        Twelve raw dummies plus an intercept is rank-deficient; eleven
        constrained columns plus an intercept is not. That is the whole point of
        the reparameterization, so both halves are asserted.
        """
        d, date, w = self._panel()
        c = constrain_industries(d, date, w, reference="Money")
        assert c.shape[1] == len(INDUSTRIES) - 1

        one = date.to_numpy() == date.iloc[0]
        X = np.column_stack([np.ones(one.sum()), c[one].to_numpy()])
        assert X.shape[0] >= X.shape[1], "need at least as many names as columns"
        assert np.linalg.matrix_rank(X) == X.shape[1]

    def test_market_factor_recovers_the_weighted_mean(self) -> None:
        """The point of the constraint, stated as a testable consequence.

        Build returns that are a pure industry effect around a known market
        level, with the industry effects constructed to sum to zero under the
        weights. The fitted intercept must recover the market level exactly. With
        a dropped level instead, it recovers the dropped industry's return.
        """
        d, date, w = self._panel()
        one_date = date == date.iloc[0]
        d, w = d[one_date].reset_index(drop=True), w[one_date].reset_index(drop=True)
        dt = date[one_date].reset_index(drop=True)

        shares = d.mul(w, axis=0).sum() / w.sum()
        effects = pd.Series(
            [0.10, -0.04, 0.02, -0.03, 0.05, 0.06, -0.02, -0.05, 0.03, 0.07, -0.01, 0.02],
            index=d.columns,
        )
        # Re-center so the weighted industry effects sum to zero, which is the
        # condition the constraint imposes.
        effects = effects - (effects * shares).sum()

        market = 0.012
        y = market + d.to_numpy() @ effects.to_numpy()

        c = constrain_industries(d, dt, w, reference="Money")
        X = np.column_stack([np.ones(len(c)), c.to_numpy()])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        assert coef[0] == pytest.approx(market, abs=1e-10)

    def test_dropping_a_level_does_not_recover_the_market(self) -> None:
        """The contrast that makes the previous test mean something."""
        d, date, w = self._panel()
        one = date == date.iloc[0]
        d, w = d[one].reset_index(drop=True), w[one].reset_index(drop=True)

        shares = d.mul(w, axis=0).sum() / w.sum()
        effects = pd.Series(
            [0.10, -0.04, 0.02, -0.03, 0.05, 0.06, -0.02, -0.05, 0.03, 0.07, -0.01, 0.02],
            index=d.columns,
        )
        effects = effects - (effects * shares).sum()
        market = 0.012
        y = market + d.to_numpy() @ effects.to_numpy()

        dropped = d.drop(columns="ind_Money")
        X = np.column_stack([np.ones(len(dropped)), dropped.to_numpy()])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        # It recovers Money's return, not the market — which is exactly the
        # misattribution the constraint exists to remove.
        assert coef[0] == pytest.approx(market + effects["ind_Money"], abs=1e-10)
        assert coef[0] != pytest.approx(market, abs=1e-4)

    def test_weights_are_recomputed_per_date(self) -> None:
        # Industry composition is not stable across time; pooling the weights
        # would apply 2020's economy to 1975's cross-section.
        d, date, w = self._panel()
        c = constrain_industries(d, date, w, reference="Money")
        jan = c[date.to_numpy() == date.iloc[0]]
        feb = c[date.to_numpy() == date.iloc[-1]]
        col = "ind_NoDur"
        assert not np.allclose(jan[col].to_numpy(), feb[col].to_numpy())

    def test_unknown_reference_raises(self) -> None:
        d, date, w = self._panel()
        with pytest.raises(KeyError, match="Nonsense"):
            constrain_industries(d, date, w, reference="Nonsense")

    def test_index_alignment_survives_a_non_range_index(self) -> None:
        d, date, w = self._panel()
        idx = pd.Index(range(100, 100 + len(d)))
        d.index, date.index, w.index = idx, idx, idx
        c = constrain_industries(d, date, w, reference="Money")
        assert c.index.equals(idx)
