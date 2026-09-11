"""Tests for the long-only portfolio optimizer.

An optimizer is unusually easy to get confidently wrong: it returns a plausible
vector of weights whatever it does. So these tests check the properties that
would be *violated* by a wrong answer -- the constraints exactly, the direction
of each term's effect, and agreement with a brute-force optimum on a problem
small enough to solve by search.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from qe_backtest.optimize import (
    AlphaScaleError,
    OptimizerConfig,
    RiskInputs,
    optimize_weights,
    scores_to_alpha,
)


def make_risk(names: pd.Index, *, n_factors: int = 3, seed: int = 0) -> RiskInputs:
    rng = np.random.default_rng(seed)
    factors = [f"f{i}" for i in range(n_factors)]
    X = pd.DataFrame(
        rng.standard_normal((len(names), n_factors)), index=names, columns=factors
    )
    A = rng.standard_normal((n_factors, n_factors))
    F = pd.DataFrame(A @ A.T / n_factors * 1e-4, index=factors, columns=factors)
    d = pd.Series(rng.uniform(1e-4, 4e-4, len(names)), index=names)
    return RiskInputs(exposures=X, factor_cov=F, specific_var=d)


@pytest.fixture
def problem() -> tuple[pd.Series, RiskInputs]:
    names = pd.Index([f"n{i}" for i in range(40)], name="permno")
    rng = np.random.default_rng(11)
    alpha = pd.Series(rng.normal(0.0, 0.01, len(names)), index=names)
    return alpha, make_risk(names)


class TestConstraints:
    def test_weights_sum_to_one(self, problem) -> None:
        alpha, risk = problem
        res = optimize_weights(alpha, risk)
        assert res.weights.sum() == pytest.approx(1.0, abs=1e-8)

    def test_long_only(self, problem) -> None:
        alpha, risk = problem
        res = optimize_weights(alpha, risk)
        assert (res.weights >= 0).all()

    def test_position_cap_is_respected(self, problem) -> None:
        alpha, risk = problem
        cfg = OptimizerConfig(max_weight=0.05, max_active_share=None)
        res = optimize_weights(alpha, risk, config=cfg)
        assert res.weights.max() <= 0.05 + 1e-9

    def test_cap_binds_when_alpha_is_concentrated(self, problem) -> None:
        """A cap that never binds is not being tested."""
        _, risk = problem
        names = risk.exposures.index
        alpha = pd.Series(0.0, index=names)
        alpha.iloc[:3] = 1.0  # three names dominate
        cfg = OptimizerConfig(max_weight=0.10, max_active_share=None, trade_cost=0.0)
        res = optimize_weights(alpha, risk, config=cfg)
        assert res.weights.max() == pytest.approx(0.10, abs=1e-6)

    def test_infeasible_cap_is_widened_not_silently_unfunded(self, problem) -> None:
        # 40 names capped at 1% cannot sum to 1. Returning a 40%-invested book
        # would be a silent, expensive wrong answer.
        alpha, risk = problem
        cfg = OptimizerConfig(max_weight=0.01, universe_size=40, max_active_share=None)
        res = optimize_weights(alpha, risk, config=cfg)
        assert res.weights.sum() == pytest.approx(1.0, abs=1e-8)


class TestObjective:
    def test_matches_brute_force_on_a_tiny_problem(self) -> None:
        """The real check: agreement with an independently-computed optimum.

        Three names, no cost, no active-share limit -- small enough to solve by
        dense grid search over the simplex, which shares no code with the solver.
        """
        names = pd.Index(["a", "b", "c"])
        alpha = pd.Series([0.02, 0.01, 0.005], index=names)
        F = pd.DataFrame([[4e-4]], index=["f"], columns=["f"])
        X = pd.DataFrame([[1.0], [0.5], [0.0]], index=names, columns=["f"])
        d = pd.Series([2e-4, 3e-4, 1e-4], index=names)
        risk = RiskInputs(exposures=X, factor_cov=F, specific_var=d)
        cfg = OptimizerConfig(
            risk_aversion=20.0, max_weight=1.0, trade_cost=0.0, max_active_share=None
        )

        res = optimize_weights(alpha, risk, config=cfg)

        S = X.to_numpy() @ F.to_numpy() @ X.to_numpy().T + np.diag(d.to_numpy())
        a = alpha.to_numpy()

        def obj(w: np.ndarray) -> float:
            return float(a @ w - 0.5 * cfg.risk_aversion * w @ S @ w)

        grid = np.linspace(0, 1, 401)
        best_w, best = None, -np.inf
        for wa in grid:
            for wb in grid:
                if wa + wb > 1.0 + 1e-12:
                    continue
                w = np.array([wa, wb, 1.0 - wa - wb])
                v = obj(w)
                if v > best:
                    best, best_w = v, w

        got = res.weights.reindex(names).fillna(0.0).to_numpy()
        assert obj(got) >= best - 1e-6
        assert np.allclose(got, best_w, atol=3e-3)

    def test_higher_risk_aversion_diversifies(self, problem) -> None:
        alpha, risk = problem
        low = optimize_weights(
            alpha, risk, config=OptimizerConfig(risk_aversion=1.0, max_active_share=None)
        )
        high = optimize_weights(
            alpha, risk, config=OptimizerConfig(risk_aversion=200.0, max_active_share=None)
        )
        # Herfindahl falls as risk aversion rises.
        assert (high.weights**2).sum() < (low.weights**2).sum()
        assert high.total_var < low.total_var

    def test_alpha_ordering_is_respected_without_risk(self) -> None:
        """With identical risk, higher alpha must not get less weight."""
        names = pd.Index([f"n{i}" for i in range(10)])
        alpha = pd.Series(np.linspace(0.001, 0.02, 10), index=names)
        X = pd.DataFrame(0.0, index=names, columns=["f"])
        F = pd.DataFrame([[1e-8]], index=["f"], columns=["f"])
        d = pd.Series(2e-4, index=names)
        risk = RiskInputs(exposures=X, factor_cov=F, specific_var=d)
        res = optimize_weights(
            alpha,
            risk,
            config=OptimizerConfig(
                max_weight=1.0, trade_cost=0.0, max_active_share=None, risk_aversion=5.0
            ),
        )
        w = res.weights.reindex(names).fillna(0.0)
        assert (w.diff().dropna() >= -1e-9).all()

    def test_risk_model_changes_the_answer(self) -> None:
        """If two names have equal alpha, the riskier one must get less.

        This is the whole reason the risk model is wired in; an optimizer that
        ignored `S` would give them equal weight.
        """
        names = pd.Index(["safe", "risky"])
        alpha = pd.Series([0.01, 0.01], index=names)
        X = pd.DataFrame(0.0, index=names, columns=["f"])
        F = pd.DataFrame([[1e-8]], index=["f"], columns=["f"])
        d = pd.Series([1e-4, 9e-4], index=names)
        risk = RiskInputs(exposures=X, factor_cov=F, specific_var=d)
        res = optimize_weights(
            alpha,
            risk,
            config=OptimizerConfig(
                max_weight=1.0, trade_cost=0.0, max_active_share=None, risk_aversion=50.0
            ),
        )
        assert res.weights["safe"] > res.weights["risky"] + 0.05


class TestTradingCost:
    def test_cost_reduces_turnover(self, problem) -> None:
        alpha, risk = problem
        names = risk.exposures.index
        current = pd.Series(1.0 / 20, index=names[:20])

        free = optimize_weights(
            alpha, risk, current=current,
            config=OptimizerConfig(trade_cost=0.0, max_active_share=None),
        )
        pricey = optimize_weights(
            alpha, risk, current=current,
            config=OptimizerConfig(trade_cost=0.05, max_active_share=None),
        )
        assert pricey.turnover < free.turnover

    def test_large_cost_pins_the_book(self, problem) -> None:
        """At a prohibitive cost the optimum is to do nothing."""
        alpha, risk = problem
        names = risk.exposures.index
        current = pd.Series(1.0 / 40, index=names)
        res = optimize_weights(
            alpha, risk, current=current,
            config=OptimizerConfig(trade_cost=10.0, max_active_share=None),
        )
        assert res.turnover < 1e-6
        assert res.weights.round(6).nunique() == 1

    def test_held_names_stay_in_the_universe(self, problem) -> None:
        """A holding outside the top-N by alpha must still be priced, not dumped.

        If it were dropped from the universe the sale would happen without ever
        being weighed against the cost of making it.
        """
        alpha, risk = problem
        laggard = alpha.nsmallest(1).index[0]
        current = pd.Series({laggard: 1.0})
        res = optimize_weights(
            alpha, risk, current=current,
            config=OptimizerConfig(universe_size=5, trade_cost=10.0,
                                   max_active_share=None, max_weight=1.0),
        )
        assert laggard in res.weights.index
        assert res.weights[laggard] > 0.9


class TestActiveShare:
    def test_limit_is_enforced(self, problem) -> None:
        alpha, risk = problem
        cfg = OptimizerConfig(max_active_share=0.20, risk_aversion=0.1, max_weight=1.0)
        res = optimize_weights(alpha, risk, config=cfg)
        w = res.weights.reindex(risk.exposures.index).fillna(0.0)
        n = len(w)
        active = float(np.abs(w - 1.0 / n).sum() / 2.0)
        assert active <= 0.20 + 1e-8

    def test_limit_preserves_budget_and_box(self, problem) -> None:
        alpha, risk = problem
        cfg = OptimizerConfig(max_active_share=0.10, max_weight=0.10, risk_aversion=0.1)
        res = optimize_weights(alpha, risk, config=cfg)
        w = res.weights.reindex(risk.exposures.index).fillna(0.0)
        assert w.sum() == pytest.approx(1.0, abs=1e-8)
        assert (w >= -1e-12).all()
        assert w.max() <= 0.10 + 1e-9

    def test_disabled_limit_permits_concentration(self, problem) -> None:
        alpha, risk = problem
        loose = optimize_weights(
            alpha, risk,
            config=OptimizerConfig(max_active_share=None, risk_aversion=0.1, max_weight=1.0),
        )
        tight = optimize_weights(
            alpha, risk,
            config=OptimizerConfig(max_active_share=0.10, risk_aversion=0.1, max_weight=1.0),
        )
        assert (loose.weights**2).sum() > (tight.weights**2).sum()


class TestRiskInputs:
    def test_unknown_factor_raises(self) -> None:
        names = pd.Index(["a", "b"])
        X = pd.DataFrame(1.0, index=names, columns=["f", "ghost"])
        F = pd.DataFrame([[1e-4]], index=["f"], columns=["f"])
        with pytest.raises(ValueError, match="unknown factors"):
            RiskInputs(X, F, pd.Series(1e-4, index=names))

    def test_missing_name_gets_median_specific_risk_not_zero(self) -> None:
        """Zero specific variance would tell the optimizer a name is riskless.

        That is the most dangerous possible default: the optimizer would pour
        the whole book into whichever name it knows least about.
        """
        names = pd.Index(["a", "b", "unknown"])
        X = pd.DataFrame(0.0, index=names[:2], columns=["f"])
        F = pd.DataFrame([[1e-8]], index=["f"], columns=["f"])
        d = pd.Series([1e-4, 3e-4], index=names[:2])
        risk = RiskInputs(X, F, d)
        _, _, aligned = risk.align(names)
        assert aligned[2] == pytest.approx(np.median([1e-4, 3e-4]))
        assert aligned[2] > 0

    def test_decomposition_shares_sum_to_one(self, problem) -> None:
        alpha, risk = problem
        res = optimize_weights(alpha, risk)
        assert res.risk_decomposition().sum() == pytest.approx(1.0)


class TestDegenerate:
    def test_empty_alpha(self) -> None:
        names = pd.Index(["a", "b"])
        risk = make_risk(names)
        res = optimize_weights(pd.Series(dtype=float), risk)
        assert res.weights.empty
        assert res.converged

    def test_all_nan_alpha(self) -> None:
        names = pd.Index(["a", "b"])
        risk = make_risk(names)
        res = optimize_weights(pd.Series(np.nan, index=names), risk)
        assert res.weights.empty

    def test_single_name(self) -> None:
        names = pd.Index(["a"])
        risk = make_risk(names, n_factors=1)
        res = optimize_weights(pd.Series([0.01], index=names), risk)
        assert res.weights.sum() == pytest.approx(1.0)

    def test_converges(self, problem) -> None:
        alpha, risk = problem
        res = optimize_weights(alpha, risk)
        assert res.converged, f"did not converge in {res.iterations} iterations"


class TestAlphaScale:
    """The guard against the failure mode that makes everything else pointless.

    Scores are z-scores or percentiles; the objective needs expected returns. At
    score scale the alpha term is ~1.0 against a risk term of ~1e-4, so risk
    aversion, the covariance, and the trading-cost term all become numerically
    irrelevant and the optimizer degenerates into "buy the top-ranked name" --
    while still returning weights that look perfectly sensible.
    """

    def test_raw_scores_raise(self, problem) -> None:
        _, risk = problem
        names = risk.exposures.index
        ranks = pd.Series(np.arange(len(names), dtype=float), index=names)
        with pytest.raises(AlphaScaleError, match="scores rather than"):
            optimize_weights(ranks, risk)

    def test_return_scaled_alpha_is_accepted(self, problem) -> None:
        alpha, risk = problem
        res = optimize_weights(alpha, risk)  # sd ~1%, plausible
        assert res.weights.sum() == pytest.approx(1.0, abs=1e-8)

    def test_a_single_name_is_not_judged(self, problem) -> None:
        # One observation has no cross-sectional SD to test.
        _, risk = problem
        one = risk.exposures.index[:1]
        res = optimize_weights(pd.Series([50.0], index=one), risk)
        assert res.weights.sum() == pytest.approx(1.0)

    def test_scalar_and_series_specific_var_agree(self) -> None:
        """Both arms take a *variance*. Treating one as a volatility would be a
        factor-of-100 error that still returns a plausible number."""
        names = pd.Index([f"n{i}" for i in range(20)])
        scores = pd.Series(np.linspace(-2, 2, 20), index=names)
        flat = scores_to_alpha(scores, ic=0.03, specific_var=4e-4)
        series = scores_to_alpha(
            scores, ic=0.03, specific_var=pd.Series(4e-4, index=names)
        )
        pd.testing.assert_series_equal(flat, series)

    def test_scores_to_alpha_lands_in_return_units(self) -> None:
        names = pd.Index([f"n{i}" for i in range(50)])
        rng = np.random.default_rng(2)
        scores = pd.Series(rng.normal(0, 3.0, len(names)), index=names)
        spec = pd.Series(4e-4, index=names)  # 2% specific vol
        a = scores_to_alpha(scores, ic=0.03, specific_var=spec)
        # alpha = ic * vol * z  ->  sd = 0.03 * 0.02 = 6bp, whatever the score scale
        assert a.std(ddof=1) == pytest.approx(0.03 * 0.02, rel=0.02)
        assert optimize_weights(a, make_risk(names)).weights.sum() == pytest.approx(1.0)

    def test_scaling_preserves_ordering(self) -> None:
        names = pd.Index([f"n{i}" for i in range(10)])
        scores = pd.Series(np.arange(10, dtype=float), index=names)
        a = scores_to_alpha(scores, ic=0.05, specific_var=6.25e-6)
        assert a.is_monotonic_increasing

    def test_higher_ic_means_more_aggressive_positions(self) -> None:
        """IC is the honest place to express confidence, not risk aversion."""
        names = pd.Index([f"n{i}" for i in range(30)])
        rng = np.random.default_rng(4)
        scores = pd.Series(rng.normal(0, 1, len(names)), index=names)
        risk = make_risk(names)
        spec = risk.specific_var
        cfg = OptimizerConfig(max_active_share=None, max_weight=1.0, trade_cost=0.0)
        timid = optimize_weights(
            scores_to_alpha(scores, ic=0.01, specific_var=spec), risk, config=cfg
        )
        bold = optimize_weights(
            scores_to_alpha(scores, ic=0.10, specific_var=spec), risk, config=cfg
        )
        assert (bold.weights**2).sum() > (timid.weights**2).sum()


class TestNoBuyBounds:
    def test_per_name_cap_blocks_adding(self, problem) -> None:
        alpha, risk = problem
        names = risk.exposures.index
        best = alpha.idxmax()
        current = pd.Series(0.02, index=names)
        caps = pd.Series(np.nan, index=names)
        caps[best] = 0.02  # may hold, may not add
        res = optimize_weights(
            alpha, risk, current=current, max_weights=caps,
            config=OptimizerConfig(max_active_share=None, trade_cost=0.0),
        )
        assert res.weights.get(best, 0.0) <= 0.02 + 1e-9

    def test_without_the_cap_the_same_name_is_bought(self, problem) -> None:
        """The contrast that makes the previous test mean something."""
        alpha, risk = problem
        names = risk.exposures.index
        best = alpha.idxmax()
        current = pd.Series(0.02, index=names)
        res = optimize_weights(
            alpha, risk, current=current,
            config=OptimizerConfig(max_active_share=None, trade_cost=0.0),
        )
        assert res.weights.get(best, 0.0) > 0.02 + 1e-6

    def test_budget_still_binds_with_per_name_caps(self, problem) -> None:
        alpha, risk = problem
        names = risk.exposures.index
        caps = pd.Series(0.03, index=names)
        res = optimize_weights(
            alpha, risk, max_weights=caps,
            config=OptimizerConfig(max_active_share=None),
        )
        assert res.weights.sum() == pytest.approx(1.0, abs=1e-8)
        assert res.weights.max() <= 0.03 + 1e-9
