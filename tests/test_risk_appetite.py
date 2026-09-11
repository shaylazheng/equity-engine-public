"""Risk-appetite profiles, and the two ways they could silently go wrong.

The dangerous failures here are not exceptions. They are (a) `balanced` drifting
away from the configuration `docs/real_backtest.md` was produced at, so every
recorded number quietly stops describing the code, and (b) a beta premium
applied on the wrong scale or in the wrong period units, which produces a
perfectly plausible book that is not the one anyone asked for. Both get a test
that fails loudly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from qe_backtest.engine import BacktestConfig
from qe_backtest.optimize import (
    ExposureScaleError,
    OptimizerConfig,
    RiskInputs,
    apply_beta_tilt,
    optimize_weights,
)
from qe_core.risk_appetite import PROFILES, RiskProfile, load_profile


@pytest.fixture
def problem():
    """A small cross-section with a real spread of betas."""
    rng = np.random.default_rng(7)
    n, k = 60, 3
    names = pd.Index([f"n{i}" for i in range(n)], name="permno")
    factors = ["market", "beta", "value"]
    X = pd.DataFrame(rng.standard_normal((n, k)), index=names, columns=factors)
    X["market"] = 1.0
    # Standardized, as qe_risk.neutralize.standardize would leave them.
    for c in ("beta", "value"):
        X[c] = (X[c] - X[c].mean()) / X[c].std(ddof=1)
    F = pd.DataFrame(np.diag([4e-4, 9e-4, 4e-4]), index=factors, columns=factors)
    spec = pd.Series(rng.uniform(2e-3, 8e-3, n), index=names)
    alpha = pd.Series(rng.standard_normal(n) * 0.004, index=names, name="alpha")
    return alpha, RiskInputs(exposures=X, factor_cov=F, specific_var=spec)


class TestShippedProfiles:
    def test_balanced_reproduces_the_documented_backtest(self) -> None:
        """`docs/real_backtest.md` was run at exactly these numbers.

        Changing any of them without re-running the backtest makes the document
        describe a configuration the code no longer has.
        """
        p = PROFILES["balanced"]
        assert p.risk_aversion == 8.0
        assert p.max_weight == 0.06
        assert p.max_active_share == 0.60
        assert p.universe_size == 200
        assert p.n_positions == 40
        assert p.trade_cost == 0.005
        assert p.score_ic == 0.03
        assert p.beta_premium_annual == 0.0

    def test_ladder_is_monotone_in_aggression(self) -> None:
        order = ["conservative", "balanced", "aggressive", "max_growth"]
        ps = [PROFILES[k] for k in order]
        assert [p.risk_aversion for p in ps] == sorted(
            [p.risk_aversion for p in ps], reverse=True
        )
        assert [p.max_weight for p in ps] == sorted([p.max_weight for p in ps])
        assert [p.max_active_share for p in ps] == sorted(
            [p.max_active_share for p in ps]
        )
        assert [p.n_positions for p in ps] == sorted(
            [p.n_positions for p in ps], reverse=True
        )
        assert [p.universe_size for p in ps] == sorted(
            [p.universe_size for p in ps], reverse=True
        )
        assert [p.beta_premium_annual for p in ps] == sorted(
            [p.beta_premium_annual for p in ps]
        )

    def test_beliefs_do_not_move_with_appetite(self) -> None:
        """IC and trading cost are claims about the world, not preferences.

        A profile that raised `score_ic` would be sizing positions on a better
        forecast than it has, which reads as confidence and behaves as leverage.
        """
        assert {p.score_ic for p in PROFILES.values()} == {0.03}
        assert {p.trade_cost for p in PROFILES.values()} == {0.005}

    def test_every_profile_can_be_fully_invested(self) -> None:
        for p in PROFILES.values():
            assert p.max_weight * p.n_positions >= 1.0


class TestValidation:
    def test_rejects_unfundable_book(self) -> None:
        with pytest.raises(ValueError, match="cannot be fully invested"):
            RiskProfile(
                name="x", risk_aversion=5.0, max_weight=0.02,
                max_active_share=0.8, universe_size=100, n_positions=20,
                beta_premium_annual=0.0,
            )

    def test_rejects_universe_smaller_than_the_book(self) -> None:
        with pytest.raises(ValueError, match="below n_positions"):
            RiskProfile(
                name="x", risk_aversion=5.0, max_weight=0.20,
                max_active_share=0.8, universe_size=10, n_positions=20,
                beta_premium_annual=0.0,
            )

    def test_rejects_a_beta_premium_that_is_a_units_error(self) -> None:
        """0.30 as a *monthly* figure would be a 3600%/yr view. It must not load."""
        with pytest.raises(ValueError, match="units error"):
            RiskProfile(
                name="x", risk_aversion=5.0, max_weight=0.10,
                max_active_share=0.8, universe_size=100, n_positions=20,
                beta_premium_annual=0.30,
            )

    def test_rejects_optimistic_ic(self) -> None:
        with pytest.raises(ValueError, match="score_ic"):
            PROFILES["balanced"].with_(score_ic=0.5)


class TestPeriodConversion:
    def test_annual_to_period(self) -> None:
        p = PROFILES["aggressive"]
        assert p.beta_premium_per_period(12) == pytest.approx(0.015 / 12)
        assert p.beta_premium_per_period(252) == pytest.approx(0.015 / 252)

    def test_optimizer_config_converts_once(self) -> None:
        p = PROFILES["aggressive"]
        cfg = OptimizerConfig.from_profile(p, periods_per_year=12)
        assert cfg.beta_premium == pytest.approx(0.015 / 12)
        assert cfg.risk_aversion == p.risk_aversion
        assert cfg.universe_size == p.universe_size

    def test_periods_per_year_is_required(self) -> None:
        with pytest.raises(TypeError):
            OptimizerConfig.from_profile(PROFILES["balanced"])  # type: ignore[call-arg]

    def test_overrides_apply_last(self) -> None:
        cfg = OptimizerConfig.from_profile(
            PROFILES["balanced"], periods_per_year=12, score_ic=None
        )
        assert cfg.score_ic is None
        assert cfg.max_weight == 0.06

    def test_balanced_config_is_unchanged_from_the_hardcoded_one(self) -> None:
        """What the four scripts typed by hand, and must keep producing."""
        cfg = OptimizerConfig.from_profile(PROFILES["balanced"], periods_per_year=12)
        assert (
            cfg.max_weight, cfg.risk_aversion, cfg.trade_cost,
            cfg.max_active_share, cfg.universe_size, cfg.score_ic,
            cfg.beta_premium,
        ) == (0.06, 8.0, 0.005, 0.60, 200, 0.03, 0.0)


class TestBetaTilt:
    def test_zero_premium_is_the_identity(self, problem) -> None:
        alpha, risk = problem
        out = apply_beta_tilt(alpha, risk.exposures, premium=0.0)
        pd.testing.assert_series_equal(out, alpha)

    def test_tilt_moves_the_book_toward_high_beta(self, problem) -> None:
        alpha, risk = problem
        base = OptimizerConfig(
            risk_aversion=4.0, max_weight=0.15, trade_cost=0.0,
            max_active_share=None, universe_size=60, score_ic=None,
        )
        flat = optimize_weights(alpha, risk, config=base)
        tilted = optimize_weights(
            alpha, risk, config=dataclasses_replace(base, beta_premium=0.015 / 12)
        )
        b = risk.exposures["beta"]
        flat_beta = float((flat.weights * b.reindex(flat.weights.index)).sum())
        tilted_beta = float((tilted.weights * b.reindex(tilted.weights.index)).sum())
        assert tilted_beta > flat_beta

    def test_tilt_is_priced_not_forced(self, problem) -> None:
        """Higher risk aversion resists the same premium — it is traded off.

        A tilt implemented as a constraint would hit the target beta at any
        risk aversion. This one competes with the variance it adds, which is the
        difference between stating a view and overriding the risk model.
        """
        alpha, risk = problem
        b = risk.exposures["beta"]

        def book_beta(lam: float) -> float:
            r = optimize_weights(
                alpha, risk,
                config=OptimizerConfig(
                    risk_aversion=lam, max_weight=0.15, trade_cost=0.0,
                    max_active_share=None, universe_size=60, score_ic=None,
                    beta_premium=0.03 / 12,
                ),
            )
            return float((r.weights * b.reindex(r.weights.index)).sum())

        assert book_beta(2.0) > book_beta(50.0)

    def test_missing_exposure_raises(self, problem) -> None:
        alpha, risk = problem
        X = risk.exposures.drop(columns=["beta"])
        with pytest.raises(ValueError, match="not among"):
            apply_beta_tilt(alpha, X, premium=0.001)

    def test_unstandardized_exposure_raises(self, problem) -> None:
        """Raw betas: mean ~1.0, SD ~0.35. The premium would mean something else."""
        alpha, risk = problem
        X = risk.exposures.copy()
        X["beta"] = 1.0 + 0.35 * X["beta"]
        with pytest.raises(ExposureScaleError, match="quoted per"):
            apply_beta_tilt(alpha, X, premium=0.001)

    def test_unknown_beta_gets_no_tilt(self, problem) -> None:
        alpha, risk = problem
        X = risk.exposures.copy()
        X.loc[X.index[:5], "beta"] = np.nan
        out = apply_beta_tilt(alpha, X, premium=0.01)
        pd.testing.assert_series_equal(
            out.iloc[:5], alpha.iloc[:5], check_names=False
        )


class TestBacktestConfigFromProfile:
    def test_takes_only_the_appetite_fields(self) -> None:
        cfg = BacktestConfig.from_profile(PROFILES["aggressive"])
        assert cfg.n_positions == 25
        assert cfg.max_weight == 0.09
        assert cfg.initial_cash == BacktestConfig().initial_cash

    def test_overrides_pass_through(self) -> None:
        cfg = BacktestConfig.from_profile(
            PROFILES["balanced"], initial_cash=250_000.0, rebalance_every=1
        )
        assert (cfg.initial_cash, cfg.rebalance_every) == (250_000.0, 1)


class TestRegistryPayload:
    def test_every_field_is_hashed(self) -> None:
        from qe_eval.registry import config_hash

        p = PROFILES["balanced"]
        base = config_hash(p.payload())
        for field, value in (
            ("risk_aversion", 7.0), ("max_weight", 0.07),
            ("max_active_share", 0.65), ("universe_size", 201),
            ("n_positions", 41), ("beta_premium_annual", 0.01),
            ("trade_cost", 0.006), ("score_ic", 0.02),
        ):
            assert config_hash(p.with_(**{field: value}).payload()) != base, field

    def test_a_swept_profile_is_a_new_trial(self) -> None:
        """Sweeping appetite and keeping the winner is a search. It must count."""
        from qe_eval.registry import TrialRegistry

        reg = TrialRegistry(":memory:")
        for name in ("balanced", "aggressive"):
            reg.record(
                signals=["sue", "short_term_reversal"], universe="crsp",
                start="1975-01-01", end="2025-12-31", sr_period=0.1, n_obs=600,
                scope="portfolio_construction",
                config=PROFILES[name].payload(),
            )
        assert reg.n_trials(scope="portfolio_construction") == 2
        # Re-running the identical profile is not a new trial.
        reg.record(
            signals=["sue", "short_term_reversal"], universe="crsp",
            start="1975-01-01", end="2025-12-31", sr_period=0.1, n_obs=600,
            scope="portfolio_construction",
            config=PROFILES["balanced"].payload(),
        )
        assert reg.n_trials(scope="portfolio_construction") == 2


class TestLoading:
    def test_by_name(self) -> None:
        assert load_profile("aggressive") is PROFILES["aggressive"]

    def test_passthrough(self) -> None:
        p = PROFILES["balanced"]
        assert load_profile(p) is p

    def test_unknown_name_lists_the_known_ones(self) -> None:
        with pytest.raises(KeyError, match="conservative"):
            load_profile("yolo")

    def test_yaml_round_trip(self, tmp_path) -> None:
        import yaml

        f = tmp_path / "spicy.yaml"
        f.write_text(yaml.safe_dump({
            "risk_aversion": 3.0, "max_weight": 0.12, "max_active_share": 0.9,
            "universe_size": 80, "n_positions": 20, "beta_premium_annual": 0.02,
        }))
        p = load_profile(f)
        assert p.name == "spicy"
        assert p.risk_aversion == 3.0

    def test_yaml_typo_raises_rather_than_ignoring(self, tmp_path) -> None:
        import yaml

        f = tmp_path / "typo.yaml"
        f.write_text(yaml.safe_dump({
            "risk_aversion": 3.0, "max_weight": 0.12, "max_active_share": 0.9,
            "universe_size": 80, "n_positions": 20, "beta_premium_annual": 0.02,
            "risk_averison": 99.0,
        }))
        with pytest.raises(ValueError, match="unknown risk-profile fields"):
            load_profile(f)


def dataclasses_replace(cfg, **kw):
    import dataclasses

    return dataclasses.replace(cfg, **kw)
