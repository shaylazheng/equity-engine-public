"""The risk model must recover known truth before it is pointed at real data.

This is the reason the synthetic generator plants factor loadings and returns
them. The sibling repo's generator produces cross-sectionally independent
returns, so a risk model fitted to it would produce a near-diagonal covariance
and look perfectly healthy while being entirely uninformative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from qe_core.synthetic import FACTORS, generate
from qe_risk.covariance import (
    build_factor_covariance,
    eigenvalue_adjust,
    ewma_cov,
    ledoit_wolf_intensity,
    newey_west_adjust,
    shrink_to_diagonal,
)
from qe_risk.model import fit_cross_sectional, industry_dummies
from qe_risk.neutralize import neutralize, neutralize_panel, standardize, winsorize


@pytest.fixture(scope="module")
def synth():
    # 400 names. The precision of a cross-sectional factor return goes as
    # 1/sqrt(N), so a thin cross-section recovers the low-dispersion factors
    # poorly — at 150 names the size factor only reaches corr 0.91. Real daily
    # cross-sections carry thousands of names, so 150 was the unrealistic part.
    return generate(n_firms=400, start="2015-01-02", end="2018-12-31", seed=21)


@pytest.fixture(scope="module")
def returns(synth):
    view = synth.panel.as_of("2018-12-31")
    rets = view.frame.loc[view.frame["concept"] == "ret"]
    return rets.pivot(index="period_end", columns="permno", values="value").sort_index()


@pytest.fixture(scope="module")
def fitted(synth, returns):
    return fit_cross_sectional(returns, synth.betas)


# -- the ground-truth test ------------------------------------------------


def test_recovers_the_planted_factor_returns(fitted, synth):
    """Given true exposures, the cross-sectional fit must return the true factor returns."""
    est = fitted.factor_returns
    truth = synth.factor_returns.loc[est.index]

    for f in FACTORS:
        corr = np.corrcoef(est[f], truth[f])[0, 1]
        assert corr > 0.95, f"{f} factor returns not recovered (corr {corr:.3f})"
        # And on the right scale, not merely correlated.
        assert est[f].std() == pytest.approx(truth[f].std(), rel=0.20)


def test_cross_sectional_fit_explains_real_variance(fitted):
    """A *daily* cross-sectional R2 is inherently low — idiosyncratic variance
    dominates on any single day, and three factors against ~3e-4 of specific
    variance can only reach ~0.11 here. The bar is that the factors explain
    something real, not that they explain most of it."""
    assert fitted.r2.mean() > 0.05, f"mean R2 {fitted.r2.mean():.3f} — factors explain nothing"
    assert (fitted.r2 <= 1.0).all()


def test_specific_returns_are_orthogonal_to_the_factors(fitted, synth):
    """Residuals must be uncorrelated with the factor returns, or the fit is wrong."""
    spec = fitted.specific_returns.mean(axis=1)
    for f in FACTORS:
        corr = np.corrcoef(spec, fitted.factor_returns[f])[0, 1]
        assert abs(corr) < 0.25, f"specific returns still carry {f} exposure ({corr:.3f})"


def test_coverage_is_reported(fitted, returns):
    assert (fitted.coverage > 0).all()
    assert fitted.coverage.max() <= returns.shape[1]


def test_thin_cross_sections_are_skipped(returns, synth):
    """A regression on a handful of names is noise with a coefficient attached."""
    with pytest.raises(ValueError, match="usable names"):
        fit_cross_sectional(returns.iloc[:, :5], synth.betas, min_names=20)


def test_sqrt_cap_weighting_changes_the_fit(returns, synth):
    rng = np.random.default_rng(1)
    caps = pd.DataFrame(
        rng.lognormal(10, 1.5, size=returns.shape), index=returns.index, columns=returns.columns
    )
    weighted = fit_cross_sectional(returns, synth.betas, mktcap=caps)
    unweighted = fit_cross_sectional(returns, synth.betas)
    assert not np.allclose(
        weighted.factor_returns.to_numpy(), unweighted.factor_returns.to_numpy()
    )


def test_industry_dummies_drop_one_level():
    ind = pd.Series(["tech", "tech", "energy", "banks"], index=[1, 2, 3, 4])
    dummies = industry_dummies(ind)
    assert dummies.shape[1] == 2, "keeping every dummy alongside an intercept is singular"


# -- covariance -----------------------------------------------------------


def test_ewma_weights_recent_observations_more(fitted):
    fr = fitted.factor_returns
    short = ewma_cov(fr, halflife=20)
    long = ewma_cov(fr, halflife=250)
    assert not np.allclose(short.to_numpy(), long.to_numpy())


def test_covariance_is_symmetric_and_psd(fitted):
    cov = build_factor_covariance(fitted.factor_returns)
    A = cov.to_numpy()
    assert np.allclose(A, A.T)
    assert np.linalg.eigvalsh(A).min() >= -1e-10


def test_newey_west_inflates_risk_under_autocorrelation():
    """Positive serial correlation means real risk exceeds the naive estimate."""
    rng = np.random.default_rng(5)
    n = 800
    shocks = rng.normal(0, 0.01, n)
    ar = np.zeros(n)
    for t in range(1, n):
        ar[t] = 0.5 * ar[t - 1] + shocks[t]
    df = pd.DataFrame({"f": ar})

    plain = ewma_cov(df, halflife=200)
    adjusted = newey_west_adjust(df, plain, lags=6)
    assert adjusted.iloc[0, 0] > plain.iloc[0, 0]


def test_shrinkage_pulls_toward_the_diagonal(fitted):
    cov = ewma_cov(fitted.factor_returns, halflife=90)
    shrunk = shrink_to_diagonal(cov, 0.5)
    off_before = np.abs(cov.to_numpy() - np.diag(np.diag(cov.to_numpy()))).sum()
    off_after = np.abs(shrunk.to_numpy() - np.diag(np.diag(shrunk.to_numpy()))).sum()
    assert off_after == pytest.approx(off_before * 0.5)


def test_full_shrinkage_gives_a_diagonal_matrix(fitted):
    cov = ewma_cov(fitted.factor_returns)
    full = shrink_to_diagonal(cov, 1.0).to_numpy()
    assert np.allclose(full, np.diag(np.diag(full)))


def test_shrinkage_intensity_rises_as_the_sample_shortens():
    """Ledoit-Wolf is not a knob: it shrinks harder when the estimate is worse."""
    rng = np.random.default_rng(9)
    wide = pd.DataFrame(rng.normal(0, 0.01, size=(1500, 8)))
    short = pd.DataFrame(rng.normal(0, 0.01, size=(40, 8)))
    assert ledoit_wolf_intensity(short) > ledoit_wolf_intensity(wide)


def test_intensity_is_a_valid_fraction(fitted):
    assert 0.0 <= ledoit_wolf_intensity(fitted.factor_returns) <= 1.0


def test_eigenvalue_adjust_repairs_a_non_psd_matrix():
    bad = pd.DataFrame([[1.0, 2.0], [2.0, 1.0]], index=["a", "b"], columns=["a", "b"])
    assert np.linalg.eigvalsh(bad.to_numpy()).min() < 0
    fixed = eigenvalue_adjust(bad)
    assert np.linalg.eigvalsh(fixed.to_numpy()).min() >= -1e-12


def test_annualization_scales_the_covariance(fitted):
    daily = build_factor_covariance(fitted.factor_returns, annualize=None)
    annual = build_factor_covariance(fitted.factor_returns, annualize=252)
    assert np.allclose(annual.to_numpy(), daily.to_numpy() * 252)


# -- neutralization -------------------------------------------------------


def test_neutralizing_a_risk_factor_leaves_nothing(synth):
    """The reason beta and friends were moved out of the signal library."""
    signal = synth.betas["market"].copy()
    resid = neutralize(signal, synth.betas)
    assert resid.abs().max() < 1e-8, "a pure risk exposure did not neutralize to zero"


def test_neutralization_removes_correlation_with_exposures(synth):
    rng = np.random.default_rng(3)
    signal = synth.betas["size"] * 2.0 + pd.Series(
        rng.normal(0, 1, len(synth.betas)), index=synth.betas.index
    )
    resid = neutralize(signal, synth.betas).dropna()
    corr = np.corrcoef(resid, synth.betas["size"].reindex(resid.index))[0, 1]
    assert abs(corr) < 1e-8


def test_neutralization_keeps_the_orthogonal_part(synth):
    rng = np.random.default_rng(4)
    pure = pd.Series(rng.normal(0, 1, len(synth.betas)), index=synth.betas.index)
    resid = neutralize(pure, synth.betas).dropna()
    assert np.corrcoef(resid, pure.reindex(resid.index))[0, 1] > 0.9


def test_unneutralizable_names_come_back_missing_not_zero(synth):
    """Calling an unscoreable name 'average' would fill the book with the names
    we know least about."""
    signal = synth.betas["market"].copy()
    signal.iloc[:5] = np.nan
    exposures = synth.betas.copy()
    exposures.iloc[5:10] = np.nan
    resid = neutralize(signal, exposures)
    assert resid.iloc[:10].isna().all()


def test_too_few_names_yields_all_missing():
    tiny = pd.Series([1.0, 2.0], index=[1, 2])
    exposures = pd.DataFrame({"a": [1.0, 2.0], "b": [2.0, 1.0]}, index=[1, 2])
    assert neutralize(tiny, exposures).isna().all()


def test_panel_neutralization_runs_cross_section_by_cross_section(returns, synth):
    panel = returns.iloc[:40]
    out = neutralize_panel(panel, synth.betas)
    assert out.shape == panel.shape
    for _, row in out.iterrows():
        valid = row.dropna()
        if len(valid) > 10:
            corr = np.corrcoef(valid, synth.betas["market"].reindex(valid.index))[0, 1]
            assert abs(corr) < 1e-6


# -- helpers --------------------------------------------------------------


def test_winsorize_clips_the_tails():
    s = pd.Series([-100.0, *range(98), 500.0], dtype=float)
    out = winsorize(s)
    assert out.max() < 500 and out.min() > -100


def test_standardize_gives_zero_mean_unit_sd():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = standardize(s)
    assert out.mean() == pytest.approx(0.0)
    assert out.std(ddof=1) == pytest.approx(1.0)


def test_standardize_handles_a_constant_series():
    out = standardize(pd.Series([3.0, 3.0, 3.0]))
    assert (out == 0).all()
