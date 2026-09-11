"""Reference numerics for the ported statistics.

The source file had 2 of 8 functions tested and nothing checked against a
reference implementation. Treating ported code as verified because it "reads
correct" is how a subtly wrong sandwich estimator survives for years, so these
check against statsmodels and scipy rather than against intuition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from qe_eval.stats import (
    _auto_lags,
    ann_sharpe,
    block_bootstrap_pvalue,
    deflated_sharpe,
    expected_max_sharpe,
    newey_west_alpha,
    placebo_distribution,
    probabilistic_sharpe,
    tstat_mean,
)
from scipy import stats as sps


@pytest.fixture
def sample():
    rng = np.random.default_rng(4)
    n = 900
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    factors = pd.DataFrame(
        rng.normal(0, 0.01, size=(n, 3)), index=idx, columns=["mkt", "smb", "hml"]
    )
    y = pd.Series(
        0.0004 + 1.1 * factors["mkt"] + 0.3 * factors["smb"] + rng.normal(0, 0.008, n),
        index=idx,
        name="strategy",
    )
    return y, factors


# -- against statsmodels --------------------------------------------------


@pytest.mark.parametrize("lags", [1, 4, 12])
def test_newey_west_matches_statsmodels(sample, lags):
    y, X = sample
    alpha, tstat = newey_west_alpha(y, X, lags=lags)

    ref = sm.OLS(y.to_numpy(), sm.add_constant(X.to_numpy())).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": False}
    )
    assert alpha == pytest.approx(ref.params[0], rel=1e-10)
    assert tstat == pytest.approx(ref.tvalues[0], rel=1e-8)


def test_newey_west_recovers_a_known_alpha(sample):
    y, X = sample
    alpha, tstat = newey_west_alpha(y, X)
    assert alpha == pytest.approx(0.0004, abs=3e-4)
    assert tstat > 0


@pytest.mark.parametrize("n, want", [(100, 4), (900, 6), (14_000, 11)])
def test_newey_west_default_lags_follow_the_1994_rule(n, want):
    """floor(4 * (n/100)^(2/9)) — and it must grow with the sample, not sit at 6."""
    assert _auto_lags(n) == want
    assert _auto_lags(n) == int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def test_newey_west_rejects_degenerate_input(sample):
    y, X = sample
    with pytest.raises(ValueError, match="no overlapping"):
        newey_west_alpha(y.iloc[:0], X)


# -- against scipy --------------------------------------------------------


def test_expected_max_sharpe_matches_the_closed_form():
    n, var = 50, 0.04
    gamma = 0.5772156649015329
    z1 = sps.norm.ppf(1 - 1 / n)
    z2 = sps.norm.ppf(1 - 1 / (n * np.e))
    want = np.sqrt(var) * ((1 - gamma) * z1 + gamma * z2)
    assert expected_max_sharpe(n, var) == pytest.approx(want, rel=1e-12)


def test_probabilistic_sharpe_is_a_probability(sample):
    y, _ = sample
    p = probabilistic_sharpe(y)
    assert 0.0 <= p <= 1.0
    # Positive-mean, but the denominator carries factor variance as well as
    # residual, so the per-period Sharpe is ~0.03 and PSR lands near 0.87.
    assert p > 0.8


def test_probabilistic_sharpe_falls_with_a_higher_bar(sample):
    y, _ = sample
    assert probabilistic_sharpe(y, 0.0) > probabilistic_sharpe(y, 0.5)


def test_ann_sharpe_scales_by_root_periods():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.001, 0.01, 1000))
    per_period = r.mean() / r.std(ddof=1)
    assert ann_sharpe(r, 252) == pytest.approx(per_period * np.sqrt(252))


def test_tstat_matches_scipy_one_sample():
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0.001, 0.01, 500))
    assert tstat_mean(r) == pytest.approx(sps.ttest_1samp(r, 0.0).statistic)


# -- deflated Sharpe ------------------------------------------------------


def test_deflated_sharpe_penalises_more_trials(sample):
    y, _ = sample
    few = deflated_sharpe(y, 252, n_trials=2, var_sr_trials=0.01)
    many = deflated_sharpe(y, 252, n_trials=10_000, var_sr_trials=0.01)
    assert many["dsr"] < few["dsr"]
    assert many["sr_star"] > few["sr_star"]


def test_deflated_sharpe_penalises_more_dispersed_trials(sample):
    """The fix that matters: DSR must respond to trial-Sharpe variance."""
    y, _ = sample
    tight = deflated_sharpe(y, 252, n_trials=100, var_sr_trials=0.001)
    wide = deflated_sharpe(y, 252, n_trials=100, var_sr_trials=0.10)
    assert wide["dsr"] < tight["dsr"], (
        "DSR ignored var_sr_trials — this is the inherited defect, where sr_star "
        "was analytic and the trial registry could not affect the answer"
    )


def test_deflated_sharpe_flags_the_fallback(sample):
    y, _ = sample
    assert deflated_sharpe(y, 252, n_trials=50)["var_sr_source"] == "fallback_1_over_n"
    assert (
        deflated_sharpe(y, 252, n_trials=50, var_sr_trials=0.01)["var_sr_source"] == "registry"
    )


def test_deflated_sharpe_reports_both_scales(sample):
    y, _ = sample
    out = deflated_sharpe(y, 252, n_trials=10, var_sr_trials=0.01)
    assert out["sr_ann"] == pytest.approx(out["sr_period"] * np.sqrt(252))


# -- resampling -----------------------------------------------------------


def test_block_bootstrap_has_correct_size_under_the_null():
    """A zero-mean series should reject at roughly the nominal rate, not always."""
    rng = np.random.default_rng(0)
    rejects = 0
    trials = 60
    for i in range(trials):
        noise = pd.Series(rng.normal(0.0, 0.01, 300))
        if block_bootstrap_pvalue(noise, n_boot=400, seed=i) < 0.05:
            rejects += 1
    assert rejects / trials < 0.20, f"rejected {rejects}/{trials} under the null — oversized"


def test_block_bootstrap_detects_a_real_effect():
    rng = np.random.default_rng(3)
    strong = pd.Series(rng.normal(0.004, 0.01, 400))
    assert block_bootstrap_pvalue(strong, n_boot=600) < 0.05


def test_placebo_distribution_centres_on_zero():
    rng = np.random.default_rng(6)
    r = pd.Series(rng.normal(0.002, 0.01, 500))
    dist = placebo_distribution(r, n_draws=400)
    assert abs(dist.mean()) < 0.02
    assert dist.std() > 0
