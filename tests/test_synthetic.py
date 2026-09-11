"""The synthetic panel must be *recoverable*, not merely well-formed.

The sibling repo's generator produces cross-sectionally independent returns, so
the covariance it yields is spuriously near-diagonal and any risk model tested
against it passes for the wrong reason. These tests exist so that failure mode
cannot recur silently here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from qe_core.synthetic import FACTORS, generate


@pytest.fixture(scope="module")
def synth():
    # Five years daily (~1258 sessions). Longer than strictly needed for the
    # tests, but this is the fixture the Phase 3 risk model will be validated
    # against, and a better-conditioned target there is worth the extra second.
    return generate(n_firms=120, start="2015-01-02", end="2019-12-31", seed=7)


def _returns_wide(synth) -> pd.DataFrame:
    view = synth.panel.as_of("2017-12-29")
    rets = view.frame.loc[view.frame["concept"] == "ret"]
    return rets.pivot(index="period_end", columns="permno", values="value").sort_index()


# -- the property that matters -------------------------------------------


def test_planted_betas_are_recovered(synth):
    """OLS on the known factor returns must return the loadings we planted.

    Judged against OLS standard error rather than an absolute threshold. With
    realistic idiosyncratic vol the SE on a daily beta over three years is ~0.1,
    so a fixed tolerance would be either unachievable or meaningless depending on
    where it was set. Testing that errors sit inside their own standard errors is
    the claim actually worth making: the estimator is unbiased.
    """
    R = _returns_wide(synth)
    F = synth.factor_returns.loc[R.index]

    X = np.column_stack([np.ones(len(F)), F.to_numpy()])
    coef, *_ = np.linalg.lstsq(X, R.to_numpy(), rcond=None)
    recovered = pd.DataFrame(coef[1:].T, index=R.columns, columns=list(FACTORS))
    truth = synth.betas.loc[recovered.index]

    resid = R.to_numpy() - X @ coef
    dof = len(F) - X.shape[1]
    sigma2 = (resid**2).sum(axis=0) / dof
    xtx_inv_diag = np.diag(np.linalg.inv(X.T @ X))[1:]
    se = pd.DataFrame(
        np.sqrt(np.outer(sigma2, xtx_inv_diag)), index=R.columns, columns=list(FACTORS)
    )

    z = ((recovered - truth) / se).abs()
    assert z.to_numpy().max() < 4.5, (
        f"beta recovery is biased; worst |z| = {z.to_numpy().max():.2f}\n{z.max().to_string()}"
    )
    assert z.to_numpy().mean() < 1.0

    # Rank agreement is capped by dispersion-to-noise, not by estimator quality:
    # market betas are drawn with sd 0.35 against size/value's 0.60, so at equal
    # estimation noise their correlation with truth is structurally lower. 0.97
    # is what is achievable here; the z-test above is the rigorous claim.
    for f in FACTORS:
        corr = np.corrcoef(recovered[f], truth[f])[0, 1]
        assert corr > 0.97, f"{f} betas do not track truth (corr {corr:.4f})"


def test_covariance_is_visibly_non_diagonal(synth):
    """A near-diagonal covariance is the failure this generator exists to avoid."""
    R = _returns_wide(synth)
    corr = R.corr().to_numpy()
    off = corr[~np.eye(len(corr), dtype=bool)]

    assert np.mean(off) > 0.15, (
        f"mean off-diagonal correlation is {np.mean(off):.3f} — returns are effectively "
        "independent, so any covariance or neutralization logic would validate against noise"
    )

    # A dominant first eigenvalue is the signature of a common factor.
    eig = np.linalg.eigvalsh(corr)[::-1]
    assert eig[0] / eig.sum() > 0.20


def test_market_beta_is_positive_on_average(synth):
    assert synth.betas["market"].mean() > 0.7
    assert (synth.betas["market"] > 0).all()


# -- panel hygiene --------------------------------------------------------


def test_fundamentals_carry_a_reporting_lag(synth):
    view = synth.panel.as_of("2017-12-29", revisions=True)
    fund = view.frame.loc[view.frame["concept"] == "at"]
    lag = (fund["reported_at"] - fund["period_end"]).dt.days
    assert (lag > 0).all(), "fundamentals reported on or before period end — no PIT trap present"
    assert lag.min() >= 70


def test_prices_are_knowable_same_session(synth):
    view = synth.panel.as_of("2017-12-29", revisions=True)
    px = view.frame.loc[view.frame["concept"] == "prc"]
    assert (px["knowledge_date"] == px["period_end"]).all()


def test_returns_have_fat_tails(synth):
    """Student-t innovations, so tail-sensitive machinery sees something real."""
    R = _returns_wide(synth)
    z = (R - R.mean()) / R.std()
    kurt = z.stack().kurtosis()
    assert kurt > 1.0, f"excess kurtosis {kurt:.2f} — tails are too thin to be interesting"


# -- the ragged case ------------------------------------------------------


def test_delisting_produces_a_valid_unbalanced_panel():
    """Delisted firms leave the panel without leaving nulls behind."""
    synth = generate(n_firms=80, start="2015-01-02", end="2017-12-29", seed=5, delist_rate=0.10)

    assert synth.delist_date.notna().any(), "delist_rate=0.10 produced no delistings"

    view = synth.panel.as_of("2017-12-29", revisions=True)
    assert view.frame["value"].notna().all(), "delisting left null values in the panel"

    # A delisted firm stops appearing after its delist date.
    gone = synth.delist_date.dropna().index[0]
    when = synth.delist_date.loc[gone]
    rows = view.frame.loc[(view.frame["permno"] == gone) & (view.frame["concept"] == "ret")]
    assert rows["period_end"].max() < when


def test_generation_is_deterministic():
    a = generate(n_firms=20, start="2016-01-04", end="2016-06-30", seed=42)
    b = generate(n_firms=20, start="2016-01-04", end="2016-06-30", seed=42)
    pd.testing.assert_frame_equal(a.betas, b.betas)
    pd.testing.assert_frame_equal(a.factor_returns, b.factor_returns)
