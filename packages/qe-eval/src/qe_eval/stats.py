"""Performance statistics, with the three inherited defects fixed.

Ported from `quant-screen/battery/stats.py`, which had the right functions. Three
things are changed, all of which were wrong rather than merely stylistic:

1. **`sr_star` is empirical, not analytic.** The original used ``sqrt(1/(n-1))``
   as a stand-in for the variance of Sharpe ratios *across trials*, which is not
   what Bailey & López de Prado's Deflated Sharpe uses — it wants the observed
   cross-section. Building a trial registry does nothing for DSR unless this
   accepts it, so :func:`deflated_sharpe` takes ``var_sr_trials``.
2. **Defaults are period-denominated, not monthly.** The original hardcoded
   ``lags=6``, ``block=12`` — sensible for the ~480 monthly bars it was written
   for, wrong for ~14k daily ones. Here they derive from the sample.
3. **scipy is a hard dependency.** The original carried a hand-rolled Acklam
   inverse-normal behind a try/except, entirely untested. Deleting an unverified
   numerics path beats reference-testing it.

Everything here works on **per-period** returns. Annualization is a reporting
concern and happens at the edges; mixing the two is how Sharpe formulas silently
go wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as _sps

__all__ = [
    "ann_sharpe",
    "block_bootstrap_pvalue",
    "deflated_sharpe",
    "newey_west_alpha",
    "placebo_distribution",
    "probabilistic_sharpe",
    "tstat_mean",
]

#: Euler-Mascheroni, used in the expected-maximum-Sharpe approximation.
_GAMMA = 0.5772156649015329


def _clean(r: pd.Series | np.ndarray) -> np.ndarray:
    a = np.asarray(pd.Series(r).dropna(), dtype=float)
    if a.size == 0:
        raise ValueError("empty return series")
    return a


def _auto_lags(n: int) -> int:
    """Newey-West lag length, Newey-West (1994) rule of thumb: 4(n/100)^(2/9)."""
    return max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))


def _auto_block(n: int) -> int:
    """Bootstrap block length ~ n^(1/3), the standard order for block bootstraps."""
    return max(2, round(n ** (1.0 / 3.0)))


# -- point statistics -----------------------------------------------------


def ann_sharpe(r: pd.Series, periods_py: int) -> float:
    """Annualized Sharpe. Excess returns in, please — nothing is subtracted here."""
    a = _clean(r)
    sd = a.std(ddof=1)
    return 0.0 if sd == 0 else float(a.mean() / sd * np.sqrt(periods_py))


def tstat_mean(r: pd.Series) -> float:
    """iid t-stat of the mean. No HAC — use :func:`newey_west_alpha` when it matters."""
    a = _clean(r)
    sd = a.std(ddof=1)
    return 0.0 if sd == 0 else float(a.mean() / (sd / np.sqrt(a.size)))


def newey_west_alpha(
    y: pd.Series, X: pd.DataFrame, lags: int | None = None
) -> tuple[float, float]:
    """Intercept and HAC t-stat from regressing `y` on `[1, X]`.

    A signal that does not span the factor model is the only interesting kind, so
    this is the workhorse. `lags` defaults to the Newey-West (1994) rule rather
    than a hardcoded 6.
    """
    joined = pd.concat([y.rename("_y"), X], axis=1).dropna()
    if joined.empty:
        raise ValueError("no overlapping observations between y and X")

    yv = joined["_y"].to_numpy(dtype=float)
    Xv = np.column_stack([np.ones(len(joined)), joined.drop(columns="_y").to_numpy(dtype=float)])
    n, k = Xv.shape
    if n <= k:
        raise ValueError(f"need more observations than regressors, got n={n}, k={k}")

    lags = _auto_lags(n) if lags is None else lags
    beta, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
    resid = yv - Xv @ beta

    # Bartlett-kernel HAC sandwich.
    S = (Xv * resid[:, None]).T @ (Xv * resid[:, None])
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        A = (Xv[lag:] * resid[lag:, None]).T @ (Xv[:-lag] * resid[:-lag, None])
        S += w * (A + A.T)

    xtx_inv = np.linalg.inv(Xv.T @ Xv)
    cov = xtx_inv @ S @ xtx_inv
    se = float(np.sqrt(np.maximum(cov[0, 0], 0.0)))
    alpha = float(beta[0])
    return alpha, (0.0 if se == 0 else alpha / se)


# -- Sharpe deflation -----------------------------------------------------


def probabilistic_sharpe(r: pd.Series, sr_benchmark: float = 0.0) -> float:
    """P(true per-period Sharpe > `sr_benchmark`), adjusting for skew and kurtosis.

    Non-normal returns are the norm, not the exception, and a plain Sharpe
    flatters negatively-skewed strategies badly.
    """
    a = _clean(r)
    n = a.size
    if n < 3:
        raise ValueError(f"need at least 3 observations, got {n}")
    sd = a.std(ddof=1)
    if sd == 0:
        return 0.0

    sr = a.mean() / sd
    skew = float(_sps.skew(a, bias=False))
    kurt = float(_sps.kurtosis(a, fisher=False, bias=False))  # raw, not excess

    denom = np.sqrt(max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2, 1e-12))
    return float(_sps.norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / denom))


def expected_max_sharpe(n_trials: int, var_sr_trials: float) -> float:
    """Expected maximum per-period Sharpe from `n_trials` independent draws.

    Bailey & López de Prado's approximation. `var_sr_trials` is the variance of
    the Sharpes actually observed across trials — the whole reason the trial
    registry exists.
    """
    n = max(int(n_trials), 2)
    z1 = _sps.norm.ppf(1.0 - 1.0 / n)
    z2 = _sps.norm.ppf(1.0 - 1.0 / (n * np.e))
    return float(np.sqrt(max(var_sr_trials, 0.0)) * ((1.0 - _GAMMA) * z1 + _GAMMA * z2))


def deflated_sharpe(
    r: pd.Series,
    periods_py: int,
    n_trials: int,
    var_sr_trials: float | None = None,
) -> dict:
    """Deflated Sharpe: is this the best of many tries, or actually good?

    Parameters
    ----------
    var_sr_trials:
        Variance of per-period Sharpes across the trials that were run — take it
        from :meth:`qe_eval.registry.TrialRegistry.sharpe_variance`. If omitted,
        falls back to the ``1/(n-1)`` stand-in the original used, and the returned
        dict says so in ``var_sr_source``. That fallback is a placeholder, not an
        estimate; a DSR computed from it should not be reported as final.
    """
    a = _clean(r)
    n = a.size
    if n < 3:
        raise ValueError(f"need at least 3 observations, got {n}")
    sd = a.std(ddof=1)
    if sd == 0:
        return {
            "sr_ann": 0.0, "sr_period": 0.0, "sr_star": 0.0, "dsr": 0.0,
            "pvalue": 1.0, "skew": 0.0, "kurt": 3.0, "n_trials": int(n_trials),
            "n_obs": n, "var_sr_source": "degenerate",
        }

    sr = float(a.mean() / sd)
    if var_sr_trials is None:
        var_sr, source = 1.0 / max(n - 1, 1), "fallback_1_over_n"
    else:
        var_sr, source = float(var_sr_trials), "registry"

    sr_star = expected_max_sharpe(n_trials, var_sr)
    skew = float(_sps.skew(a, bias=False))
    kurt = float(_sps.kurtosis(a, fisher=False, bias=False))

    denom = np.sqrt(max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2, 1e-12))
    dsr = float(_sps.norm.cdf((sr - sr_star) * np.sqrt(n - 1) / denom))

    return {
        "sr_ann": sr * np.sqrt(periods_py),
        "sr_period": sr,
        "sr_star": sr_star,
        "dsr": dsr,
        "pvalue": 1.0 - dsr,
        "skew": skew,
        "kurt": kurt,
        "n_trials": max(int(n_trials), 2),
        "n_obs": n,
        "var_sr_source": source,
    }


# -- resampling -----------------------------------------------------------


def _circular_blocks(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Index array for one circular block-bootstrap resample of length n."""
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=n_blocks)
    idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
    return idx[:n]


def block_bootstrap_pvalue(
    diff: pd.Series,
    n_boot: int = 2000,
    block: int | None = None,
    seed: int = 11,
) -> float:
    """One-sided p-value for H0: mean(diff) <= 0, via circular block bootstrap.

    Preserves autocorrelation, which an iid bootstrap destroys and which matters
    for anything computed on overlapping windows.
    """
    a = _clean(diff)
    n = a.size
    block = _auto_block(n) if block is None else block
    observed = a.mean()

    centered = a - observed  # impose the null
    rng = np.random.default_rng(seed)
    means = np.array(
        [centered[_circular_blocks(n, block, rng)].mean() for _ in range(n_boot)]
    )
    return float((means >= observed).mean())


def placebo_distribution(
    returns_like: pd.Series,
    n_draws: int,
    seed: int = 13,
    block: int | None = None,
) -> np.ndarray:
    """Null distribution of per-period Sharpe under block-resampled, sign-randomized returns.

    Demeaned first, so placebos carry no drift and the distribution centres on zero.
    """
    a = _clean(returns_like)
    n = a.size
    block = _auto_block(n) if block is None else block
    centered = a - a.mean()

    rng = np.random.default_rng(seed)
    out = np.empty(n_draws, dtype=float)
    for i in range(n_draws):
        sample = centered[_circular_blocks(n, block, rng)]
        n_blocks = int(np.ceil(n / block))
        signs = np.repeat(rng.choice([-1.0, 1.0], size=n_blocks), block)[:n]
        sample = sample * signs
        sd = sample.std(ddof=1)
        out[i] = 0.0 if sd == 0 else sample.mean() / sd
    return out
