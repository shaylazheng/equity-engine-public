"""Factor covariance: EWMA, autocorrelation correction, shrinkage.

A raw sample covariance on N assets from T observations is badly conditioned
whenever N is not small relative to T, and its extreme eigenvalues are biased —
the smallest too small, the largest too large. An optimizer handed that matrix
will pile into whatever direction the estimate wrongly calls low-risk, which is
how "risk models" produce concentrated portfolios that blow up.

Three corrections, in the order they should be applied:

1. **EWMA** — recent observations matter more; volatility clusters.
2. **Newey-West** — daily factor returns are autocorrelated, so the naive
   scaling to longer horizons understates risk.
3. **Shrinkage** — pull the estimate toward a structured target, trading a little
   bias for a large variance reduction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "build_factor_covariance",
    "eigenvalue_adjust",
    "ewma_cov",
    "ledoit_wolf_intensity",
    "newey_west_adjust",
    "shrink_to_diagonal",
]


def ewma_cov(returns: pd.DataFrame, halflife: float = 90.0, min_periods: int = 30) -> pd.DataFrame:
    """Exponentially weighted covariance, as of the last observation."""
    X = returns.dropna(how="all")
    if len(X) < min_periods:
        raise ValueError(f"need at least {min_periods} observations, got {len(X)}")

    lam = 0.5 ** (1.0 / halflife)
    n = len(X)
    w = lam ** np.arange(n - 1, -1, -1)
    w /= w.sum()

    A = X.to_numpy(dtype=float)
    mu = np.average(A, axis=0, weights=w)
    C = np.einsum("t,ti,tj->ij", w, A - mu, A - mu)
    return pd.DataFrame(C, index=X.columns, columns=X.columns)


def newey_west_adjust(
    returns: pd.DataFrame, cov: pd.DataFrame, lags: int = 5
) -> pd.DataFrame:
    """Inflate a covariance for serial correlation in the underlying series.

    Positive autocorrelation means multi-period risk exceeds the naive
    sqrt(horizon) scaling. Without this the model reports a portfolio as safer
    than it is at exactly the horizons you actually hold it.
    """
    A = returns.dropna(how="all").to_numpy(dtype=float)
    A = A - A.mean(axis=0)
    n = len(A)
    S = cov.to_numpy(dtype=float).copy()

    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        G = (A[lag:].T @ A[:-lag]) / n
        S += weight * (G + G.T)

    # Serial correlation can push the result out of the PSD cone; project back.
    return eigenvalue_adjust(pd.DataFrame(S, index=cov.index, columns=cov.columns))


def shrink_to_diagonal(cov: pd.DataFrame, intensity: float) -> pd.DataFrame:
    """Shrink toward the diagonal (zero off-diagonal correlation) target."""
    if not 0.0 <= intensity <= 1.0:
        raise ValueError(f"intensity must be in [0, 1], got {intensity}")
    A = cov.to_numpy(dtype=float)
    target = np.diag(np.diag(A))
    out = (1.0 - intensity) * A + intensity * target
    return pd.DataFrame(out, index=cov.index, columns=cov.columns)


def ledoit_wolf_intensity(returns: pd.DataFrame) -> float:
    """Ledoit-Wolf optimal shrinkage intensity toward a diagonal target.

    Estimates the intensity that minimises expected squared error, so it is not a
    tuning knob: it is larger when the sample is short relative to the number of
    series, which is exactly when the raw estimate is least trustworthy.
    """
    X = returns.dropna(how="all").to_numpy(dtype=float)
    n, _p = X.shape
    if n < 3:
        raise ValueError(f"need at least 3 observations, got {n}")

    Xc = X - X.mean(axis=0)
    S = (Xc.T @ Xc) / n
    target = np.diag(np.diag(S))

    # Sum of asymptotic variances of the sample covariance entries.
    var_sum = 0.0
    for t in range(n):
        outer = np.outer(Xc[t], Xc[t])
        var_sum += float(np.sum((outer - S) ** 2))
    pi_hat = var_sum / n

    gamma = float(np.sum((S - target) ** 2))
    if gamma <= 0:
        return 0.0
    return float(np.clip((pi_hat / n) / gamma, 0.0, 1.0))


def eigenvalue_adjust(cov: pd.DataFrame, floor: float = 1e-12) -> pd.DataFrame:
    """Project onto the PSD cone, flooring non-positive eigenvalues.

    Shrinkage and Newey-West can each produce a matrix with tiny negative
    eigenvalues. An optimizer given one will happily find an "arbitrage" along
    that direction, so it has to be removed rather than tolerated.
    """
    A = cov.to_numpy(dtype=float)
    A = (A + A.T) / 2.0
    vals, vecs = np.linalg.eigh(A)
    vals = np.maximum(vals, floor)
    out = vecs @ np.diag(vals) @ vecs.T
    return pd.DataFrame((out + out.T) / 2.0, index=cov.index, columns=cov.columns)


def build_factor_covariance(
    factor_returns: pd.DataFrame,
    *,
    halflife: float = 90.0,
    nw_lags: int = 5,
    shrink: float | None = None,
    annualize: int | None = 252,
) -> pd.DataFrame:
    """The full pipeline: EWMA, Newey-West, shrinkage, PSD projection.

    `shrink=None` uses the Ledoit-Wolf optimal intensity rather than a guess.
    """
    cov = ewma_cov(factor_returns, halflife=halflife)
    cov = newey_west_adjust(factor_returns, cov, lags=nw_lags)

    intensity = ledoit_wolf_intensity(factor_returns) if shrink is None else shrink
    cov = shrink_to_diagonal(cov, intensity)
    cov = eigenvalue_adjust(cov)

    if annualize:
        cov = cov * annualize
    return cov
