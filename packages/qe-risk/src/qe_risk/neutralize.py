"""Stripping risk exposure out of a signal.

Every scored signal is neutralized against the risk factors before it is scored,
so what remains is the part the risk model cannot explain. This is where a
"signal" that is really a size tilt goes to die, and it is meant to.

It is also why beta, idiosyncratic volatility, Amihud illiquidity, and turnover
were moved out of the signal library and into this package: a signal that *is* a
risk factor neutralizes to approximately zero, so scoring it as alpha is a
category error that produces a confidently meaningless number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["neutralize", "neutralize_panel", "standardize", "winsorize"]


def winsorize(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Clip to percentiles. Every one of these signals has fat tails."""
    valid = s.dropna()
    if valid.empty:
        return s
    lo, hi = valid.quantile(lower), valid.quantile(upper)
    return s.clip(lo, hi)


def standardize(s: pd.Series, *, weights: pd.Series | None = None) -> pd.Series:
    """Zero mean, unit standard deviation."""
    valid = s.dropna()
    if valid.empty:
        return s
    if weights is not None:
        w = weights.reindex(valid.index).fillna(0.0)
        mu = np.average(valid, weights=w) if w.sum() > 0 else valid.mean()
    else:
        mu = valid.mean()
    sd = valid.std(ddof=1)
    return (s - mu) / sd if sd > 0 else s - mu


def neutralize(
    signal: pd.Series,
    exposures: pd.DataFrame,
    *,
    weights: pd.Series | None = None,
) -> pd.Series:
    """Residualize one cross-section of a signal against risk exposures.

    Returns the part of `signal` orthogonal to `exposures`. Names missing from
    either input come back as NaN rather than zero — a name we cannot neutralize
    is a name we cannot score, and silently calling it average would quietly
    populate the portfolio with exactly the names we know least about.
    """
    names = signal.dropna().index.intersection(exposures.dropna().index)
    if len(names) < exposures.shape[1] + 2:
        return pd.Series(np.nan, index=signal.index, name=signal.name)

    y = signal.reindex(names).to_numpy(dtype=float)
    X = exposures.reindex(names).to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(names)), X])

    if weights is not None:
        w = weights.reindex(names).fillna(0.0).to_numpy(dtype=float)
        w = np.where(w > 0, w, np.nanmedian(w[w > 0]) if (w > 0).any() else 1.0)
        sw = np.sqrt(w)
        beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    else:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    resid = pd.Series(y - X @ beta, index=names, name=signal.name)
    return resid.reindex(signal.index)


def neutralize_panel(
    signal: pd.DataFrame,
    exposures: pd.DataFrame,
    *,
    weights: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Neutralize a date x permno signal panel, one cross-section at a time.

    Cross-sectionally, not pooled: pooling would let the relationship between
    signal and exposure in one era contaminate another, and these relationships
    are famously unstable across decades.
    """
    time_varying = isinstance(exposures.index, pd.MultiIndex)
    out = {}
    for date, row in signal.iterrows():
        if time_varying:
            if date not in exposures.index.get_level_values(0):
                continue
            X = exposures.loc[date]
        else:
            X = exposures
        w = weights.loc[date] if weights is not None and date in weights.index else None
        out[date] = neutralize(row, X, weights=w)
    return pd.DataFrame(out).T.reindex(columns=signal.columns)
