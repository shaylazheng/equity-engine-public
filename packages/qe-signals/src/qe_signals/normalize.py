"""Dual-baseline normalization.

Raw activity is meaningless without a baseline, and one baseline is not enough.
Both are required:

- **Own history** — how unusual is this *for this company*, against its own
  trailing 2-3 years.
- **Cross-section** — how unusual *versus peers right now*, as a percentile
  within sector x size bucket.

Two details from that spec are easy to paraphrase away and are kept exactly:

1. The cross-sectional leg is a **percentile**, not a z-score. These signals all
   have fat tails, and a z-score against a fat-tailed cross-section is dominated
   by whichever name happened to blow up.
2. A **materiality floor** applies in *raw* units while the threshold applies to
   the *normalized* statistic. Two scales, deliberately not blended — otherwise a
   statistically remarkable but trivially small event fires.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "apply_materiality",
    "cross_sectional_percentile",
    "dual_baseline",
    "own_history_z",
    "size_buckets",
]


def size_buckets(mktcap: pd.Series, n: int = 5) -> pd.Series:
    """Label each name by size quintile. Peer groups need a size axis, not just sector."""
    valid = mktcap.dropna()
    if valid.empty:
        return pd.Series(index=mktcap.index, dtype="object")
    try:
        labels = pd.qcut(valid.rank(method="first"), n, labels=[f"q{i + 1}" for i in range(n)])
    except ValueError:  # too few distinct values to bucket
        labels = pd.Series("q1", index=valid.index)
    return pd.Series(labels, index=valid.index).reindex(mktcap.index)


def cross_sectional_percentile(
    signal: pd.Series,
    *,
    sector: pd.Series | None = None,
    size_bucket: pd.Series | None = None,
    min_group: int = 10,
) -> pd.Series:
    """Percentile rank within sector x size bucket, mapped to [-1, 1].

    Groups smaller than `min_group` fall back to the whole cross-section: a
    percentile computed over four names is not a percentile, it is a coin toss
    with decimal places.
    """
    valid = signal.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=signal.index, name=signal.name)

    keys = []
    if sector is not None:
        keys.append(sector.reindex(valid.index).astype("object").fillna("_na"))
    if size_bucket is not None:
        keys.append(size_bucket.reindex(valid.index).astype("object").fillna("_na"))

    if not keys:
        ranked = valid.rank(pct=True)
    else:
        group = pd.Series(list(zip(*[k.to_numpy() for k in keys], strict=True)), index=valid.index)
        counts = group.map(group.value_counts())
        ranked = pd.Series(np.nan, index=valid.index)

        big = counts >= min_group
        if big.any():
            ranked.loc[big] = valid[big].groupby(group[big]).rank(pct=True)
        if (~big).any():
            ranked.loc[~big] = valid[~big].rank(pct=True)

    # (0, 1] -> [-1, 1], centred so a median name scores zero.
    return (2.0 * ranked - 1.0).reindex(signal.index).rename(signal.name)


def own_history_z(
    history: pd.DataFrame,
    *,
    window: int = 756,
    min_periods: int = 252,
    clip: float = 5.0,
) -> pd.Series:
    """Z-score of the latest observation against each name's own trailing window.

    Default window is three years of trading days, matching the spec's "2-3y".
    Returns NaN for names without enough history rather than scoring them against
    a handful of observations.
    """
    if history.empty:
        return pd.Series(dtype=float)

    tail = history.tail(window)
    if len(tail) < min_periods:
        return pd.Series(np.nan, index=history.columns)

    latest = tail.iloc[-1]
    mu = tail.mean()
    sd = tail.std(ddof=1)
    n_obs = tail.notna().sum()

    z = (latest - mu) / sd.where(sd > 0)
    z = z.where(n_obs >= min_periods)
    return z.clip(-clip, clip)


def dual_baseline(
    signal: pd.Series,
    *,
    history: pd.DataFrame | None = None,
    sector: pd.Series | None = None,
    size_bucket: pd.Series | None = None,
    cross_weight: float = 0.5,
) -> pd.Series:
    """Combine the two baselines into one normalized score.

    When history is unavailable — a newly listed name, or a signal with no
    meaningful time series — this falls back to the cross-sectional leg alone
    rather than dropping the name. Both legs are required *by design*; only one
    is required *to produce a number*, and which happened is worth knowing, so
    callers that care should check `history` themselves.
    """
    cross = cross_sectional_percentile(signal, sector=sector, size_bucket=size_bucket)
    if history is None or history.empty:
        return cross

    own = own_history_z(history).reindex(signal.index)
    own_scaled = (own / 3.0).clip(-1.0, 1.0)  # roughly comparable to the [-1, 1] percentile

    both = own_scaled.notna() & cross.notna()
    out = cross.copy()
    out.loc[both] = cross_weight * cross[both] + (1.0 - cross_weight) * own_scaled[both]
    return out.rename(signal.name)


def apply_materiality(
    normalized: pd.Series, raw: pd.Series, floors: dict[str, float]
) -> pd.Series:
    """Blank out names failing an absolute floor, whatever their normalized score.

    Applied to `raw` in its own units, never to `normalized` — that separation is
    the point. A microcap with a statistically extreme but economically trivial
    reading should not reach the portfolio.
    """
    keep = pd.Series(True, index=normalized.index)
    if (lo := floors.get("min_abs")) is not None:
        keep &= raw.abs() >= lo
    if (lo := floors.get("min_value")) is not None:
        keep &= raw >= lo
    if (hi := floors.get("max_value")) is not None:
        keep &= raw <= hi
    return normalized.where(keep)
