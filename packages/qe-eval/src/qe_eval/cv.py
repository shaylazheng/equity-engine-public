"""Purged cross-validation for overlapping labels.

Ordinary k-fold leaks whenever labels span time. If a signal predicts returns
over the next 21 days, an observation on 1 March and one on 10 March describe
overlapping futures — put one in train and the other in test and the "out-of-
sample" score is contaminated. The fix is to *purge* training observations whose
label span overlaps the test window, and to *embargo* a stretch immediately after
it (serial correlation leaks forward, not backward — the embargo is one-sided).

The inherited implementation took an integer length and applied a symmetric
positional window, so it could not purge on label overlap at all and embargoed
the wrong side. It was also an O(n·k) Python loop, fine at the ~480 monthly bars
it was written for and not at ~14k daily ones. This takes label spans and is
vectorized.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

__all__ = ["cpcv_splits", "label_spans", "n_cpcv_paths", "purged_kfold"]

Split = tuple[np.ndarray, np.ndarray]


def label_spans(index: pd.DatetimeIndex, horizon: int) -> pd.Series:
    """Label resolution times for a fixed forward horizon in observations.

    ``t1[i]`` is when the label for observation ``i`` is fully known. The last
    `horizon` observations have no resolved label and are dropped — including
    them is itself a subtle look-ahead, since their outcome is not yet knowable.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    index = pd.DatetimeIndex(index)
    if len(index) <= horizon:
        raise ValueError(f"need more than {horizon} observations, got {len(index)}")
    return pd.Series(index[horizon:], index=index[:-horizon], name="t1")


def _purge(t1: pd.Series, test_idx: np.ndarray, embargo: int) -> np.ndarray:
    """Training positions surviving purge + forward embargo against one test block."""
    n = len(t1)
    starts = t1.index.to_numpy()
    ends = t1.to_numpy()

    test_start = starts[test_idx[0]]
    test_end = ends[test_idx].max()

    train = np.ones(n, dtype=bool)
    train[test_idx] = False

    # Purge: any observation whose label span touches the test window.
    train &= ~((ends >= test_start) & (starts <= test_end))

    # Embargo: forward-only, immediately after the test block.
    if embargo > 0:
        last = test_idx[-1]
        train[last + 1 : min(n, last + 1 + embargo)] = False

    return np.flatnonzero(train)


def purged_kfold(t1: pd.Series, n_splits: int = 5, embargo_pct: float = 0.01) -> list[Split]:
    """Contiguous k-fold with label purging and a forward embargo.

    Parameters
    ----------
    t1:
        From :func:`label_spans` — indexed by observation time, valued by label
        resolution time.
    embargo_pct:
        Embargo length as a fraction of the sample. Expressed as a fraction
        deliberately: the inherited code took ``cv_embargo_days: 21`` and applied
        it against a monthly index, so "21 days" silently meant 21 months.
    """
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")
    n = len(t1)
    if n < n_splits:
        raise ValueError(f"need at least {n_splits} observations, got {n}")

    embargo = round(n * embargo_pct)
    out: list[Split] = []
    for test_idx in np.array_split(np.arange(n), n_splits):
        if test_idx.size == 0:
            continue
        train_idx = _purge(t1, test_idx, embargo)
        if train_idx.size:
            out.append((train_idx, test_idx))
    return out


def n_cpcv_paths(n_groups: int, n_test: int) -> int:
    """How many distinct backtest paths a CPCV configuration yields."""
    from math import comb

    return comb(n_groups, n_test) * n_test // n_groups


def cpcv_splits(
    t1: pd.Series,
    n_groups: int = 6,
    n_test: int = 2,
    embargo_pct: float = 0.01,
) -> list[Split]:
    """Combinatorial purged CV — every choice of `n_test` groups out of `n_groups`.

    Single-path k-fold gives one backtest path and therefore one Sharpe, with no
    way to see its sampling distribution. CPCV gives many, which is what makes
    the probability of backtest overfitting estimable rather than asserted.
    """
    if not 1 <= n_test < n_groups:
        raise ValueError(f"need 1 <= n_test < n_groups, got n_test={n_test}, n_groups={n_groups}")
    n = len(t1)
    if n < n_groups:
        raise ValueError(f"need at least {n_groups} observations, got {n}")

    embargo = round(n * embargo_pct)
    groups = np.array_split(np.arange(n), n_groups)

    out: list[Split] = []
    for combo in combinations(range(n_groups), n_test):
        test_idx = np.sort(np.concatenate([groups[g] for g in combo]))
        if test_idx.size == 0:
            continue
        # Purge against each contiguous test block separately, then intersect:
        # a training point must survive every block it might overlap.
        train = np.ones(n, dtype=bool)
        train[test_idx] = False
        for g in combo:
            block = groups[g]
            if block.size == 0:
                continue
            keep = np.zeros(n, dtype=bool)
            keep[_purge(t1, block, embargo)] = True
            train &= keep
        train_idx = np.flatnonzero(train)
        if train_idx.size:
            out.append((train_idx, test_idx))
    return out
