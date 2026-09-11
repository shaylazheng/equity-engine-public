"""Multiple-testing control across a set of candidate strategies.

None of this was inherited — the source repo's entire multiple-testing story was
Deflated Sharpe plus a t >= 3 hurdle. These answer the different question of
whether *the best of many* strategies beats a benchmark, which is exactly the
situation a signal library creates.

All three share one non-negotiable implementation detail: **the bootstrap
resamples the same time indices for every strategy**. Resampling each
independently would destroy the cross-sectional dependence between them and give
a far too optimistic p-value — the strategies in a signal library are correlated,
and pretending otherwise is the whole error being controlled for.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .stats import _auto_block, _circular_blocks

__all__ = ["hansen_spa", "probability_of_backtest_overfitting", "whites_reality_check"]


def _as_matrix(perf: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    clean = perf.dropna()
    if clean.empty:
        raise ValueError("no overlapping observations across strategies")
    if clean.shape[1] < 1:
        raise ValueError("need at least one strategy")
    return clean.to_numpy(dtype=float), list(clean.columns)


def whites_reality_check(
    perf: pd.DataFrame,
    n_boot: int = 2000,
    block: int | None = None,
    seed: int = 17,
) -> dict:
    """White's (2000) Reality Check.

    H0: the *best* strategy in `perf` is no better than the benchmark.

    Parameters
    ----------
    perf:
        Per-period outperformance of each strategy over the benchmark — one
        column per strategy, already differenced. Rows are periods.
    """
    d, names = _as_matrix(perf)
    n, k = d.shape
    block = _auto_block(n) if block is None else block

    means = d.mean(axis=0)
    v_obs = float(np.sqrt(n) * means.max())

    rng = np.random.default_rng(seed)
    v_boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = _circular_blocks(n, block, rng)  # shared across all k strategies
        boot_means = d[idx].mean(axis=0)
        v_boot[b] = np.sqrt(n) * (boot_means - means).max()

    return {
        "statistic": v_obs,
        "pvalue": float((v_boot >= v_obs).mean()),
        "best": names[int(np.argmax(means))],
        "n_strategies": k,
        "n_obs": n,
    }


def hansen_spa(
    perf: pd.DataFrame,
    n_boot: int = 2000,
    block: int | None = None,
    seed: int = 19,
) -> dict:
    """Hansen's (2005) Superior Predictive Ability test.

    Strictly better than White's Reality Check for this use, and the reason both
    are here: White's is badly sensitive to *irrelevant* strategies. Add fifty
    hopeless signals to the set and the RC p-value degrades even though nothing
    about the good one changed. SPA studentizes and recentres so poor performers
    stop diluting the test.
    """
    d, names = _as_matrix(perf)
    n, k = d.shape
    block = _auto_block(n) if block is None else block

    means = d.mean(axis=0)

    # Bootstrap the scale of each strategy's mean, sharing indices.
    rng = np.random.default_rng(seed)
    boot_means = np.empty((n_boot, k), dtype=float)
    for b in range(n_boot):
        boot_means[b] = d[_circular_blocks(n, block, rng)].mean(axis=0)

    omega = np.sqrt(n) * boot_means.std(axis=0, ddof=1)
    omega = np.where(omega <= 0, np.inf, omega)  # a constant strategy contributes nothing

    t_obs = float(np.max(np.maximum(0.0, np.sqrt(n) * means / omega)))

    # Recentre: strategies far enough below zero are treated as truly poor and
    # dropped from the null, which is the whole point of SPA over RC.
    threshold = -(omega / np.sqrt(n)) * np.sqrt(2.0 * np.log(np.log(max(n, 3))))
    g = np.where(means >= threshold, means, 0.0)

    t_boot = np.max(
        np.maximum(0.0, np.sqrt(n) * (boot_means - g) / omega), axis=1
    )

    return {
        "statistic": t_obs,
        "pvalue": float((t_boot >= t_obs).mean()),
        "best": names[int(np.argmax(means / omega))],
        "n_strategies": k,
        "n_obs": n,
        "n_recentred": int((means < threshold).sum()),
    }


def probability_of_backtest_overfitting(
    perf: pd.DataFrame,
    n_partitions: int = 10,
) -> dict:
    """PBO via combinatorially symmetric cross-validation (Bailey et al. 2015).

    Splits the sample into `n_partitions` blocks, and for every way of choosing
    half as in-sample, asks: does the strategy that looked best in-sample hold up
    out-of-sample? PBO is the fraction of splits where the in-sample winner lands
    in the bottom half out-of-sample.

    A PBO near 0.5 means selection carries no information — the "best" strategy
    is whichever one got lucky. This is the number that tells you a horse race
    was decided by noise rather than skill.
    """
    d, _names = _as_matrix(perf)
    n, k = d.shape
    if k < 2:
        raise ValueError(f"PBO needs at least 2 strategies, got {k}")
    if n_partitions % 2 != 0:
        raise ValueError(f"n_partitions must be even, got {n_partitions}")
    if n < n_partitions:
        raise ValueError(f"need at least {n_partitions} observations, got {n}")

    blocks = np.array_split(np.arange(n), n_partitions)
    half = n_partitions // 2

    logits: list[float] = []
    for combo in combinations(range(n_partitions), half):
        is_idx = np.concatenate([blocks[i] for i in combo])
        oos_idx = np.concatenate([blocks[i] for i in range(n_partitions) if i not in combo])

        is_sr = _sharpe_cols(d[is_idx])
        oos_sr = _sharpe_cols(d[oos_idx])

        best = int(np.argmax(is_sr))
        # Relative rank of the in-sample winner, out-of-sample, in (0, 1).
        rank = float((oos_sr <= oos_sr[best]).sum()) / (k + 1)
        rank = min(max(rank, 1.0 / (k + 1)), k / (k + 1))
        logits.append(float(np.log(rank / (1.0 - rank))))

    arr = np.array(logits)
    return {
        "pbo": float((arr <= 0).mean()),
        "n_splits": len(arr),
        "median_logit": float(np.median(arr)),
        "n_strategies": k,
    }


def _sharpe_cols(x: np.ndarray) -> np.ndarray:
    sd = x.std(axis=0, ddof=1)
    sd = np.where(sd <= 0, np.inf, sd)
    return x.mean(axis=0) / sd
