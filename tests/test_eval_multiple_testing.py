"""Reality Check, SPA, and PBO — is the best of many actually good?"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from qe_eval.multiple_testing import (
    hansen_spa,
    probability_of_backtest_overfitting,
    whites_reality_check,
)


def _noise(n_strategies: int, n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Pure noise: nothing outperforms the benchmark."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(0.0, 0.01, size=(n, n_strategies)),
        columns=[f"s{i}" for i in range(n_strategies)],
    )


def _one_winner(n_strategies: int, edge: float = 0.004, n: int = 500, seed: int = 1):
    df = _noise(n_strategies, n, seed)
    df["winner"] = np.random.default_rng(seed + 99).normal(edge, 0.01, n)
    return df


# -- White's Reality Check ------------------------------------------------


def test_reality_check_does_not_reject_pure_noise():
    out = whites_reality_check(_noise(20), n_boot=500)
    assert out["pvalue"] > 0.10, f"rejected on noise (p={out['pvalue']:.3f})"


def test_reality_check_finds_a_real_winner():
    out = whites_reality_check(_one_winner(20), n_boot=500)
    assert out["pvalue"] < 0.05
    assert out["best"] == "winner"


def test_reality_check_gets_harder_with_more_candidates():
    """The multiple-testing penalty: the same winner buried in more noise is
    harder to call, which is the entire point."""
    few = whites_reality_check(_one_winner(3, edge=0.0012, seed=5), n_boot=800)
    many = whites_reality_check(_one_winner(60, edge=0.0012, seed=5), n_boot=800)
    assert many["pvalue"] >= few["pvalue"]


# -- Hansen SPA -----------------------------------------------------------


def test_spa_does_not_reject_pure_noise():
    out = hansen_spa(_noise(20), n_boot=500)
    assert out["pvalue"] > 0.10


def test_spa_finds_a_real_winner():
    out = hansen_spa(_one_winner(20), n_boot=500)
    assert out["pvalue"] < 0.05
    assert out["best"] == "winner"


def test_spa_is_less_diluted_by_hopeless_strategies_than_rc():
    """SPA's advantage over RC, and the reason both are implemented.

    Adding strategies that are clearly *bad* should not weaken the evidence for a
    good one. RC is sensitive to them; SPA recentres them out of the null.
    """
    base = _one_winner(5, edge=0.0015, seed=11)

    rng = np.random.default_rng(3)
    hopeless = pd.DataFrame(
        rng.normal(-0.006, 0.01, size=(len(base), 40)),
        columns=[f"bad{i}" for i in range(40)],
        index=base.index,
    )
    padded = pd.concat([base, hopeless], axis=1)

    rc_hit = whites_reality_check(padded, n_boot=800)["pvalue"] - \
        whites_reality_check(base, n_boot=800)["pvalue"]
    spa_hit = hansen_spa(padded, n_boot=800)["pvalue"] - \
        hansen_spa(base, n_boot=800)["pvalue"]

    assert spa_hit <= rc_hit + 1e-9, (
        f"SPA degraded more than RC when hopeless strategies were added "
        f"(spa +{spa_hit:.4f} vs rc +{rc_hit:.4f})"
    )
    assert hansen_spa(padded, n_boot=800)["n_recentred"] > 0


def test_spa_reports_how_many_were_recentred():
    rng = np.random.default_rng(4)
    df = pd.DataFrame(
        np.column_stack([
            rng.normal(0.003, 0.01, 400),
            rng.normal(-0.02, 0.01, 400),
        ]),
        columns=["good", "awful"],
    )
    assert hansen_spa(df, n_boot=400)["n_recentred"] >= 1


# -- PBO ------------------------------------------------------------------


def test_pbo_is_near_a_half_when_selection_is_noise():
    """No strategy is better, so the in-sample winner is a coin flip out-of-sample."""
    out = probability_of_backtest_overfitting(_noise(12, n=600, seed=7), n_partitions=8)
    assert 0.3 < out["pbo"] < 0.7, f"PBO {out['pbo']:.2f} — expected ~0.5 on noise"


def test_pbo_is_low_when_one_strategy_genuinely_dominates():
    out = probability_of_backtest_overfitting(
        _one_winner(12, edge=0.006, n=600, seed=2), n_partitions=8
    )
    assert out["pbo"] < 0.2, f"PBO {out['pbo']:.2f} — a real winner should persist OOS"


def test_pbo_rejects_odd_partition_counts():
    with pytest.raises(ValueError, match="even"):
        probability_of_backtest_overfitting(_noise(4), n_partitions=7)


def test_pbo_needs_multiple_strategies():
    with pytest.raises(ValueError, match="at least 2"):
        probability_of_backtest_overfitting(_noise(1), n_partitions=8)


# -- shared resampling ----------------------------------------------------


def test_results_are_deterministic_given_a_seed():
    df = _one_winner(10)
    assert whites_reality_check(df, n_boot=300, seed=5) == whites_reality_check(
        df, n_boot=300, seed=5
    )
