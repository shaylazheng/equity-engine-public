"""qe-eval — statistics, purged CV, the trial registry, and multiple-testing control."""

from .cv import cpcv_splits, label_spans, n_cpcv_paths, purged_kfold
from .multiple_testing import (
    hansen_spa,
    probability_of_backtest_overfitting,
    whites_reality_check,
)
from .registry import TrialRegistry, config_hash
from .stats import (
    ann_sharpe,
    block_bootstrap_pvalue,
    deflated_sharpe,
    expected_max_sharpe,
    newey_west_alpha,
    placebo_distribution,
    probabilistic_sharpe,
    tstat_mean,
)

__version__ = "0.1.0"

__all__ = [
    "TrialRegistry",
    "ann_sharpe",
    "block_bootstrap_pvalue",
    "config_hash",
    "cpcv_splits",
    "deflated_sharpe",
    "expected_max_sharpe",
    "hansen_spa",
    "label_spans",
    "n_cpcv_paths",
    "newey_west_alpha",
    "placebo_distribution",
    "probabilistic_sharpe",
    "probability_of_backtest_overfitting",
    "purged_kfold",
    "tstat_mean",
    "whites_reality_check",
]
