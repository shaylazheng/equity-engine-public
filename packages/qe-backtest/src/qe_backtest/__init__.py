"""qe-backtest — event-driven daily engine with costs and tax lots."""

from .costs import CostBreakdown, CostModel
from .engine import BacktestConfig, BacktestResult, run_backtest, sale_tax_cost
from .optimize import (
    AlphaScaleError,
    OptimizerConfig,
    OptimizerResult,
    RiskInputs,
    optimize_weights,
    scores_to_alpha,
)

__version__ = "0.1.0"

__all__ = [
    "AlphaScaleError",
    "BacktestConfig",
    "BacktestResult",
    "CostBreakdown",
    "CostModel",
    "OptimizerConfig",
    "OptimizerResult",
    "RiskInputs",
    "optimize_weights",
    "run_backtest",
    "sale_tax_cost",
    "scores_to_alpha",
]
