"""qe-signals — the signal library, its registry, and dual-baseline normalization."""

from .accruals import Accruals, accruals_from_fundq
from .base import REGISTRY, BaseSignal, SignalRegistry
from .fundamentals import (
    FUNDAMENTAL_SPECS,
    FundamentalSpec,
    compute_all,
    compute_fundamental,
)
from .normalize import (
    apply_materiality,
    cross_sectional_percentile,
    dual_baseline,
    own_history_z,
    size_buckets,
)
from .price import PRICE_SPECS, long_term_reversal, short_term_reversal
from .reference import EarningsYield, Momentum12_1

__version__ = "0.1.0"

__all__ = [
    "REGISTRY",
    "Accruals",
    "BaseSignal",
    "EarningsYield",
    "Momentum12_1",
    "SignalRegistry",
    "accruals_from_fundq",
    "apply_materiality",
    "cross_sectional_percentile",
    "dual_baseline",
    "own_history_z",
    "size_buckets",
]

# Balance-sheet construction is the default: it is Sloan's own, and the
# cash-flow arm cannot reach his 1962-1991 sample.
REGISTRY.register(Accruals())
