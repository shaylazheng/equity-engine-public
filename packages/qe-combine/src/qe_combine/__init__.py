"""qe-combine — signal combination that emits its own attribution ledger."""

from .composite import combine, orthogonalize_signals, shrink_weights

__version__ = "0.1.0"

__all__ = ["combine", "orthogonalize_signals", "shrink_weights"]
