"""qe-core — identifiers, calendar, config, and the point-in-time contract."""

from .config import Config, repo_root
from .ledger import AttributionLedger, FlagLeakError, LedgerError
from .panel import AsOfPanel, AsOfView, LookAheadError, PanelSchemaError
from .risk_appetite import PROFILES, RiskProfile, load_profile
from .signal import Signal, SignalContractError, validate_signal

__version__ = "0.1.0"

__all__ = [
    "PROFILES",
    "AsOfPanel",
    "AsOfView",
    "AttributionLedger",
    "Config",
    "FlagLeakError",
    "LedgerError",
    "LookAheadError",
    "PanelSchemaError",
    "RiskProfile",
    "Signal",
    "SignalContractError",
    "load_profile",
    "repo_root",
    "validate_signal",
]
