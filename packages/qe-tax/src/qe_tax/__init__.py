"""qe-tax — personalized, headroom-aware capital gains tax modelling.

Not tax advice. These are inputs to a backtest; a CPA validates an actual return.
"""

from .brackets import BracketTable, UnverifiedStatusError, load_brackets, tax_from_bands
from .engine import Netting, TaxBreakdown, TaxEngine, TaxProfile, net_gains
from .lots import LONG_TERM_DAYS, WASH_SALE_DAYS, Lot, LotBook, SaleResult

__version__ = "0.1.0"

__all__ = [
    "LONG_TERM_DAYS",
    "WASH_SALE_DAYS",
    "BracketTable",
    "Lot",
    "LotBook",
    "Netting",
    "SaleResult",
    "TaxBreakdown",
    "TaxEngine",
    "TaxProfile",
    "UnverifiedStatusError",
    "load_brackets",
    "net_gains",
    "tax_from_bands",
]
