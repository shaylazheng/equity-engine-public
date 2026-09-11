"""qe-risk — cross-sectional factor model, covariance, and orthogonalization."""

from .covariance import (
    build_factor_covariance,
    eigenvalue_adjust,
    ewma_cov,
    ledoit_wolf_intensity,
    newey_west_adjust,
    shrink_to_diagonal,
)
from .industry import (
    INDUSTRIES,
    constrain_industries,
    ff12_from_sic,
    industry_coverage,
    industry_dummies_ff12,
)
from .model import RiskModelResult, fit_cross_sectional, industry_dummies
from .neutralize import neutralize, neutralize_panel, standardize, winsorize

__version__ = "0.1.0"

__all__ = [
    "INDUSTRIES",
    "RiskModelResult",
    "build_factor_covariance",
    "constrain_industries",
    "eigenvalue_adjust",
    "ewma_cov",
    "ff12_from_sic",
    "fit_cross_sectional",
    "industry_coverage",
    "industry_dummies",
    "industry_dummies_ff12",
    "ledoit_wolf_intensity",
    "neutralize",
    "neutralize_panel",
    "newey_west_adjust",
    "shrink_to_diagonal",
    "standardize",
    "winsorize",
]
