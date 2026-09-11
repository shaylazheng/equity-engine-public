"""Cross-sectional factor risk model.

"Connecting the factors" *is* the risk model. Without one you cannot tell a
genuine insider signal from an accidental small-cap-value tilt — and since roughly
every published anomaly correlates with size and value to some degree, that
distinction is most of what separates a real result from a repackaged one.

The estimation is Barra-style: each day, regress the cross-section of returns on
standardized style exposures and industry dummies. The regression *coefficients*
are the factor returns; the residuals are specific returns. Note the direction —
exposures are known and factor returns are estimated, which is the opposite of a
time-series (Fama-French) model where the factor returns are given and betas are
estimated.

Weights are proportional to sqrt(market cap), the standard choice: it downweights
microcap noise in estimation without letting mega-caps determine the whole fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["RiskModelResult", "fit_cross_sectional", "industry_dummies"]


@dataclass(frozen=True)
class RiskModelResult:
    """Output of a cross-sectional fit."""

    #: date x factor — the estimated factor returns.
    factor_returns: pd.DataFrame
    #: date x permno — residuals, i.e. the part not explained by the factors.
    specific_returns: pd.DataFrame
    #: Cross-sectional R^2 per date.
    r2: pd.Series
    #: Names actually used each date, after dropping missing data.
    coverage: pd.Series

    @property
    def factors(self) -> list[str]:
        return list(self.factor_returns.columns)

    def specific_risk(self, halflife: float = 60.0) -> pd.Series:
        """EWMA idiosyncratic vol per name, annualized."""
        var = self.specific_returns.pow(2).ewm(halflife=halflife, min_periods=20).mean()
        return var.iloc[-1].pow(0.5) * np.sqrt(252)

    def __repr__(self) -> str:
        return (
            f"<RiskModelResult {len(self.factor_returns)} dates, "
            f"{len(self.factors)} factors, mean R2 {self.r2.mean():.3f}>"
        )


def industry_dummies(industry: pd.Series, *, drop_first: bool = True) -> pd.DataFrame:
    """One-hot industry exposures indexed by permno.

    `drop_first` avoids perfect collinearity with the implicit market intercept.
    Keeping every dummy *and* an intercept makes X'X singular, which is the
    classic way a Barra-style fit blows up on day one.
    """
    dummies = pd.get_dummies(industry.astype("category"), prefix="ind", dtype=float)
    if drop_first and dummies.shape[1] > 1:
        dummies = dummies.iloc[:, 1:]
    return dummies


def _weights(mktcap: pd.Series | None, names: pd.Index) -> np.ndarray:
    if mktcap is None:
        return np.ones(len(names))
    w = mktcap.reindex(names).astype(float)
    w = w.where(w > 0).fillna(w[w > 0].median() if (w > 0).any() else 1.0)
    root = np.sqrt(w.to_numpy())
    return root / root.mean()


def fit_cross_sectional(
    returns: pd.DataFrame,
    exposures: pd.DataFrame,
    *,
    mktcap: pd.DataFrame | None = None,
    min_names: int = 20,
) -> RiskModelResult:
    """Fit the model date by date.

    Parameters
    ----------
    returns:
        date x permno.
    exposures:
        Either permno x factor (static, broadcast across dates) or a MultiIndexed
        (date, permno) x factor frame for time-varying exposures.
    mktcap:
        date x permno, for sqrt-cap weighting. Equal weights if omitted.
    min_names:
        Dates with fewer usable names are skipped rather than fitted badly. A
        cross-sectional regression on a handful of names is noise with a
        coefficient attached.
    """
    time_varying = isinstance(exposures.index, pd.MultiIndex)
    factors = list(exposures.columns)

    f_rows: dict[pd.Timestamp, np.ndarray] = {}
    s_rows: dict[pd.Timestamp, pd.Series] = {}
    r2_rows: dict[pd.Timestamp, float] = {}
    cov_rows: dict[pd.Timestamp, int] = {}

    for date, row in returns.iterrows():
        X_all = exposures.loc[date] if time_varying and date in exposures.index else exposures
        if time_varying and date not in exposures.index.get_level_values(0):
            continue

        names = row.dropna().index.intersection(X_all.dropna().index)
        if len(names) < min_names:
            continue

        y = row.reindex(names).to_numpy(dtype=float)
        X = X_all.reindex(names).to_numpy(dtype=float)
        w = _weights(mktcap.loc[date] if mktcap is not None else None, names)

        sw = np.sqrt(w)
        Xw, yw = X * sw[:, None], y * sw

        beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        fitted = X @ beta
        resid = y - fitted

        ss_res = float(np.sum(w * resid**2))
        ss_tot = float(np.sum(w * (y - np.average(y, weights=w)) ** 2))

        f_rows[date] = beta
        s_rows[date] = pd.Series(resid, index=names)
        r2_rows[date] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        cov_rows[date] = len(names)

    if not f_rows:
        raise ValueError(
            f"no date had at least {min_names} usable names — check that returns and "
            "exposures share an index"
        )

    return RiskModelResult(
        factor_returns=pd.DataFrame.from_dict(f_rows, orient="index", columns=factors),
        specific_returns=pd.DataFrame(s_rows).T,
        r2=pd.Series(r2_rows, name="r2"),
        coverage=pd.Series(cov_rows, name="n_names"),
    )
