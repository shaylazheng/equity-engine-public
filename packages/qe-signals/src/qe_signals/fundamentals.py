"""Fundamental signals computed on the raw quarterly frame.

## Why these, and why computed here

The library had three signals. Every multiple-testing statistic the project
relies on — deflated Sharpe, Hansen SPA, PBO — is uninformative at three trials:
DSR reads 1.000 mechanically and PBO's own null at k=3 is 0.73. Growing the
library is the one move that improves both the statistics and the expected
return, so these exist to make the evaluation honest as much as to add alpha.

## What is *not* here, and why

Everything is chosen against two hard constraints:

1. **Computable from `comp.fundq` plus CRSP.** No EDGAR, so nothing needing
   Form 4, 13F or 8-K. And `fundq` as pulled has no `cogs`, `capx`, `xsga`,
   `dvq` or `prstkc`, which rules out gross profitability (Novy-Marx),
   investment-to-capital, and shareholder yield until the pull widens.
2. **Not already a risk factor.** The fitted model carries market, beta, size,
   value, momentum and industry. A signal that *is* one of those neutralizes to
   approximately zero, and scoring it as alpha is a category error — the
   momentum result in `docs/go_no_go.md` is exactly that, and it is why the
   emphasis here falls on profitability, investment and issuance, which are
   orthogonal to all six.

Nothing price-scaled lives here. `fundq` has no market cap, so cash-flow-to-price
and sales-to-price are necessarily computed *after* the as-of merge onto the
trading calendar, where a price is available — see `scripts/go_no_go.py`. The
split is not cosmetic: everything in this module is a lag in fiscal quarters and
must be computed before the merge, while anything divided by price changes every
day and must be computed after it.

## Lags are fiscal quarters, never calendar

Every year-over-year delta is four *observations of the concept* back, taken on
the quarterly frame before any merge onto a trading calendar. Taking it after an
as-of merge would difference against whatever quarter happened to be current
instead of the same quarter a year earlier — and would still return a number.
Most balance-sheet items are strongly seasonal, so a quarter-over-quarter delta
measures the season.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "FUNDAMENTAL_SPECS",
    "FundamentalSpec",
    "compute_all",
    "compute_fundamental",
]

YOY_LAG = 4


def _prep(fundq: pd.DataFrame, permno_col: str, date_col: str) -> pd.DataFrame:
    return fundq.sort_values([permno_col, date_col], kind="stable")


def _col(f: pd.DataFrame, name: str) -> pd.Series:
    if name not in f.columns:
        return pd.Series(np.nan, index=f.index, dtype=float)
    return pd.to_numeric(f[name], errors="coerce").astype(float)


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Divide, treating a non-positive denominator as unknown rather than huge.

    Scaling by total assets or book equity is only meaningful when the scaler is
    positive. A firm with negative book equity divided into a positive profit
    produces a large *negative* ratio that ranks as the worst name in the
    cross-section, when the honest answer is that the ratio does not mean
    anything for that firm.
    """
    return (num / den.where(den > 0)).replace([np.inf, -np.inf], np.nan)


@dataclass(frozen=True)
class FundamentalSpec:
    """One signal: how to compute it, and what the literature expects of it."""

    name: str
    family: str
    #: The sign of the expected long-short spread, fixed from the literature
    #: before any run. A sign chosen after seeing the answer tests nothing.
    expected_sign: int
    citation: str
    rationale: str
    fn: Callable[[pd.DataFrame, str, str], pd.Series] = field(repr=False)
    tier: str = "exploratory"
    #: Set when the fitted risk model already carries this exposure, so the
    #: neutralized result will be structurally suppressed. Reported, not hidden.
    self_factor: str | None = None


# --- profitability -------------------------------------------------------


def _roa(f, p, d):
    """Income before extraordinary items over assets. The plainest quality proxy."""
    return _safe_div(_col(f, "ibq"), _col(f, "atq"))


def _roe(f, p, d):
    return _safe_div(_col(f, "ibq"), _col(f, "ceqq"))


def _asset_turnover(f, p, d):
    """Sales per dollar of assets — the DuPont half of profitability that does
    not depend on margin, so it survives the missing `cogs`."""
    return _safe_div(_col(f, "revtq"), _col(f, "atq"))


def _roa_change(f, p, d):
    """Year-over-year *change* in ROA. The level says who is good; the change
    says who is improving, and the two are far from the same portfolio."""
    roa = _safe_div(_col(f, "ibq"), _col(f, "atq"))
    return roa - roa.groupby(f[p]).shift(YOY_LAG)


# --- investment and issuance ---------------------------------------------


def _asset_growth(f, p, d):
    """Total asset growth. Firms that expand their balance sheet fast
    subsequently underperform — the most robust of the investment anomalies."""
    at = _col(f, "atq")
    prior = at.groupby(f[p]).shift(YOY_LAG)
    return _safe_div(at - prior, prior)


def _net_operating_assets(f, p, d):
    """(Operating assets - operating liabilities) / lagged assets.

    Hirshleifer et al.'s cumulative-accruals measure: a *level*, not a change,
    which is why it survives where period accruals wash out. Built from
    `atq - cheq` and `ltq - dlcq` because the pull has no debt-in-current-
    liabilities detail beyond `dlcq`.
    """
    oa = _col(f, "atq") - _col(f, "cheq")
    ol = _col(f, "ltq") - _col(f, "dlcq")
    prior_at = _col(f, "atq").groupby(f[p]).shift(YOY_LAG)
    return _safe_div(oa - ol, prior_at)


def _net_share_issuance(f, p, d):
    """Log growth in shares outstanding. Firms that issue underperform, firms
    that buy back outperform, and the effect is distinct from the value tilt
    issuance creates."""
    csho = _col(f, "cshoq")
    prior = csho.groupby(f[p]).shift(YOY_LAG)
    ratio = (csho / prior.where(prior > 0)).replace([np.inf, -np.inf], np.nan)
    return np.log(ratio.where(ratio > 0))


def _sales_growth(f, p, d):
    """Fast top-line growth is extrapolated by the market and mean-reverts."""
    rev = _col(f, "revtq")
    prior = rev.groupby(f[p]).shift(YOY_LAG)
    return _safe_div(rev - prior, prior)


# --- balance-sheet quality ------------------------------------------------


def _cash_to_assets(f, p, d):
    return _safe_div(_col(f, "cheq"), _col(f, "atq"))


def _leverage(f, p, d):
    return _safe_div(_col(f, "ltq"), _col(f, "atq"))


def _current_ratio(f, p, d):
    return _safe_div(_col(f, "actq"), _col(f, "lctq"))


# --- earnings quality and surprise ---------------------------------------


def _sue(f, p, d):
    """Standardized unexpected earnings — post-earnings-announcement drift.

    The seasonal random walk: this quarter's earnings minus the same quarter a
    year ago, standardized by the trailing volatility of that difference. Both
    halves matter — the surprise without the standardization is dominated by
    whichever firm is largest.
    """
    ibq = _col(f, "ibq")
    at = _col(f, "atq")
    scaled = _safe_div(ibq, at)
    diff = scaled - scaled.groupby(f[p]).shift(YOY_LAG)
    vol = diff.groupby(f[p]).transform(
        lambda s: s.rolling(8, min_periods=4).std()
    )
    return (diff / vol.where(vol > 0)).replace([np.inf, -np.inf], np.nan)


def _earnings_volatility(f, p, d):
    """Trailing volatility of ROA. Unstable earnings are priced as if they were
    stable, so the negative sign is the claim."""
    scaled = _safe_div(_col(f, "ibq"), _col(f, "atq"))
    return scaled.groupby(f[p]).transform(lambda s: s.rolling(12, min_periods=6).std())


FUNDAMENTAL_SPECS: tuple[FundamentalSpec, ...] = (
    FundamentalSpec(
        "roa", "quality", +1, "Balakrishnan et al. (2010)",
        "Profitable firms earn higher subsequent returns than the market prices in.",
        _roa,
    ),
    FundamentalSpec(
        "roe", "quality", +1, "Haugen & Baker (1996)",
        "Return on equity, the profitability measure scaled by what shareholders own.",
        _roe,
    ),
    FundamentalSpec(
        "asset_turnover", "quality", +1, "Soliman (2008)",
        "The DuPont component independent of margin — usable without `cogs`.",
        _asset_turnover,
    ),
    FundamentalSpec(
        "roa_change", "quality", +1, "Piotroski (2000)",
        "Improving profitability, which is a different portfolio from high profitability.",
        _roa_change,
    ),
    FundamentalSpec(
        "asset_growth", "investment", -1, "Cooper, Gulen & Schill (2008)",
        "Balance-sheet expansion predicts underperformance; the most robust investment effect.",
        _asset_growth, tier="core",
    ),
    FundamentalSpec(
        "net_operating_assets", "investment", -1, "Hirshleifer et al. (2004)",
        "Cumulative accruals as a level, which survives where period accruals wash out.",
        _net_operating_assets,
    ),
    FundamentalSpec(
        "net_share_issuance", "investment", -1, "Pontiff & Woodgate (2008)",
        "Issuers underperform and repurchasers outperform, distinct from the value tilt.",
        _net_share_issuance, tier="core",
    ),
    FundamentalSpec(
        "sales_growth", "investment", -1, "Lakonishok, Shleifer & Vishny (1994)",
        "Extrapolated top-line growth mean-reverts.",
        _sales_growth,
    ),
    FundamentalSpec(
        "cash_to_assets", "quality", +1, "Palazzo (2012)",
        "Cash-rich firms carry riskier growth options and are compensated for it.",
        _cash_to_assets,
    ),
    FundamentalSpec(
        "leverage", "quality", -1, "Bhandari (1988)",
        "Included with a negative sign and low confidence — the literature is genuinely split.",
        _leverage,
    ),
    FundamentalSpec(
        "current_ratio", "quality", +1, "Ou & Penman (1989)",
        "Short-term solvency; a weak effect, carried to test the pipeline more than the claim.",
        _current_ratio,
    ),
    FundamentalSpec(
        "sue", "earnings", +1, "Bernard & Thomas (1989)",
        "Post-earnings-announcement drift on a seasonal random walk.",
        _sue, tier="core",
    ),
    FundamentalSpec(
        "earnings_volatility", "quality", -1, "Francis et al. (2004)",
        "Unstable earnings are priced as if stable.",
        _earnings_volatility,
    ),
)


def compute_fundamental(
    fundq: pd.DataFrame,
    spec: FundamentalSpec,
    *,
    permno_col: str = "permno",
    date_col: str = "datadate",
) -> pd.Series:
    """One signal, on the raw quarterly frame, aligned to `fundq`'s index."""
    f = _prep(fundq, permno_col, date_col)
    out = spec.fn(f, permno_col, date_col)
    out = pd.Series(out, index=f.index, dtype=float)
    return out.reindex(fundq.index).rename(spec.name)


def compute_all(
    fundq: pd.DataFrame,
    *,
    permno_col: str = "permno",
    date_col: str = "datadate",
    specs: tuple[FundamentalSpec, ...] = FUNDAMENTAL_SPECS,
) -> pd.DataFrame:
    """Every fundamental signal as one frame, aligned to `fundq`'s index."""
    return pd.DataFrame(
        {
            s.name: compute_fundamental(
                fundq, s, permno_col=permno_col, date_col=date_col
            )
            for s in specs
        },
        index=fundq.index,
    )
