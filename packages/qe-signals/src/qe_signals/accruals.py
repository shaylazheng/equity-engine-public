"""Accruals — both constructions, as one signal.

Sloan (1996) showed that earnings which cash flow does not back are less
persistent, and that the market prices them as if they were. There are two ways
to measure it, and the choice is forced by data availability rather than taste:

**Balance-sheet** (Sloan's own)::

    ACC = (dCA - dCash) - (dCL - dSTD - dTP) - Dep

Working-capital accruals net of depreciation, scaled by average total assets.
Sloan used this over 1962-1991 because cash-flow statements did not exist for
most of that span.

**Cash-flow**::

    ACC = (NI - CFO) / average total assets

Cleaner and the modern standard — but `oancf` only begins 1986-12, since SFAS 95
first mandated cash-flow statements for FY1988+. Measured on the real pull, the
cash-flow arm yields a 1987-2025 sample, entirely outside Sloan's and mostly
after his paper, where the effect comes out at t=-0.91.

They are **one signal with a flag**, not two, so the trial registry charges one
trial rather than two for measurements that largely agree where both exist.

Deltas are **year-over-year** (four quarters back), never quarter-over-quarter.
Working capital is strongly seasonal for most firms; a QoQ delta measures the
season rather than the accrual.
"""

from __future__ import annotations

import datetime as dt
from typing import ClassVar, Literal

import numpy as np
import pandas as pd
from qe_core.panel import AsOfView

from .base import BaseSignal

__all__ = ["YOY_LAG", "Accruals", "Construction", "accruals_from_fundq"]

Construction = Literal["balance_sheet", "cash_flow"]

#: Periods back for the year-over-year delta, in *observations of the concept* —
#: 4 for quarterly fundamentals (the real `fundq` panel), 1 for annual. It is a
#: parameter rather than a constant because the lag is frequency-dependent and a
#: silent mismatch differences against the wrong year: on annual data a lag of 4
#: reaches back four years, which still returns a number.
YOY_LAG = 4

BALANCE_SHEET_INPUTS = ("actq", "cheq", "lctq", "dlcq", "txpq", "dpq", "atq")
CASH_FLOW_INPUTS = ("niq", "oancfy_q", "atq")


def accruals_from_fundq(
    fundq: pd.DataFrame,
    construction: Construction = "balance_sheet",
    *,
    permno_col: str = "permno",
    date_col: str = "datadate",
    yoy_lag: int = YOY_LAG,
) -> pd.Series:
    """Accruals per firm-quarter, computed on the raw quarterly frame.

    Computed *before* any merge onto a trading calendar: the year-over-year lag
    is a lag in fiscal quarters, and taking it after an as-of merge onto monthly
    dates would difference against whatever quarter happened to be current, not
    against the same quarter a year earlier.

    Returns a Series aligned to `fundq`'s index.
    """
    f = fundq.sort_values([permno_col, date_col], kind="stable")
    grp = f.groupby(permno_col, sort=False)

    def lag(col: str) -> pd.Series:
        return grp[col].shift(yoy_lag) if col in f.columns else pd.Series(np.nan, index=f.index)

    def delta(col: str) -> pd.Series:
        if col not in f.columns:
            return pd.Series(np.nan, index=f.index)
        return f[col] - lag(col)

    at = f["atq"] if "atq" in f.columns else pd.Series(np.nan, index=f.index)
    avg_at = (at + lag("atq")) / 2.0
    avg_at = avg_at.where(avg_at > 0)

    if construction == "balance_sheet":
        missing = [c for c in ("actq", "cheq", "lctq", "dpq", "atq") if c not in f.columns]
        if missing:
            raise KeyError(
                f"balance-sheet accruals needs {missing} — add them to FUNDQ_SQL and re-pull"
            )
        # `txpq` (income taxes payable) and `dlcq` (debt in current liabilities)
        # are frequently unreported and small relative to the other terms.
        # Treating a missing delta as zero costs far less coverage than dropping
        # the firm-quarter outright.
        acc = (
            (delta("actq") - delta("cheq"))
            - (delta("lctq") - delta("dlcq").fillna(0.0) - delta("txpq").fillna(0.0))
            - f["dpq"]
        )
    elif construction == "cash_flow":
        missing = [c for c in CASH_FLOW_INPUTS if c not in f.columns]
        if missing:
            raise KeyError(f"cash-flow accruals needs {missing}")
        acc = f["niq"] - f["oancfy_q"]
    else:
        raise ValueError(f"unknown construction {construction!r}")

    out = (acc / avg_at).replace([np.inf, -np.inf], np.nan)
    return out.reindex(fundq.index)


class Accruals(BaseSignal):
    """Sloan's accrual anomaly. High accruals predict underperformance."""

    name = "accruals"
    family = "quality"
    tier = "core"
    evidence = "scored"
    requires: ClassVar[list[str]] = ["ni", "at"]
    min_date = dt.date(1963, 7, 1)
    expected_sign = -1  # high accruals -> *under*performance
    horizon = 252
    native_freq = "quarterly"
    neutralize: ClassVar[list[str]] = ["market", "size", "value"]
    citation = "Sloan (1996)"

    def __init__(
        self, construction: Construction = "balance_sheet", *, yoy_lag: int = YOY_LAG
    ) -> None:
        if construction not in ("balance_sheet", "cash_flow"):
            raise ValueError(f"unknown construction {construction!r}")
        if yoy_lag < 1:
            raise ValueError(f"yoy_lag must be >= 1, got {yoy_lag}")
        self.construction: Construction = construction
        #: 4 for quarterly fundamentals, 1 for annual. See YOY_LAG.
        self.yoy_lag = yoy_lag

    def compute(self, view: AsOfView) -> pd.Series:
        """Accruals for the view's date.

        A view holds the whole knowable past, so `series()` supplies the fiscal
        history the year-over-year lag needs — the same quantity
        :func:`accruals_from_fundq` computes, taken at one date.
        """
        if not self.has_inputs(view):
            return pd.Series(dtype=float, name=self.name)

        available = set(view.concepts())

        def latest_and_lag(concept: str) -> tuple[pd.Series, pd.Series] | None:
            if concept not in available:
                return None
            hist = view.series(concept)
            if len(hist) <= self.yoy_lag:
                return None
            return hist.iloc[-1], hist.iloc[-(self.yoy_lag + 1)]

        at_pair = latest_and_lag("at")
        if at_pair is None:
            return pd.Series(dtype=float, name=self.name)
        at_now, at_then = at_pair
        avg_at = ((at_now + at_then) / 2.0).where(lambda s: s > 0)

        if self.construction == "cash_flow":
            wide = view.pivot(["ni", "oancf"])
            if wide.empty or "oancf" not in wide.columns:
                return pd.Series(dtype=float, name=self.name)
            acc = wide["ni"] - wide["oancf"]
        else:
            terms = {c: latest_and_lag(c) for c in ("act", "che", "lct", "dlc", "txp")}
            if any(terms[c] is None for c in ("act", "che", "lct")):
                return pd.Series(dtype=float, name=self.name)

            def d(c: str) -> pd.Series:
                pair = terms[c]
                if pair is None:
                    return pd.Series(0.0, index=at_now.index)
                return (pair[0] - pair[1]).fillna(0.0)

            dep = view.field("dp") if "dp" in available else pd.Series(0.0, index=at_now.index)
            acc = (d("act") - d("che")) - (d("lct") - d("dlc") - d("txp")) - dep

        return (acc / avg_at).replace([np.inf, -np.inf], np.nan).rename(self.name)

    def __repr__(self) -> str:
        return (f"<Accruals construction={self.construction!r} "
                f"yoy_lag={self.yoy_lag} tier={self.tier}>")
