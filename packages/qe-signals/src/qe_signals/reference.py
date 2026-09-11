"""Reference signals for the Phase 2a replication gate.

These three exist because the gate needs them. The original plan put the gate in
Phase 2 and the signals in Phase 6, which meant Phase 2's acceptance test could
not run until four phases after Phase 2 — a circular dependency, so they moved
here.

They are also the three most heavily replicated results in the cross-sectional
literature, which is exactly what makes them suitable as an instrument check: if
the engine cannot reproduce momentum, value, and accruals, nothing novel it
produces later means anything.
"""

from __future__ import annotations

import datetime as dt
from typing import ClassVar

import numpy as np
import pandas as pd
from qe_core.panel import AsOfView

from .base import REGISTRY, BaseSignal

__all__ = ["EarningsYield", "Momentum12_1"]

TRADING_DAYS_YEAR = 252
SKIP_MONTH = 21


class Momentum12_1(BaseSignal):
    """Twelve-month return, skipping the most recent month (Jegadeesh-Titman).

    The skip is not decoration. One-month reversal is a distinct and opposing
    effect, so including the most recent month contaminates momentum with its own
    negative. Nearly every published momentum result skips it, and a "momentum"
    signal that does not is measuring something else.
    """

    name = "momentum_12_1"
    family = "momentum"
    tier = "core"
    evidence = "scored"
    requires: ClassVar[list[str]] = ["prc"]
    min_date = dt.date(1926, 1, 1)
    expected_sign = 1
    horizon = 21
    native_freq = "daily"
    neutralize: ClassVar[list[str]] = ["market", "size", "value"]
    citation = "Jegadeesh & Titman (1993)"

    def compute(self, view: AsOfView) -> pd.Series:
        if not self.has_inputs(view):
            return pd.Series(dtype=float, name=self.name)
        prices = view.series("prc")
        if len(prices) < TRADING_DAYS_YEAR + 1:
            return pd.Series(dtype=float, name=self.name)

        start = prices.iloc[-(TRADING_DAYS_YEAR + 1)]
        end = prices.iloc[-(SKIP_MONTH + 1)]
        mom = (end / start.where(start > 0)) - 1.0
        return mom.replace([np.inf, -np.inf], np.nan).rename(self.name)


class EarningsYield(BaseSignal):
    """Earnings to price — the value leg of the gate.

    Deliberately earnings yield rather than book-to-market: the plan's Core tier
    keeps EBIT/EV and shareholder yield as the distinct value measures, and the
    seven classic yields run 0.7-0.9 correlated with each other, so testing all
    of them buys seven trials and roughly one signal.

    `native_freq` is quarterly: the numerator only changes when a filing lands, so
    recomputing daily across a 150M-row panel repeats an unchanged number about
    sixty times a quarter.
    """

    name = "earnings_yield"
    family = "value"
    tier = "core"
    evidence = "scored"
    requires: ClassVar[list[str]] = ["ni", "mktcap"]
    min_date = dt.date(1963, 7, 1)
    expected_sign = 1
    horizon = 63
    native_freq = "quarterly"
    neutralize: ClassVar[list[str]] = ["market", "size"]
    materiality: ClassVar[dict] = {"min_value": -10.0, "max_value": 10.0}
    citation = "Basu (1977)"

    def compute(self, view: AsOfView) -> pd.Series:
        if not self.has_inputs(view):
            return pd.Series(dtype=float, name=self.name)
        wide = view.pivot(["ni", "mktcap"])
        if wide.empty:
            return pd.Series(dtype=float, name=self.name)

        cap = wide["mktcap"].where(wide["mktcap"] > 0)
        return (wide["ni"] / cap).replace([np.inf, -np.inf], np.nan).rename(self.name)



REGISTRY.register(Momentum12_1())
REGISTRY.register(EarningsYield())
