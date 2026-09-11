"""Price-based signals that are not already risk factors.

Momentum lives in `reference.py`; beta, idiosyncratic volatility, Amihud
illiquidity and turnover live in `qe-risk`, because a signal that *is* a risk
factor neutralizes to approximately zero and scoring it as alpha is a category
error. What is left on the price side, and genuinely orthogonal to the fitted
model, is the two reversal horizons.

Both are the same shape as momentum with a different window, and both are real
effects with opposite sign to it — which is precisely why momentum skips the most
recent month. A "momentum" signal that includes it is measuring the two against
each other.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["PRICE_SPECS", "long_term_reversal", "short_term_reversal"]


def short_term_reversal(monthly: pd.DataFrame, *, ret_col: str = "ret") -> pd.Series:
    """Last month's return, negated so that a high score means "buy".

    Jegadeesh (1990). Microstructure as much as behaviour — bid-ask bounce and
    liquidity provision both push last month's losers up — which is why it is the
    one signal here most likely to be eaten by trading costs. Carried anyway,
    because measuring that is the point of having a cost model.
    """
    return (-pd.to_numeric(monthly[ret_col], errors="coerce")).rename(
        "short_term_reversal"
    )


def long_term_reversal(
    monthly: pd.DataFrame,
    *,
    permno_col: str = "permno",
    ret_col: str = "ret",
    start: int = 60,
    end: int = 13,
) -> pd.Series:
    """Return from 60 months ago to 13 months ago, negated.

    De Bondt & Thaler (1985). The window deliberately *stops* at 13 months so it
    cannot overlap 12-1 momentum: the two have opposite signs, and an overlapping
    window measures their difference rather than either one.
    """
    m = monthly.sort_values([permno_col, "ym"], kind="stable")
    logret = np.log1p(pd.to_numeric(m[ret_col], errors="coerce").clip(-0.99, None))
    cum = logret.groupby(m[permno_col], sort=False).cumsum()
    g = cum.groupby(m[permno_col], sort=False)
    past = np.expm1(g.shift(end) - g.shift(start))

    n_hist = m.groupby(permno_col, sort=False)[ret_col].transform("cumcount")
    past = past.where(n_hist >= start)
    return (-past).reindex(monthly.index).rename("long_term_reversal")


#: (name, family, expected sign of the long-short spread, citation).
#: Signs are the literature's, fixed before any run — both are *already negated*
#: above, so a positive sign here means the negated series predicts positively.
PRICE_SPECS: tuple[tuple[str, str, int, str], ...] = (
    ("short_term_reversal", "reversal", +1, "Jegadeesh (1990)"),
    ("long_term_reversal", "reversal", +1, "De Bondt & Thaler (1985)"),
)
