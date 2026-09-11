"""Turning a target portfolio into a trade list.

Deliberately stops at the list. Nothing here places an order — execution is out
of scope for the whole project, and a trade list a human reads before acting is
the correct output for real money running off a model this young.

Every row carries *why*: the score that drove it, the tax consequence of the
sale, and whether the name is blocked. A trade list you cannot interrogate is one
you should not execute.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from qe_tax.engine import TaxEngine
from qe_tax.lots import LotBook

__all__ = ["TradeList", "drift", "generate_trade_list"]


@dataclass(frozen=True)
class TradeList:
    rows: pd.DataFrame
    as_of: pd.Timestamp
    equity: float
    #: Names skipped because selling them would trigger a wash sale.
    blocked: tuple[int, ...] = ()
    #: Remaining 0% long-term band, if a tax engine was supplied.
    headroom: float | None = None

    @property
    def buys(self) -> pd.DataFrame:
        return self.rows.loc[self.rows["shares"] > 0]

    @property
    def sells(self) -> pd.DataFrame:
        return self.rows.loc[self.rows["shares"] < 0]

    def notional_traded(self) -> float:
        return float(self.rows["notional"].abs().sum())

    def turnover(self) -> float:
        return self.notional_traded() / self.equity if self.equity > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"<TradeList {self.as_of.date()}: {len(self.buys)} buys, {len(self.sells)} sells, "
            f"turnover {self.turnover():.1%}>"
        )


def generate_trade_list(
    target_weights: pd.Series,
    prices: pd.Series,
    equity: float,
    *,
    book: LotBook,
    as_of,
    scores: pd.Series | None = None,
    tax_engine: TaxEngine | None = None,
    min_trade_value: float = 100.0,
) -> TradeList:
    """Compare the target portfolio to what is actually held.

    Wash-sale-blocked names are excluded from *buys* and reported separately
    rather than silently dropped — a name missing from a trade list with no
    explanation is indistinguishable from a bug.
    """
    as_of = pd.Timestamp(as_of)
    blocked = sorted(book.blocked_names(as_of))

    headroom = None
    ytd_short, ytd_long = book.realized_gains(as_of.year)
    if tax_engine is not None:
        headroom = tax_engine.headroom(short_gain=ytd_short, long_gain=ytd_long)

    names = set(target_weights.index) | set(book.holdings())
    rows = []

    for permno in sorted(names):
        price = prices.get(permno, np.nan)
        if not np.isfinite(price) or price <= 0:
            continue

        want_weight = float(target_weights.get(permno, 0.0))
        if permno in blocked and want_weight > 0:
            want_weight = 0.0

        held = book.shares(permno)
        want_shares = want_weight * equity / price
        delta = want_shares - held
        notional = delta * price

        if abs(notional) < min_trade_value:
            continue

        tax_impact = 0.0
        gain = 0.0
        if delta < 0 and held > 0:
            qty = min(-delta, held)
            gain, long_term = _estimate_gain(book, permno, qty, price, as_of)
            if tax_engine is not None:
                tax_impact = tax_engine.tax_on_realization(
                    gain, long_term=long_term, short_gain=ytd_short, long_gain=ytd_long
                )

        rows.append(
            {
                "permno": permno,
                "action": "BUY" if delta > 0 else "SELL",
                "shares": round(delta, 4),
                "price": price,
                "notional": notional,
                "current_weight": held * price / equity if equity > 0 else 0.0,
                "target_weight": want_weight,
                "score": float(scores.get(permno, np.nan)) if scores is not None else np.nan,
                "est_gain": gain,
                "est_tax": tax_impact,
                "blocked": permno in blocked,
            }
        )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("notional", key=abs, ascending=False).reset_index(drop=True)

    return TradeList(
        rows=frame, as_of=as_of, equity=equity,
        blocked=tuple(blocked), headroom=headroom,
    )


def _estimate_gain(
    book: LotBook, permno: int, qty: float, price: float, as_of: pd.Timestamp
) -> tuple[float, bool]:
    """Gain and holding-period character if `qty` were sold, without mutating the book."""
    lots = sorted(book.lots(permno), key=lambda x: x.basis_per_share, reverse=True)
    remaining, gain, long_shares = qty, 0.0, 0.0
    for lot in lots:
        if remaining <= 0:
            break
        take = min(lot.shares, remaining)
        gain += take * (price - lot.basis_per_share)
        if lot.is_long_term(as_of):
            long_shares += take
        remaining -= take
    return gain, long_shares >= qty / 2


def drift(target_weights: pd.Series, actual_weights: pd.Series) -> pd.Series:
    """Per-name gap between intent and reality, largest first.

    The Phase 9 monitor. A live portfolio drifts from its target through price
    moves alone, and knowing *where* it drifted is what distinguishes an
    acceptable gap from a failed fill nobody noticed.
    """
    all_names = target_weights.index.union(actual_weights.index)
    gap = actual_weights.reindex(all_names).fillna(0.0) - target_weights.reindex(
        all_names
    ).fillna(0.0)
    return gap.reindex(gap.abs().sort_values(ascending=False).index).rename("drift")
