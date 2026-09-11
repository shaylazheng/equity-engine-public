"""Per-lot accounting: holding periods, lot selection, and wash sales.

Long-only taxable investing is path-dependent in a way a vectorized backtest
cannot express. Which *lot* you sell changes the tax; how long you held it
changes the rate; and selling at a loss blocks repurchase for 30 days. All three
depend on the order things happened, which is why the backtester is an explicit
event loop rather than a returns multiplier.

The 365-day boundary is the one with real teeth: crossing it moves a gain from
ordinary rates to the long-term schedule, which at low income can mean moving
from 12% to **zero**. That single fact should change rebalance timing, and an
engine that ignores holding periods cannot see it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

__all__ = ["WASH_SALE_DAYS", "Lot", "LotBook", "LotMethod", "SaleResult"]

LotMethod = Literal["fifo", "lifo", "hifo", "lofo"]

#: A loss is disallowed if substantially identical shares are bought within 30
#: days *either side* of the sale — a 61-day window in total.
WASH_SALE_DAYS = 30

#: Held *more than* one year qualifies for long-term treatment.
LONG_TERM_DAYS = 365


@dataclass
class Lot:
    permno: int
    shares: float
    basis_per_share: float
    acquired: pd.Timestamp

    def is_long_term(self, on: pd.Timestamp) -> bool:
        return (on - self.acquired).days > LONG_TERM_DAYS

    @property
    def cost(self) -> float:
        return self.shares * self.basis_per_share


@dataclass
class SaleResult:
    permno: int
    date: pd.Timestamp
    shares: float
    proceeds: float
    short_gain: float
    long_gain: float
    #: Loss disallowed by the wash-sale rule and added to replacement-lot basis.
    disallowed_loss: float = 0.0
    lots_consumed: int = 0

    @property
    def realized_gain(self) -> float:
        return self.short_gain + self.long_gain


class LotBook:
    """Open tax lots for a whole portfolio."""

    def __init__(self, method: LotMethod = "hifo") -> None:
        self.method: LotMethod = method
        self._lots: dict[int, list[Lot]] = {}
        #: Dates on which a loss was realized per name, for wash-sale checks.
        self._loss_sales: dict[int, list[pd.Timestamp]] = {}
        self._purchases: dict[int, list[pd.Timestamp]] = {}
        #: Losses disallowed by a wash sale, waiting to be absorbed into the
        #: basis of a replacement lot — which is where they are recovered.
        self._pending_disallowed: dict[int, float] = {}
        self.realized: list[SaleResult] = []

    # -- positions ---------------------------------------------------------

    def shares(self, permno: int) -> float:
        return sum(lot.shares for lot in self._lots.get(permno, []))

    def cost_basis(self, permno: int) -> float:
        return sum(lot.cost for lot in self._lots.get(permno, []))

    def lots(self, permno: int) -> list[Lot]:
        return list(self._lots.get(permno, []))

    def holdings(self) -> dict[int, float]:
        return {p: s for p in self._lots if (s := self.shares(p)) > 0}

    def unrealized(self, prices: dict[int, float]) -> float:
        return sum(
            self.shares(p) * prices[p] - self.cost_basis(p)
            for p in self._lots
            if p in prices
        )

    # -- transactions ------------------------------------------------------

    def buy(self, permno: int, shares: float, price: float, date) -> None:
        if shares <= 0:
            raise ValueError(f"buy shares must be positive, got {shares}")
        date = pd.Timestamp(date)
        basis = price

        # A purchase inside the wash window of a recent loss sale absorbs the
        # disallowed loss into this lot's basis, which is where it is recovered.
        pending = self._pending_disallowed.pop(permno, 0.0)
        if pending:
            basis += pending / shares

        self._lots.setdefault(permno, []).append(Lot(permno, shares, basis, date))
        self._purchases.setdefault(permno, []).append(date)

    def sell(
        self, permno: int, shares: float, price: float, date, method: LotMethod | None = None
    ) -> SaleResult:
        """Sell `shares`, consuming lots by the chosen method.

        Default is HIFO — highest cost first — which realizes the smallest gain
        and defers the most tax. That is usually right, but not always: when
        headroom is unused, realizing a *large* long-term gain at 0% is strictly
        better, and the optimizer may deliberately choose LOFO to fill it.
        """
        date = pd.Timestamp(date)
        available = self.shares(permno)
        if shares > available + 1e-9:
            raise ValueError(
                f"cannot sell {shares:g} of permno {permno}: only {available:g} held "
                "(this engine is long-only)"
            )

        order = self._selection_order(permno, method or self.method, date)
        remaining = shares
        short_gain = long_gain = 0.0
        consumed = 0

        for lot in order:
            if remaining <= 1e-12:
                break
            take = min(lot.shares, remaining)
            gain = take * (price - lot.basis_per_share)
            if lot.is_long_term(date):
                long_gain += gain
            else:
                short_gain += gain
            lot.shares -= take
            remaining -= take
            consumed += 1

        self._lots[permno] = [lot for lot in self._lots[permno] if lot.shares > 1e-12]

        total_gain = short_gain + long_gain
        disallowed = 0.0
        if total_gain < 0 and self._in_wash_window(permno, date):
            disallowed = -total_gain
            self._pending_disallowed[permno] = (
                self._pending_disallowed.get(permno, 0.0) + disallowed
            )
            short_gain = long_gain = 0.0

        if total_gain < 0:
            self._loss_sales.setdefault(permno, []).append(date)

        result = SaleResult(
            permno=permno, date=date, shares=shares, proceeds=shares * price,
            short_gain=short_gain, long_gain=long_gain,
            disallowed_loss=disallowed, lots_consumed=consumed,
        )
        self.realized.append(result)
        return result

    # -- wash sales --------------------------------------------------------

    def _in_wash_window(self, permno: int, date: pd.Timestamp) -> bool:
        """Was there a purchase within 30 days before this sale?

        The forward half of the window is enforced by :meth:`repurchase_blocked`,
        which the optimizer must consult *before* buying — you cannot detect a
        future purchase at sale time.
        """
        window = pd.Timedelta(days=WASH_SALE_DAYS)
        return any(
            abs((date - bought).days) <= window.days
            for bought in self._purchases.get(permno, [])
        )

    def repurchase_blocked(self, permno: int, date) -> bool:
        """Would buying this name now disallow a recent loss? Ask before buying."""
        date = pd.Timestamp(date)
        return any(
            (date - sold).days <= WASH_SALE_DAYS
            for sold in self._loss_sales.get(permno, [])
        )

    def blocked_names(self, date) -> set[int]:
        """Every name currently inside a wash-sale window — a constraint set."""
        return {p for p in self._loss_sales if self.repurchase_blocked(p, date)}

    # -- realized totals ---------------------------------------------------

    def realized_gains(self, year: int | None = None) -> tuple[float, float]:
        """(short, long) realized gains, optionally restricted to a tax year."""
        rows = self.realized
        if year is not None:
            rows = [r for r in rows if r.date.year == year]
        return (sum(r.short_gain for r in rows), sum(r.long_gain for r in rows))

    # -- internals ---------------------------------------------------------

    def _selection_order(self, permno: int, method: LotMethod, date: pd.Timestamp) -> list[Lot]:
        lots = self._lots.get(permno, [])
        match method:
            case "fifo":
                return sorted(lots, key=lambda x: x.acquired)
            case "lifo":
                return sorted(lots, key=lambda x: x.acquired, reverse=True)
            case "hifo":
                return sorted(lots, key=lambda x: x.basis_per_share, reverse=True)
            case "lofo":
                return sorted(lots, key=lambda x: x.basis_per_share)
            case _:
                raise ValueError(f"unknown lot method {method!r}")

    def __repr__(self) -> str:
        return f"<LotBook {self.method}, {len(self.holdings())} positions>"
