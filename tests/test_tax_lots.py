"""Lot accounting: holding periods, lot selection, wash sales."""

from __future__ import annotations

import pandas as pd
import pytest
from qe_tax.lots import LONG_TERM_DAYS, LotBook

JAN = pd.Timestamp("2024-01-15")


def _seeded(method="hifo"):
    """Three lots at different prices and dates."""
    book = LotBook(method)
    book.buy(1, 100, 10.0, "2022-01-10")   # cheap, long-term by 2024
    book.buy(1, 100, 30.0, "2022-06-10")   # dear, long-term
    book.buy(1, 100, 20.0, "2023-12-20")   # mid, short-term at Jan 2024
    return book


# -- positions ------------------------------------------------------------


def test_shares_and_basis_accumulate():
    book = _seeded()
    assert book.shares(1) == 300
    assert book.cost_basis(1) == pytest.approx(100 * 10 + 100 * 30 + 100 * 20)


def test_cannot_sell_more_than_held():
    book = _seeded()
    with pytest.raises(ValueError, match="long-only"):
        book.sell(1, 500, 25.0, JAN)


# -- holding period -------------------------------------------------------


def test_the_365_day_boundary():
    book = LotBook()
    book.buy(1, 10, 10.0, "2023-01-10")
    assert not book.lots(1)[0].is_long_term(pd.Timestamp("2024-01-10"))  # exactly 365
    assert book.lots(1)[0].is_long_term(pd.Timestamp("2024-01-11"))      # 366


def test_gains_split_by_holding_period():
    book = _seeded("fifo")
    # FIFO takes the 2022 lot first (long-term), then the other 2022 lot.
    result = book.sell(1, 150, 40.0, JAN)
    assert result.long_gain > 0
    assert result.short_gain == pytest.approx(0.0)


def test_a_short_lot_produces_short_gain():
    book = _seeded("lifo")
    # LIFO takes the Dec 2023 lot first — short-term in Jan 2024.
    result = book.sell(1, 50, 40.0, JAN)
    assert result.short_gain == pytest.approx(50 * (40 - 20))
    assert result.long_gain == pytest.approx(0.0)


# -- lot selection --------------------------------------------------------


def test_hifo_realizes_the_smallest_gain():
    hifo = _seeded("hifo").sell(1, 100, 40.0, JAN)
    fifo = _seeded("fifo").sell(1, 100, 40.0, JAN)
    assert hifo.realized_gain < fifo.realized_gain
    assert hifo.realized_gain == pytest.approx(100 * (40 - 30))


def test_lofo_realizes_the_largest_gain():
    """Deliberately useful: with unused 0% headroom, a big long-term gain is free
    and steps up basis, so filling the band beats deferring."""
    lofo = _seeded("lofo").sell(1, 100, 40.0, JAN)
    assert lofo.realized_gain == pytest.approx(100 * (40 - 10))


def test_method_can_be_overridden_per_sale():
    book = _seeded("hifo")
    result = book.sell(1, 100, 40.0, JAN, method="lofo")
    assert result.realized_gain == pytest.approx(100 * (40 - 10))


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="unknown lot method"):
        _seeded().sell(1, 10, 40.0, JAN, method="random")  # type: ignore[arg-type]


def test_partial_lot_consumption_leaves_a_remainder():
    book = _seeded("fifo")
    book.sell(1, 50, 40.0, JAN)
    assert book.shares(1) == 250
    assert book.lots(1)[0].shares == pytest.approx(50)


# -- wash sales -----------------------------------------------------------


def test_a_loss_blocks_repurchase_for_thirty_days():
    book = LotBook()
    book.buy(1, 100, 50.0, "2024-01-02")
    book.sell(1, 100, 40.0, "2024-03-01")  # realized loss, well outside the buy window

    assert book.repurchase_blocked(1, "2024-03-15")
    assert book.repurchase_blocked(1, "2024-03-31")
    assert not book.repurchase_blocked(1, "2024-04-05")


def test_blocked_names_is_a_constraint_set():
    book = LotBook()
    book.buy(1, 10, 50.0, "2024-01-02")
    book.buy(2, 10, 50.0, "2024-01-02")
    book.sell(1, 10, 40.0, "2024-03-01")
    assert book.blocked_names("2024-03-10") == {1}


def test_a_gain_does_not_block_anything():
    book = LotBook()
    book.buy(1, 100, 40.0, "2024-01-02")
    book.sell(1, 100, 50.0, "2024-03-01")
    assert not book.repurchase_blocked(1, "2024-03-10")


def test_selling_at_a_loss_soon_after_buying_disallows_it():
    book = LotBook()
    book.buy(1, 100, 50.0, "2024-01-02")
    book.buy(1, 100, 48.0, "2024-02-20")          # replacement inside the window
    result = book.sell(1, 100, 40.0, "2024-03-01")

    assert result.disallowed_loss > 0
    assert result.short_gain == 0.0 and result.long_gain == 0.0, (
        "a disallowed loss must not reduce taxable gains"
    )


def test_a_disallowed_loss_is_recovered_in_the_replacement_basis():
    """The loss is deferred, not destroyed — it lands in the next lot's basis."""
    book = LotBook()
    book.buy(1, 100, 50.0, "2024-01-02")
    book.buy(1, 100, 48.0, "2024-02-20")
    book.sell(1, 100, 40.0, "2024-03-01")

    before = book.cost_basis(1)
    book.buy(1, 100, 45.0, "2024-03-05")
    added = book.cost_basis(1) - before
    assert added > 100 * 45.0, "disallowed loss was not added to the replacement lot"


# -- totals ---------------------------------------------------------------


def test_realized_gains_split_by_character():
    book = _seeded("fifo")
    book.sell(1, 250, 40.0, JAN)
    short, long = book.realized_gains()
    assert long > 0
    assert short > 0  # the 250th share reaches the Dec 2023 lot


def test_realized_gains_filter_by_year():
    book = LotBook()
    book.buy(1, 100, 10.0, "2020-01-02")
    book.sell(1, 50, 20.0, "2023-06-01")
    book.sell(1, 50, 30.0, "2024-06-01")
    assert book.realized_gains(2023)[1] == pytest.approx(50 * 10)
    assert book.realized_gains(2024)[1] == pytest.approx(50 * 20)


def test_holding_period_constant_is_a_year():
    assert LONG_TERM_DAYS == 365
