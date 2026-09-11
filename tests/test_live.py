"""Trade list generation — the thing a human reads before acting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from qe_live.tradelist import drift, generate_trade_list
from qe_tax.engine import TaxEngine, TaxProfile
from qe_tax.lots import LotBook

AS_OF = pd.Timestamp("2024-06-28")
PRICES = pd.Series({10_001: 50.0, 10_002: 25.0, 10_003: 100.0, 10_004: 10.0})
EQUITY = 100_000.0


@pytest.fixture
def book():
    b = LotBook()
    b.buy(10_001, 400, 40.0, "2022-01-10")   # long-term, big gain
    b.buy(10_002, 800, 30.0, "2024-05-01")   # short-term, at a loss
    return b


# -- the basics -----------------------------------------------------------


def test_a_trade_list_moves_holdings_toward_target(book):
    target = pd.Series({10_001: 0.30, 10_003: 0.20})
    tl = generate_trade_list(target, PRICES, EQUITY, book=book, as_of=AS_OF)

    applied = {}
    for _, row in tl.rows.iterrows():
        applied[row["permno"]] = book.shares(row["permno"]) + row["shares"]

    for permno, weight in target.items():
        want = weight * EQUITY / PRICES[permno]
        assert applied.get(permno, book.shares(permno)) == pytest.approx(want, rel=1e-6)


def test_names_not_in_the_target_are_sold_down(book):
    tl = generate_trade_list(pd.Series({10_001: 0.30}), PRICES, EQUITY, book=book, as_of=AS_OF)
    sells = tl.sells
    assert 10_002 in set(sells["permno"]), "a dropped holding was not sold"


def test_buys_and_sells_partition_the_rows(book):
    target = pd.Series({10_003: 0.40})
    tl = generate_trade_list(target, PRICES, EQUITY, book=book, as_of=AS_OF)
    assert len(tl.buys) + len(tl.sells) == len(tl.rows)
    assert (tl.buys["shares"] > 0).all()
    assert (tl.sells["shares"] < 0).all()


def test_tiny_trades_are_suppressed(book):
    """Churn that only pays commission is not a trade."""
    current = book.shares(10_001) * PRICES[10_001] / EQUITY
    target = pd.Series({10_001: current + 0.0001})
    tl = generate_trade_list(
        target, PRICES, EQUITY, book=book, as_of=AS_OF, min_trade_value=1_000
    )
    assert 10_001 not in set(tl.rows["permno"]) if not tl.rows.empty else True


def test_rows_are_ordered_by_size(book):
    target = pd.Series({10_001: 0.10, 10_003: 0.50, 10_004: 0.05})
    tl = generate_trade_list(target, PRICES, EQUITY, book=book, as_of=AS_OF)
    notionals = tl.rows["notional"].abs().to_numpy()
    assert (np.diff(notionals) <= 1e-9).all(), "largest trades should come first"


def test_turnover_is_reported(book):
    target = pd.Series({10_003: 0.50})
    tl = generate_trade_list(target, PRICES, EQUITY, book=book, as_of=AS_OF)
    assert tl.turnover() > 0
    assert tl.notional_traded() == pytest.approx(tl.rows["notional"].abs().sum())


# -- wash sales are reported, not hidden ---------------------------------


def test_blocked_names_are_named(book):
    book.sell(10_002, 800, 25.0, "2024-06-20")  # realize a loss inside the window
    tl = generate_trade_list(
        pd.Series({10_002: 0.30}), PRICES, EQUITY, book=book, as_of=AS_OF
    )
    assert 10_002 in tl.blocked


def test_a_blocked_name_is_not_bought(book):
    book.sell(10_002, 800, 25.0, "2024-06-20")
    tl = generate_trade_list(
        pd.Series({10_002: 0.30}), PRICES, EQUITY, book=book, as_of=AS_OF
    )
    bought = tl.buys
    assert 10_002 not in set(bought["permno"]) if not bought.empty else True


def test_an_unblocked_name_reports_no_block(book):
    tl = generate_trade_list(
        pd.Series({10_003: 0.20}), PRICES, EQUITY, book=book, as_of=AS_OF
    )
    assert tl.blocked == ()


# -- tax impact -----------------------------------------------------------


def test_selling_inside_headroom_costs_nothing(book):
    engine = TaxEngine(TaxProfile(ordinary_income=20_000))
    tl = generate_trade_list(
        pd.Series(dtype=float), PRICES, EQUITY, book=book, as_of=AS_OF, tax_engine=engine
    )
    row = tl.rows.loc[tl.rows["permno"] == 10_001].iloc[0]
    assert row["est_gain"] > 0, "the long-term lot should show a gain"
    assert row["est_tax"] == pytest.approx(0.0)


def test_selling_beyond_headroom_costs_something(book):
    engine = TaxEngine(TaxProfile(ordinary_income=700_000))
    tl = generate_trade_list(
        pd.Series(dtype=float), PRICES, EQUITY, book=book, as_of=AS_OF, tax_engine=engine
    )
    row = tl.rows.loc[tl.rows["permno"] == 10_001].iloc[0]
    assert row["est_tax"] > 0


def test_headroom_is_reported(book):
    engine = TaxEngine(TaxProfile(ordinary_income=20_000))
    tl = generate_trade_list(
        pd.Series({10_003: 0.2}), PRICES, EQUITY, book=book, as_of=AS_OF, tax_engine=engine
    )
    assert tl.headroom is not None and tl.headroom > 0


def test_no_tax_engine_leaves_headroom_unset(book):
    tl = generate_trade_list(
        pd.Series({10_003: 0.2}), PRICES, EQUITY, book=book, as_of=AS_OF
    )
    assert tl.headroom is None


def test_estimating_a_gain_does_not_mutate_the_book(book):
    before = book.shares(10_001)
    generate_trade_list(pd.Series(dtype=float), PRICES, EQUITY, book=book, as_of=AS_OF)
    assert book.shares(10_001) == before, "estimating a sale actually sold something"


# -- scores travel with the trade ----------------------------------------


def test_scores_are_attached_when_supplied(book):
    scores = pd.Series({10_001: 1.4, 10_003: 2.2})
    tl = generate_trade_list(
        pd.Series({10_003: 0.3}), PRICES, EQUITY, book=book, as_of=AS_OF, scores=scores
    )
    row = tl.rows.loc[tl.rows["permno"] == 10_003].iloc[0]
    assert row["score"] == pytest.approx(2.2)


def test_unpriced_names_are_skipped(book):
    prices = PRICES.copy()
    prices[10_003] = np.nan
    tl = generate_trade_list(
        pd.Series({10_003: 0.5}), prices, EQUITY, book=book, as_of=AS_OF
    )
    assert 10_003 not in set(tl.rows["permno"]) if not tl.rows.empty else True


# -- drift ----------------------------------------------------------------


def test_drift_is_ordered_by_absolute_gap():
    target = pd.Series({1: 0.20, 2: 0.30, 3: 0.50})
    actual = pd.Series({1: 0.22, 2: 0.10, 3: 0.51})
    out = drift(target, actual)
    assert out.index[0] == 2
    assert out.loc[2] == pytest.approx(-0.20)


def test_drift_covers_names_in_either_portfolio():
    target = pd.Series({1: 0.5})
    actual = pd.Series({2: 0.5})
    out = drift(target, actual)
    assert set(out.index) == {1, 2}
    assert out.loc[1] == pytest.approx(-0.5)
    assert out.loc[2] == pytest.approx(0.5)


def test_no_drift_when_aligned():
    w = pd.Series({1: 0.5, 2: 0.5})
    assert drift(w, w).abs().max() == pytest.approx(0.0)
