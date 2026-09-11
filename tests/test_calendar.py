"""Trading calendar — a wrong session count is a silent off-by-one in every lagged signal."""

from __future__ import annotations

import pandas as pd
from qe_core.calendar import is_trading_day, next_trading_day, shift_trading_days, trading_days


def test_weekends_are_excluded():
    days = trading_days("2024-01-01", "2024-01-31")
    assert (days.dayofweek < 5).all()


def test_known_holidays_are_excluded():
    days = trading_days("2024-01-01", "2024-12-31")
    for holiday in ("2024-01-01", "2024-07-04", "2024-11-28", "2024-12-25"):
        assert pd.Timestamp(holiday) not in days, f"{holiday} should not be a session"


def test_session_count_is_plausible():
    """~252 sessions a year. A naive freq='B' gives 262 and is wrong by every holiday."""
    assert 248 <= len(trading_days("2024-01-01", "2024-12-31")) <= 254


def test_good_friday_is_closed_but_not_a_federal_holiday():
    """A case a naive federal-holiday calendar gets wrong."""
    assert pd.Timestamp("2024-03-29") not in trading_days("2024-03-25", "2024-04-01")


def test_next_trading_day_skips_the_weekend():
    assert next_trading_day("2024-03-01") == pd.Timestamp("2024-03-04")  # Fri -> Mon


def test_next_trading_day_skips_a_holiday_weekend():
    # Fri 2024-03-28 -> Good Friday closed -> Mon 2024-04-01
    assert next_trading_day("2024-03-28") == pd.Timestamp("2024-04-01")


def test_shift_forward_and_back_round_trips():
    start = pd.Timestamp("2024-06-03")
    assert shift_trading_days(shift_trading_days(start, 10), -10) == start


def test_shift_by_zero_is_identity():
    assert shift_trading_days("2024-06-03", 0) == pd.Timestamp("2024-06-03")


def test_shift_one_matches_next_trading_day():
    assert shift_trading_days("2024-03-28", 1) == next_trading_day("2024-03-28")


def test_is_trading_day():
    assert is_trading_day("2024-06-03")
    assert not is_trading_day("2024-06-02")  # Sunday
    assert not is_trading_day("2024-07-04")


def test_the_1968_paperwork_crisis_closures_are_present():
    """Real NYSE history, not a generated weekday series.

    The exchange closed Wednesdays in H2 1968 to clear a settlement backlog. Any
    engine claiming a 1970s-present panel should get this era right.
    """
    days = trading_days("1968-07-01", "1968-12-31")
    wednesdays = days[days.dayofweek == 2]
    assert len(wednesdays) < 13, f"expected most Wednesdays closed, found {len(wednesdays)} open"
