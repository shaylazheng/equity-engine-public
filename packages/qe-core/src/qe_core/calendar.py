"""NYSE trading calendar.

Backed by `pandas_market_calendars`, which carries real NYSE history — the 1968
paperwork crisis closures, special closures, the pre-1952 Saturday sessions. A
naive `freq="B"` is wrong by hundreds of days across the 1970s-present span this
engine targets, and each wrong day is a silent off-by-one in every lagged signal.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

__all__ = ["is_trading_day", "next_trading_day", "shift_trading_days", "trading_days"]


@lru_cache(maxsize=8)
def _schedule(start: str, end: str, name: str = "NYSE") -> pd.DatetimeIndex:
    # Load the calendar only when requested.
    import pandas_market_calendars as mcal

    cal = mcal.get_calendar(name)
    sched = cal.schedule(start_date=start, end_date=end)
    return pd.DatetimeIndex(sched.index).normalize()


def trading_days(start, end, *, name: str = "NYSE") -> pd.DatetimeIndex:
    """Trading sessions in [start, end], inclusive."""
    return _schedule(str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date()), name)


def is_trading_day(day, *, name: str = "NYSE") -> bool:
    day = pd.Timestamp(day).normalize()
    return day in trading_days(day, day, name=name)


def next_trading_day(day, *, name: str = "NYSE") -> pd.Timestamp:
    """The next session strictly after `day`.

    Signals fire at the close on t and trade at t+1 — never same-bar — so this is
    on the path of every backtest fill.
    """
    day = pd.Timestamp(day).normalize()
    window = trading_days(day, day + pd.Timedelta(days=15), name=name)
    later = window[window > day]
    if len(later) == 0:
        raise ValueError(f"no trading day found within 15 days after {day.date()}")
    return later[0]


def shift_trading_days(day, n: int, *, name: str = "NYSE") -> pd.Timestamp:
    """Move `n` sessions from `day` (negative shifts back)."""
    day = pd.Timestamp(day).normalize()
    if n == 0:
        return day
    pad = pd.Timedelta(days=int(abs(n) * 2 + 15))
    lo, hi = (day - pad, day) if n < 0 else (day, day + pad)
    window = trading_days(lo, hi, name=name)
    side = window[window < day] if n < 0 else window[window > day]
    if len(side) < abs(n):
        raise ValueError(f"cannot shift {n} sessions from {day.date()} within the padded window")
    return side[n] if n < 0 else side[n - 1]
