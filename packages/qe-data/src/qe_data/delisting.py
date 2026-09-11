"""Delisting returns, including the half of the Shumway convention nobody implements.

Omitting delisting returns biases every backtest upward, because the firms that
vanish are disproportionately the ones that did badly. CRSP supplies `delret`
for **88.7%** of delistings (verified 2026-07-30); the remaining 11.3% are
missing precisely where it matters most — bankruptcies and compliance failures,
where the true return is very negative and unrecorded.

Shumway (1997) and Shumway-Warther (1999) supply the standard fill:

- **-30%** for NYSE / NYSE American
- **-55%** for NASDAQ

The sibling project implemented only the -30% leg, applied it to *every*
exchange, and could not do better because `exchcd` was never pulled. Under CIZ
the exchange lives on `stksecurityinfohist` as `primaryexch`, so both legs are
now possible — and the difference is large: NASDAQ delistings are the bulk of
the sample, and -30% where -55% belongs understates the loss by 25 points on
thousands of names.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "PERFORMANCE_REASONS",
    "SHUMWAY_NASDAQ",
    "SHUMWAY_NYSE_AMEX",
    "apply_delisting_returns",
    "fill_missing_delisting_returns",
]

SHUMWAY_NYSE_AMEX = -0.30
SHUMWAY_NASDAQ = -0.55

#: CIZ `delreasontype` codes treated as performance/compliance delistings — the
#: analogue of SIZ's `dlstcd in [400, 599]` band.
#:
#: These are the cases where a missing `delret` means "it went to zero and
#: nobody recorded it", so the Shumway fill applies. Mergers and voluntary
#: actions are deliberately excluded: a missing return there usually means the
#: shareholder was paid, and filling -55% would be a large fabricated loss.
#:
#: Derived from the code distribution observed on local demo. Worth re-checking
#: against CRSP's own documentation before any published result — the mapping is
#: a judgement call, not a lookup.
PERFORMANCE_REASONS: frozenset[str] = frozenset(
    {
        "BKPY",  # bankruptcy
        "DELQ",  # delinquent filings
        "FING",  # financial condition
        "INSC",  # insufficient capital / equity
        "INSF",  # insufficient float
        "LP",    # low price
        "MTMK",  # insufficient market makers
        "MVOT",  # insufficient market value of public shares
        "SHLD",  # insufficient shareholders
        "PUBI",  # public interest
        "CORQ",  # failure to meet corporate governance requirements
    }
)


def fill_missing_delisting_returns(
    delist: pd.DataFrame,
    *,
    performance_reasons: frozenset[str] = PERFORMANCE_REASONS,
    nyse_amex: float = SHUMWAY_NYSE_AMEX,
    nasdaq: float = SHUMWAY_NASDAQ,
) -> pd.DataFrame:
    """Fill absent `delret` on performance delistings, by exchange.

    Adds two columns rather than overwriting silently:

    - ``delret_filled`` — the return to use
    - ``delret_source`` — ``crsp`` / ``shumway_nyse_amex`` / ``shumway_nasdaq`` /
      ``zero``, so the share of the panel resting on an assumption is countable
      rather than invisible.
    """
    required = {"delret", "delreasontype", "primaryexch"}
    missing = required - set(delist.columns)
    if missing:
        raise KeyError(f"delisting frame is missing {sorted(missing)}")

    out = delist.copy()
    ret = pd.to_numeric(out["delret"], errors="coerce")

    is_missing = ret.isna()
    is_performance = out["delreasontype"].isin(performance_reasons)
    exch = out["primaryexch"].astype("string")

    on_nasdaq = exch.eq("Q")
    on_nyse_amex = exch.isin(["N", "A"])

    source = pd.Series("crsp", index=out.index, dtype="object")
    source[is_missing] = "zero"

    fill_nasdaq = is_missing & is_performance & on_nasdaq
    fill_listed = is_missing & is_performance & on_nyse_amex

    ret = ret.copy()
    ret[fill_nasdaq] = nasdaq
    ret[fill_listed] = nyse_amex
    source[fill_nasdaq] = "shumway_nasdaq"
    source[fill_listed] = "shumway_nyse_amex"

    # A missing non-performance delisting (mostly mergers) is treated as zero:
    # the position was paid out, and inventing a loss would be worse than
    # inventing nothing. Still tracked, because it is an assumption either way.
    ret = ret.fillna(0.0)

    out["delret_filled"] = ret
    out["delret_source"] = source
    return out


def apply_delisting_returns(
    prices: pd.DataFrame,
    delist: pd.DataFrame,
    *,
    date_col: str = "dlycaldt",
    ret_col: str = "dlyret",
) -> pd.DataFrame:
    """Compound the delisting return into the final session's return.

    Compounded — ``(1 + r)(1 + d) - 1`` — not substituted. The security may have
    traded on its last day *and then* delisted, and substituting discards that
    day's move.
    """
    if "delret_filled" not in delist.columns:
        delist = fill_missing_delisting_returns(delist)

    events = delist[["permno", "delistingdt", "delret_filled", "delret_source"]].rename(
        columns={"delistingdt": date_col}
    )

    out = prices.merge(events, on=["permno", date_col], how="left")
    hit = out["delret_filled"].notna()

    base = pd.to_numeric(out.loc[hit, ret_col], errors="coerce").fillna(0.0)
    out.loc[hit, ret_col] = (1.0 + base) * (1.0 + out.loc[hit, "delret_filled"]) - 1.0
    out[ret_col] = out[ret_col].replace([np.inf, -np.inf], np.nan)
    return out


def delisting_fill_summary(delist: pd.DataFrame) -> pd.Series:
    """How much of the delisting sample rests on an assumption. Report this."""
    if "delret_source" not in delist.columns:
        delist = fill_missing_delisting_returns(delist)
    return delist["delret_source"].value_counts()
