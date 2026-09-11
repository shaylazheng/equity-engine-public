"""Compustat quarterly fundamentals, and the two traps in them.

**Trap 1 — `rdq` is quarterly.** The plan's point-in-time table says to gate
Compustat on `rdq`. That design is right, but `rdq` does not exist in
`comp.funda`; it is in `comp.fundq`. Verified 2026-07-30 — `funda` has 949
columns and none of them is `rdq`, and its date alternatives are far too sparse
to substitute (`apdedate` 23.8%, `fdate` 21.6%, `pdate` 10.0%).

`fundq.rdq` is **84.9%** populated on the analysis universe (`indfmt='INDL'`
etc.), and unevenly: 1970s 83.8%, **1980s 60.3%**, 1990s 72.2%, 2000s 98.7%,
2010s 98.5%, 2020s 99.1%, 1960s zero. So ~15% of the panel is dated by
assumption, concentrated in the 1980s-90s. Reported per era rather than averaged,
because the average hides that the 1980s are only 60%.

(An earlier figure of 66.5% came from querying the table *unfiltered*, which
includes restated and alternative-format rows that carry no `rdq`.)

**Trap 2 — cash-flow items are year-to-date.** `oancfy` accumulates within the
fiscal year: mean |value| runs 146 / 313 / 406 / 610 across Q1-Q4. Using it
directly as a quarterly figure makes Q4 look four times Q1 for an unchanged
business, which would corrupt any accruals signal built on it. Quarterly values
require differencing consecutive quarters within the fiscal year.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "REPORT_LAG_DAYS",
    "YTD_ITEMS",
    "assign_knowledge_date",
    "rdq_coverage",
    "ytd_to_quarterly",
]

#: Fallback when `rdq` is absent. Compustat's own guidance and the standard in
#: the literature; deliberately conservative, since a too-early date leaks.
REPORT_LAG_DAYS = 90

#: Compustat quarterly items carrying a `y` suffix accumulate within the fiscal
#: year. Cash-flow items are the ones that bite; income-statement `q` items do not.
YTD_ITEMS: tuple[str, ...] = ("oancfy", "capxy", "dvy", "ivncfy", "fincfy", "oancfq")


def ytd_to_quarterly(
    frame: pd.DataFrame,
    columns: tuple[str, ...] | list[str] = ("oancfy",),
    *,
    gvkey_col: str = "gvkey",
    year_col: str = "fyearq",
    quarter_col: str = "fqtr",
    suffix: str = "_q",
) -> pd.DataFrame:
    """Difference year-to-date items into quarterly flows.

    Q1 is taken as-is (nothing precedes it in the fiscal year); Q2-Q4 are the
    difference against the prior quarter. A gap in the quarter sequence yields
    NaN rather than a wrong difference — silently differencing Q4 against Q2
    would double-count a quarter.
    """
    out = frame.sort_values([gvkey_col, year_col, quarter_col], kind="stable").copy()
    grp = out.groupby([gvkey_col, year_col], sort=False)

    prev_q = grp[quarter_col].shift(1)

    # Real Compustat carries rows with a null `fqtr`. A null quarter has no
    # position in the fiscal year, so it can be neither "first" nor "contiguous
    # with the previous one" — both masks resolve to False and the row yields
    # NaN. Without the fillna these stay pandas NA and `np.where` raises
    # "boolean value of NA is ambiguous"; synthetic fixtures never had one.
    contiguous = prev_q.eq(out[quarter_col] - 1).fillna(False).astype(bool)
    is_first = out[quarter_col].eq(1).fillna(False).astype(bool)

    for col in columns:
        if col not in out.columns:
            continue
        prev = grp[col].shift(1)
        diff = out[col] - prev
        out[f"{col}{suffix}"] = np.where(
            is_first, out[col], np.where(contiguous, diff, np.nan)
        )

    return out


def assign_knowledge_date(
    frame: pd.DataFrame,
    *,
    rdq_col: str = "rdq",
    datadate_col: str = "datadate",
    lag_days: int = REPORT_LAG_DAYS,
) -> pd.DataFrame:
    """Derive the point-in-time dates a fundamental fact carries.

    Adds:

    - ``reported_at`` — `rdq` where present, else ``datadate + lag_days``
    - ``knowledge_date`` — same (a fresh pull learns it when it was reported)
    - ``report_date_source`` — ``rdq`` or ``assumed_lag``

    Keeping the source is the point. "Populated **or explicitly flagged**" was
    the Phase 1 acceptance criterion, and a fact dated by assumption should never
    be indistinguishable from one dated by the filing.
    """
    if datadate_col not in frame.columns:
        raise KeyError(f"missing {datadate_col!r}")

    out = frame.copy()
    datadate = pd.to_datetime(out[datadate_col])

    if rdq_col in out.columns:
        rdq = pd.to_datetime(out[rdq_col], errors="coerce")
    else:
        rdq = pd.Series(pd.NaT, index=out.index)

    fallback = datadate + pd.Timedelta(days=lag_days)

    # An rdq at or before period end is impossible — you cannot report a quarter
    # before it ends. Treat it as bad data rather than trusting it, since a
    # too-early knowledge_date is exactly the look-ahead being guarded against.
    implausible = rdq.notna() & (rdq <= datadate)
    rdq = rdq.mask(implausible)

    out["reported_at"] = rdq.fillna(fallback)
    out["knowledge_date"] = out["reported_at"]
    out["report_date_source"] = np.where(rdq.notna(), "rdq", "assumed_lag")
    out["report_date_implausible"] = implausible
    return out


def rdq_coverage(frame: pd.DataFrame, *, by: str = "decade") -> pd.DataFrame:
    """Share of rows dated by a real report date, by era.

    Report this alongside any result. The panel average is ~85%; by decade it
    shows the 1980s at 60% against ~99% from 2000 onward.
    """
    if "report_date_source" not in frame.columns:
        frame = assign_knowledge_date(frame)

    out = frame.copy()
    year = pd.to_datetime(out["datadate"]).dt.year
    out["_era"] = (year // 10) * 10 if by == "decade" else year

    grouped = out.groupby("_era")["report_date_source"]
    summary = pd.DataFrame(
        {
            "rows": grouped.size(),
            "from_rdq": grouped.apply(lambda s: int((s == "rdq").sum())),
        }
    )
    summary["share"] = summary["from_rdq"] / summary["rows"]
    summary.index.name = by
    return summary
