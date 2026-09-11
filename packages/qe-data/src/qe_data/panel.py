"""Raw synthetic dataset frames → the tri-axial point-in-time panel.

This is where the three time axes get set, and where getting it wrong produces a
backtest that looks excellent and is worthless:

| source | `period_end` | `reported_at` | `knowledge_date` |
|---|---|---|---|
| CRSP daily | trade date | same, post-close | same |
| Compustat quarterly | `datadate` (period end) | `rdq`, else `datadate + 90d` | same as reported |

The Compustat row is the classic backtest killer. A quarter ending 31 March is
not knowable until it is announced in mid-May; dating it `datadate` leaks roughly
six weeks of the future into every signal that reads it.
"""

from __future__ import annotations

import pandas as pd
from qe_core.panel import AsOfPanel

__all__ = [
    "FUNDAMENTAL_CONCEPTS",
    "PRICE_CONCEPTS",
    "PRICE_SCALES",
    "build_panel",
    "check_units",
    "fundamentals_to_facts",
    "prices_to_facts",
]

#: CRSP daily columns carried into the panel, and what they are called there.
PRICE_CONCEPTS: dict[str, str] = {
    "dlyret": "ret",
    "dlyretx": "retx",
    "dlyprc": "prc",
    "dlyclose": "close",
    "dlyhigh": "high",
    "dlylow": "low",
    "dlyvol": "vol",
    "dlyprcvol": "dolvol",
    "dlycap": "mktcap",
    # Derived in `prices_to_facts`, not pulled: shares for THIS permno's share
    # class. Distinct from Compustat `csho`, which is company-wide. See the note
    # there — for dual-class firms the two legitimately differ by 2-3x.
    "shrout": "shrout",
}

#: Compustat quarterly items, mapped to the names signals expect. `oancfy_q` is
#: the *differenced* quarterly cash flow, never the raw year-to-date column.
FUNDAMENTAL_CONCEPTS: dict[str, str] = {
    "atq": "at",
    "ltq": "lt",
    "ceqq": "ceq",
    "seqq": "seq",
    "niq": "ni",
    "ibq": "ib",
    "revtq": "revt",
    "dpq": "dp",
    "oancfy_q": "oancf",
    "cshoq": "csho",
}

#: **The panel's unit convention.** CRSP and Compustat disagree on denomination
#: in three different ways and nothing warns you about any of them, so the
#: convention is stated once here and asserted by :func:`check_units`.
#:
#: | quantity | panel unit |
#: |---|---|
#: | money (market cap, assets, income, dollar volume) | **$ millions** |
#: | share counts (volume, shares outstanding) | **millions of shares** |
#: | prices (close, high, low, bid, ask) | **$ per share**, unscaled |
#: | returns and ratios | fractions, unscaled |
#:
#: What each source actually supplies, verified against the 2018 pull:
#:
#: - CRSP `dlycap` — *thousands* of dollars. Apple's end-2018 value reads
#:   746,079,125 against a real market cap of ~$746 billion.
#: - CRSP `dlyprcvol` — *raw* dollars (exactly `dlyprc x dlyvol`).
#: - CRSP `dlyvol` — *raw shares*.
#: - Compustat `atq`, `niq`, ... — *millions* of dollars.
#: - Compustat `cshoq` — *millions* of shares. Cross-checked: MSFT's CRSP-implied
#:   share count (`dlycap * 1000 / dlyprc`) equals `cshoq * 1e6` to a ratio of
#:   1.000.
#:
#: Unscaled, cross-source ratios are wrong by 1000x (money) or 1,000,000x
#: (shares). Both failure modes produce numbers that *look* plausible rather than
#: obviously broken: `earnings_yield` came back at 0.00003 instead of ~0.01, and
#: raw turnover at 4,317 instead of 0.0043.
PRICE_SCALES: dict[str, float] = {
    "dlycap": 1e-3,     # thousands of dollars -> $ millions
    "dlyprcvol": 1e-6,  # raw dollars          -> $ millions
    "dlyvol": 1e-6,     # raw shares           -> millions of shares
}

_SOURCE_CRSP = "crsp"
_SOURCE_COMP = "compustat"
_URL_CRSP = "synthetic dataset://crsp/stkdlysecuritydata"
_URL_COMP = "synthetic dataset://comp/fundq"


def _melt(
    frame: pd.DataFrame,
    concepts: dict[str, str],
    *,
    id_cols: list[str],
    source: str,
    source_url: str,
) -> pd.DataFrame:
    present = {k: v for k, v in concepts.items() if k in frame.columns}
    if not present:
        raise KeyError(
            f"none of the expected columns {sorted(concepts)} are present; "
            f"got {sorted(frame.columns)[:20]}"
        )

    long = frame.melt(
        id_vars=id_cols,
        value_vars=list(present),
        var_name="raw_concept",
        value_name="value",
    )
    long["concept"] = long["raw_concept"].map(present)
    long["source"] = source
    long["source_url"] = source_url
    return long.dropna(subset=["value"])


def prices_to_facts(prices: pd.DataFrame) -> pd.DataFrame:
    """CRSP daily rows → panel facts.

    A price is knowable the same session, post-close: all three time axes
    coincide. That is the one source where they legitimately do.
    """
    frame = prices.copy()
    frame["dlycaldt"] = pd.to_datetime(frame["dlycaldt"])

    # Denominate everything in $ millions before melting, so nothing downstream
    # has to know which source a number came from.
    for col, scale in PRICE_SCALES.items():
        if col in frame.columns:
            frame[col] = frame[col] * scale

    # Derive a **share-class-level** share count from CRSP. Compustat's `csho` is
    # company-level and counts every class, so for a dual-class firm it is simply
    # a different quantity: on the 2018 pull GOOGL's permno implies 43% of
    # Alphabet's `csho`, FOXA 57%, DISCA 30%. Both numbers are correct; they
    # answer different questions. A per-permno signal — net share issuance,
    # turnover — wants this one.
    if {"dlycap", "dlyprc"} <= set(frame.columns):
        prc = frame["dlyprc"].where(frame["dlyprc"] > 0)
        frame["shrout"] = frame["dlycap"] / prc

    long = _melt(
        frame,
        PRICE_CONCEPTS,
        id_cols=["permno", "dlycaldt"],
        source=_SOURCE_CRSP,
        source_url=_URL_CRSP,
    )

    long = long.rename(columns={"dlycaldt": "period_end"})
    long["period_start"] = pd.NaT
    long["reported_at"] = long["period_end"]
    long["knowledge_date"] = long["period_end"]
    return long


def fundamentals_to_facts(fundq: pd.DataFrame) -> pd.DataFrame:
    """Compustat quarterly rows → panel facts, gated on the report date.

    Expects :func:`qe_data.fundamentals.assign_knowledge_date` to have run, so
    `reported_at` and `knowledge_date` are already present and their provenance
    is recorded in `report_date_source`.
    """
    required = {"permno", "datadate", "reported_at", "knowledge_date"}
    missing = required - set(fundq.columns)
    if missing:
        raise KeyError(
            f"fundamentals frame is missing {sorted(missing)} — run "
            "assign_knowledge_date() and attach_permno() first"
        )

    # Guard the year-to-date trap rather than trusting the caller to remember.
    # `oancfy` accumulates within the fiscal year; letting it through as `oancf`
    # would make Q4 read ~4x Q1 for an unchanged business and quietly corrupt
    # every accruals signal built on it.
    if "oancfy" in fundq.columns and "oancfy_q" not in fundq.columns:
        raise KeyError(
            "fundamentals carry raw `oancfy`, which is year-to-date, but no "
            "`oancfy_q`. Run qe_data.fundamentals.ytd_to_quarterly() first — "
            "using the YTD column as a quarterly flow corrupts accruals."
        )

    frame = fundq[fundq["permno"].notna()].copy()
    frame["permno"] = frame["permno"].astype("int64")
    frame["datadate"] = pd.to_datetime(frame["datadate"])

    id_cols = ["permno", "datadate", "reported_at", "knowledge_date"]
    long = _melt(
        frame,
        FUNDAMENTAL_CONCEPTS,
        id_cols=id_cols,
        source=_SOURCE_COMP,
        source_url=_URL_COMP,
    )

    long = long.rename(columns={"datadate": "period_end"})
    # A fiscal quarter: period_end is the quarter end, so period_start is ~3
    # months earlier. Approximated rather than derived from fyearq/fqtr, since
    # nothing downstream reads it and a wrong exact date would be worse.
    long["period_start"] = long["period_end"] - pd.offsets.QuarterBegin(startingMonth=1)
    return long


def check_units(panel: AsOfPanel, as_of, *, tolerance: float = 0.05) -> dict:
    """Assert the panel's unit convention via a dimensional identity.

    ``mktcap == prc * shares_outstanding`` must hold. Since `mktcap` is in $
    millions and `prc` is in $ per share, that identity is only satisfiable if
    share counts are in *millions*. So one check pins the whole convention, and
    it catches unit drift in either source — which is the failure mode that
    produces plausible-looking numbers rather than obvious breakage.

    Compares CRSP-implied shares against Compustat `csho`. **The median pins the
    units; the dispersion does not.** Roughly 15% of names sit outside a 5% band
    for a real reason — Compustat `csho` is company-wide while a CRSP permno is
    one share class, so dual-class firms legitimately disagree (GOOGL 0.43,
    FOXA 0.57, DISCA 0.30 on the 2018 pull). Judge units on `median_ratio`;
    treat `match_rate` as a share-class diagnostic, not a units one.
    """
    view = panel.as_of(as_of)
    wide = view.pivot(["mktcap", "prc", "csho"])
    both = wide.dropna(subset=["mktcap", "prc", "csho"])
    both = both[(both["prc"] > 0) & (both["csho"] > 0)]

    if both.empty:
        return {"names": 0, "median_ratio": float("nan"), "match_rate": float("nan")}

    implied = both["mktcap"] / both["prc"]          # millions of shares
    ratio = implied / both["csho"]
    within = ratio.between(1 - tolerance, 1 + tolerance)
    iqr = float(ratio.quantile(0.75) - ratio.quantile(0.25))

    return {
        "names": len(both),
        "median_ratio": float(ratio.median()),
        "iqr": iqr,
        "match_rate": float(within.mean()),
        "likely_multiclass": int((ratio < 0.9).sum()),
    }


def build_panel(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame | None = None,
    *,
    validate: bool = True,
) -> AsOfPanel:
    """Assemble raw frames into an `AsOfPanel`.

    The panel's own schema validation runs on the way in, so a missing
    `source_url` or a `knowledge_date` preceding `reported_at` fails here rather
    than surfacing as a mysterious result months later.
    """
    parts = [prices_to_facts(prices)]
    if fundamentals is not None and not fundamentals.empty:
        parts.append(fundamentals_to_facts(fundamentals))

    facts = pd.concat(parts, ignore_index=True)
    facts = facts.astype(
        {
            "permno": "int64",
            "concept": "string",
            "raw_concept": "string",
            "value": "float64",
            "source": "string",
            "source_url": "string",
        }
    )
    keep = [
        "permno", "concept", "value", "period_start", "period_end",
        "reported_at", "knowledge_date", "raw_concept", "source", "source_url",
    ]
    return AsOfPanel(facts[keep], validate=validate)
