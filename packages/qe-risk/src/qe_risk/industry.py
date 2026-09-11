"""Fama-French industry classification from SIC codes.

Industry is not optional in a cross-sectional risk model. Without it the fit
attributes a sector-wide move — an oil shock, a bank crisis — to whichever style
factor happens to correlate with that sector's size or valuation, and every
signal neutralized against those styles inherits the misattribution.

Implemented at the **12-industry** granularity rather than 49. FF49 buys finer
resolution at the cost of thin cells: a daily cross-section of ~4,000 names split
49 ways leaves industries with a handful of members, and a dummy fitted on five
observations is noise with a coefficient. FF12 keeps every cell populated across
the whole 1970-2025 span, including the sparse early years.

Ranges are Ken French's published SIC definitions. `Other` is a genuine bucket in
that scheme, not a dumping ground for failures — but a *high* share of `Other`
means the codes are not what you think, so :func:`industry_coverage` exists to
check rather than assume.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "FF12",
    "INDUSTRIES",
    "constrain_industries",
    "ff12_from_sic",
    "industry_coverage",
    "industry_dummies_ff12",
]

#: (label, [(lo, hi), ...]) — inclusive SIC ranges, in priority order. First
#: match wins, so overlapping definitions resolve the way French's do.
FF12: tuple[tuple[str, tuple[tuple[int, int], ...]], ...] = (
    ("NoDur", ((100, 999), (2000, 2399), (2700, 2749), (2770, 2799),
               (3100, 3199), (3940, 3989))),
    ("Durbl", ((2500, 2519), (2590, 2599), (3630, 3659), (3710, 3711),
               (3714, 3714), (3716, 3716), (3750, 3751), (3792, 3792),
               (3900, 3939), (3990, 3999))),
    ("Enrgy", ((1200, 1399), (2900, 2999))),
    ("Chems", ((2800, 2829), (2840, 2899))),
    ("Hlth", ((2830, 2839), (3693, 3693), (3840, 3859), (8000, 8099))),
    ("BusEq", ((3570, 3579), (3660, 3692), (3694, 3699), (3810, 3829),
               (7370, 7379))),
    ("Telcm", ((4800, 4899),)),
    ("Utils", ((4900, 4949),)),
    ("Shops", ((5000, 5999), (7200, 7299), (7600, 7699))),
    ("Money", ((6000, 6999),)),
    ("Manuf", ((2520, 2589), (2600, 2699), (2750, 2769), (3000, 3099),
               (3200, 3569), (3580, 3629), (3700, 3709), (3712, 3713),
               (3715, 3715), (3717, 3749), (3752, 3791), (3793, 3799),
               (3830, 3839), (3860, 3899))),
)

INDUSTRIES: tuple[str, ...] = tuple(label for label, _ in FF12) + ("Other",)


def ff12_from_sic(siccd: pd.Series) -> pd.Series:
    """Map SIC codes to the 12 Fama-French industries.

    Unmapped or missing codes become ``Other``, which is a real category here —
    but check :func:`industry_coverage` before trusting a fit, because a large
    ``Other`` share usually means the codes are stale or in the wrong column
    rather than that the firms are genuinely miscellaneous.
    """
    codes = pd.to_numeric(siccd, errors="coerce")
    out = pd.Series("Other", index=siccd.index, dtype="object")
    assigned = pd.Series(False, index=siccd.index)

    for label, ranges in FF12:
        hit = pd.Series(False, index=siccd.index)
        for lo, hi in ranges:
            hit |= codes.between(lo, hi)
        # First match wins — later definitions never steal an assigned name.
        take = hit & ~assigned
        out[take] = label
        assigned |= take

    out[codes.isna()] = "Other"
    return out.rename("industry")


def industry_coverage(industry: pd.Series) -> pd.DataFrame:
    """Share of names per industry. Read the `Other` row before trusting a fit."""
    counts = industry.value_counts(dropna=False)
    return pd.DataFrame({"names": counts, "share": counts / counts.sum()})


def constrain_industries(
    dummies: pd.DataFrame,
    date: pd.Series,
    weight: pd.Series,
    *,
    reference: str,
) -> pd.DataFrame:
    """Reparameterize industry dummies so their weighted returns sum to zero.

    Dropping a level and adding a market column is the cheap way to make the
    design full rank, but it changes what "market" *means*: the market factor
    becomes the return of the dropped industry, and every industry coefficient
    becomes a spread against it. On this panel that put the fitted market factor
    only 0.84-correlated with the actual cap-weighted market -- roughly 30% of it
    was the dropped industry's own idiosyncrasy. Since signals are neutralized
    against these factors, that misattribution propagates into every score.

    The standard fix is a constraint rather than a deletion: require the
    cap-weighted industry factor returns to sum to zero, so the market column
    carries the market and industries carry deviations from it. Imposed by
    substitution -- eliminate the reference industry via
    ``f_ref = -(1 / W_ref) * sum_k W_k f_k`` and fold it into the remaining
    columns, which is exact rather than a penalty.

    Weights are recomputed per date because industry composition is not stable:
    the cap share of `Money` and of `BusEq` in 1975 and in 2000 are different
    economies.
    """
    ref = f"ind_{reference}"
    if ref not in dummies.columns:
        raise KeyError(f"reference industry {reference!r} not among {list(dummies.columns)}")

    w = pd.to_numeric(weight, errors="coerce").astype(float)
    w = w.where(w > 0).fillna(0.0)

    mass = dummies.mul(w, axis=0).groupby(date.to_numpy()).sum()
    ratio = mass.div(mass[ref].where(mass[ref] > 0), axis=0)

    keyed = pd.Index(date.to_numpy())
    out = {}
    ref_col = dummies[ref].to_numpy()
    for col in dummies.columns:
        if col == ref:
            continue
        r = ratio[col].reindex(keyed).to_numpy()
        out[col] = dummies[col].to_numpy() - np.nan_to_num(r) * ref_col
    return pd.DataFrame(out, index=dummies.index)


def industry_dummies_ff12(
    siccd: pd.Series, *, drop: str | None = "Other"
) -> pd.DataFrame:
    """One-hot FF12 exposures.

    The dummies sum to one for every name, so keeping all twelve alongside any
    intercept-like factor makes X'X singular. Two ways out, and they are not
    equivalent:

    - ``drop="Other"`` (the default) deletes a level. Cheap, but it silently
      redefines the intercept as *that industry's* return.
    - ``drop=None`` keeps all twelve and hands them to
      :func:`constrain_industries`, which imposes the cap-weighted sum-to-zero
      constraint instead. This is what the real fit uses, and it is the reason
      the market factor means the market.
    """
    industry = ff12_from_sic(siccd)
    dummies = pd.get_dummies(industry, dtype=float)
    for label in INDUSTRIES:
        if label not in dummies.columns:
            dummies[label] = 0.0
    dummies = dummies[list(INDUSTRIES)]
    if drop is not None and drop in dummies.columns and dummies.shape[1] > 1:
        dummies = dummies.drop(columns=drop)
    return dummies.replace(0.0, 0.0).astype(float).set_axis(
        [f"ind_{c}" for c in dummies.columns], axis=1
    )
