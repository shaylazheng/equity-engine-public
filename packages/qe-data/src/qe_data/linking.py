"""Join sample security identifiers by their validity intervals."""

from __future__ import annotations

import pandas as pd

__all__ = ["LINKPRIM_RANK", "attach_permno", "link_coverage"]

#: Lower rank wins. Primary beats consolidated — stated, not implied by sorting.
LINKPRIM_RANK: dict[str, int] = {"P": 0, "C": 1}


def attach_permno(
    fundamentals: pd.DataFrame,
    link: pd.DataFrame,
    *,
    date_col: str = "datadate",
    keep_unlinked: bool = True,
) -> pd.DataFrame:
    """Attach `permno` to Compustat rows, respecting the link validity window.

    Parameters
    ----------
    keep_unlinked:
        Retain rows that found no link, with a null permno. Roughly 13% of
        Compustat rows are unlinked in practice, and dropping them silently
        shrinks the universe in a way that is hard to notice later. They are kept
        by default and counted by :func:`link_coverage`.
    """
    for col in (date_col, "gvkey"):
        if col not in fundamentals.columns:
            raise KeyError(f"fundamentals frame is missing {col!r}")

    merged = fundamentals.merge(link, on="gvkey", how="left")

    in_window = (merged[date_col] >= merged["linkdt"]) & (
        merged[date_col] <= merged["linkenddt"]
    )
    unlinked = merged["permno"].isna()
    merged = merged[in_window | unlinked].copy()

    # Explicit preference, so the winner does not depend on string ordering.
    merged["_rank"] = merged["linkprim"].map(LINKPRIM_RANK).fillna(99).astype(int)
    merged = (
        merged.sort_values(["gvkey", date_col, "_rank"], kind="stable")
        .drop_duplicates(["gvkey", date_col], keep="first")
        .drop(columns="_rank")
    )

    if not keep_unlinked:
        merged = merged[merged["permno"].notna()]

    return merged.reset_index(drop=True)


def link_coverage(linked: pd.DataFrame) -> dict:
    """Share of rows that found a permno. Report it — a silent drop is a bias."""
    n = len(linked)
    if n == 0:
        return {"rows": 0, "linked": 0, "share": 0.0}
    ok = int(linked["permno"].notna().sum())
    return {"rows": n, "linked": ok, "share": ok / n}
