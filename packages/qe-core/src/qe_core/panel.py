"""The point-in-time contract.

Every fact carries three time axes, not two:

    period_start / period_end   valid time      — the interval the number describes
    reported_at                 assertion time  — the source's own claim date
    knowledge_date              transaction time — when *this system* learned it

Collapsing the last two is the usual design, and it is fine right up until the
first backfill: re-pulling history stamps everything with today's knowledge, and
you either lose the ability to reproduce an old run or you silently inject
look-ahead. `reported_at` is what distinguishes an original 10-K from a 10-K/A;
`knowledge_date` is what keeps a backfill honest.

The only read path is :meth:`AsOfPanel.as_of`. There is deliberately no public
accessor that returns unfiltered data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

__all__ = [
    "PANEL_COLUMNS",
    "AsOfPanel",
    "AsOfView",
    "LookAheadError",
    "PanelSchemaError",
]


class LookAheadError(RuntimeError):
    """A read would have returned data that was not knowable at the as-of instant.

    This is raised, never silently corrected. A silent filter here would mean a
    broken PIT layer looks exactly like a working one.
    """


class PanelSchemaError(ValueError):
    """A frame does not satisfy the panel contract."""


#: Required columns and their pandas dtypes. `period_start` is nullable — an
#: instant fact (a price, a balance-sheet level) has no start.
PANEL_COLUMNS: dict[str, str] = {
    "permno": "int64",
    "concept": "string",
    "value": "float64",
    "period_start": "datetime64[ns]",
    "period_end": "datetime64[ns]",
    "reported_at": "datetime64[ns]",
    "knowledge_date": "datetime64[ns]",
    "raw_concept": "string",
    "source": "string",
    "source_url": "string",
}

_NON_NULL = [
    "permno",
    "concept",
    "value",
    "period_end",
    "reported_at",
    "knowledge_date",
    "source",
    "source_url",
]

#: Sentinel used only to make `period_start` groupable; never surfaces to callers.
_NAT_KEY = pd.Timestamp("1677-09-22")


def _guard(frame: pd.DataFrame, as_of: pd.Timestamp) -> None:
    """Assert nothing in `frame` postdates `as_of`.

    Runs on every view regardless of how it was constructed, so a bypassed or
    broken filter still fails loudly. This is the belt to :meth:`AsOfPanel.as_of`'s
    braces — and it is what the leak test actually exercises.
    """
    if frame.empty:
        return
    worst = frame["knowledge_date"].max()
    if worst > as_of:
        n = int((frame["knowledge_date"] > as_of).sum())
        raise LookAheadError(
            f"{n} row(s) not knowable at {as_of.date()}: latest knowledge_date is "
            f"{worst.date()}, which is {(worst - as_of).days} day(s) in the future. "
            "A signal must never see these."
        )


@dataclass(frozen=True)
class AsOfView:
    """Panel data as it was knowable at a single instant.

    Constructing one runs :func:`_guard`, so this type cannot hold future data —
    that invariant does not depend on the caller having used the right filter.
    """

    frame: pd.DataFrame
    as_of: pd.Timestamp
    _guarded: bool = field(default=True, repr=False)

    def __post_init__(self) -> None:
        if self._guarded:
            _guard(self.frame, self.as_of)

    def __len__(self) -> int:
        return len(self.frame)

    def _latest_per_name(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Collapse to the most recent period per (permno, concept).

        A view holds the whole knowable past, so a fundamental has one row per
        fiscal period, not one row per name. Both cross-sectional accessors below
        want the *current* value, and taking it by explicit ordering rather than
        by incidental row order matters: relying on the latter silently returns a
        stale period whenever the frame is reordered, and duplicates the index
        when a name has several periods in view.
        """
        if frame.empty:
            return frame
        return (
            frame.sort_values(["period_end", "reported_at"], kind="stable")
            .groupby(["permno", "concept"], sort=False)
            .tail(1)
        )

    def field(self, concept: str) -> pd.Series:
        """One concept's current value, as a Series indexed by permno."""
        sub = self.frame.loc[self.frame["concept"] == concept]
        sub = self._latest_per_name(sub)
        return sub.set_index("permno")["value"].rename(concept)

    def pivot(self, concepts: list[str] | None = None) -> pd.DataFrame:
        """Wide frame of current values: permno rows, concept columns."""
        sub = self.frame
        if concepts is not None:
            sub = sub.loc[sub["concept"].isin(concepts)]
        sub = self._latest_per_name(sub)
        if sub.empty:
            return pd.DataFrame(index=pd.Index([], name="permno"), columns=concepts or [])
        wide = sub.pivot(index="permno", columns="concept", values="value")
        wide.columns.name = None
        if concepts is not None:
            wide = wide.reindex(columns=concepts)
        return wide

    def series(self, concept: str) -> pd.DataFrame:
        """Wide history for one concept: period_end rows, permno columns.

        A view holds everything knowable at `as_of`, which includes the whole
        past — so a trailing-window signal like 12-1 momentum is computable from
        a single view without ever handling the panel. That is the property that
        lets `Signal.compute` take a view rather than the panel itself.
        """
        sub = self.frame.loc[self.frame["concept"] == concept]
        if sub.empty:
            return pd.DataFrame()
        wide = sub.pivot_table(
            index="period_end", columns="permno", values="value", aggfunc="last"
        ).sort_index()
        wide.columns.name = None
        return wide

    def concepts(self) -> list[str]:
        return sorted(self.frame["concept"].dropna().unique().tolist())


class AsOfPanel:
    """A long-format panel with an enforced point-in-time read path.

    Parameters
    ----------
    frame:
        Long frame carrying at least :data:`PANEL_COLUMNS`. Extra columns are
        preserved and carried through views.
    validate:
        Run schema and coherence checks at construction. Leave on except in hot
        loops where the frame is known-good.
    """

    def __init__(self, frame: pd.DataFrame, *, validate: bool = True) -> None:
        frame = frame.copy()
        if validate:
            self._validate(frame)
        self._frame = frame.sort_values(["knowledge_date", "reported_at"], kind="stable")

    # -- construction -----------------------------------------------------

    @staticmethod
    def _validate(frame: pd.DataFrame) -> None:
        missing = [c for c in PANEL_COLUMNS if c not in frame.columns]
        if missing:
            raise PanelSchemaError(f"panel is missing required column(s): {missing}")

        nulls = [c for c in _NON_NULL if frame[c].isna().any()]
        if nulls:
            raise PanelSchemaError(f"null values in non-nullable column(s): {nulls}")

        # P3, "no untraceable data". Postgres enforced this with a CHECK
        # constraint; parquet enforces nothing, so it has to happen here or it is
        # merely a convention.
        blank = frame["source_url"].astype("string").str.strip().eq("")
        if blank.any():
            raise PanelSchemaError(
                f"{int(blank.sum())} row(s) have an empty source_url; every fact must be traceable"
            )

        # You cannot learn a fact before its source asserted it. This catches a
        # whole class of ingest bugs that would otherwise read as look-ahead.
        early = frame["knowledge_date"] < frame["reported_at"]
        if early.any():
            i = frame.index[early][0]
            raise PanelSchemaError(
                f"{int(early.sum())} row(s) have knowledge_date < reported_at — e.g. row {i}: "
                f"learned {frame.loc[i, 'knowledge_date'].date()} but reported "
                f"{frame.loc[i, 'reported_at'].date()}"
            )

    # -- the only read path -----------------------------------------------

    def as_of(self, t, *, revisions: bool = False) -> AsOfView:
        """Everything knowable at `t`.

        By default returns one row per (permno, concept, period), being the latest
        assertion known by `t` — so a restatement filed after `t` is invisible,
        and one filed before it wins. Pass ``revisions=True`` for the full
        lineage instead.
        """
        t = pd.Timestamp(t)
        known = self._frame.loc[self._frame["knowledge_date"] <= t]

        if not revisions and not known.empty:
            key = known["period_start"].fillna(_NAT_KEY)
            known = (
                known.assign(_pstart=key)
                .sort_values(["reported_at", "knowledge_date"], kind="stable")
                .groupby(["permno", "concept", "_pstart", "period_end"], as_index=False, sort=False)
                .tail(1)
                .drop(columns="_pstart")
            )

        return AsOfView(known.reset_index(drop=True), t)

    # -- introspection (metadata only, never values) ----------------------

    @property
    def coverage(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """(earliest, latest) knowledge_date. Metadata, not data."""
        return self._frame["knowledge_date"].min(), self._frame["knowledge_date"].max()

    @property
    def vocabulary(self) -> set[str]:
        """Every concept the panel contains, at any date.

        Metadata rather than data — it exposes which fields *exist*, never their
        values or timing, so it leaks nothing. This is what distinguishes "this
        signal is misconfigured" from "this signal's inputs are not knowable
        yet", which are very different problems with the same symptom.
        """
        return set(self._frame["concept"].dropna().unique())

    def __len__(self) -> int:
        return len(self._frame)

    def __repr__(self) -> str:
        lo, hi = self.coverage
        return f"<AsOfPanel {len(self._frame):,} rows, knowable {lo.date()}..{hi.date()}>"
