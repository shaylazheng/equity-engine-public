"""The attribution ledger — scores stored as constituents, not as totals.

A score of 2.1 is uninterpretable. "2.1, of which 1.4 is insider cluster buying
and 0.9 is profitability, against -0.2 of momentum, with no unusual size or value
exposure" is a claim you can argue with. Being able to disagree with the engine
is the point of being able to see inside it.

So the total is **derived, never stored**. There is no code path that produces a
score without its decomposition, which makes `total == sum(contribution)` true by
construction rather than by discipline. :meth:`AttributionLedger.reconcile` exists
for the other direction — checking that a combiner's own output agrees with the
ledger it emitted.

Risk exposures are carried alongside but deliberately outside the sum, so
"scores well" can be told apart from "is a small-cap value tilt wearing a hat".
"""

from __future__ import annotations

import pandas as pd

__all__ = [
    "LEDGER_COLUMNS",
    "AttributionLedger",
    "FlagLeakError",
    "LedgerError",
]

LEDGER_COLUMNS: dict[str, str] = {
    "permno": "int64",
    "date": "datetime64[ns]",
    "signal": "string",
    "family": "string",
    "tier": "string",
    "weight": "float64",
    "z_value": "float64",
    "contribution": "float64",
}

#: Tiers permitted to contribute to a score. `risk` and `flag` are excluded, for
#: different reasons: a risk exposure is already priced, a flag makes no return claim.
_SCORING_TIERS = frozenset({"core", "exploratory"})

_DEFAULT_TOL = 1e-9


class LedgerError(ValueError):
    """The ledger is malformed or fails to reconcile."""


class FlagLeakError(LedgerError):
    """A flag-evidence signal reached the composite.

    Enforced by a check rather than left to convention — "flags never enter the
    score" is the kind of rule that quietly stops being true.
    """


class AttributionLedger:
    """Per-signal contributions to a score, one row per (permno, date, signal)."""

    def __init__(self, frame: pd.DataFrame, *, validate: bool = True) -> None:
        if validate:
            self._validate(frame)
        self.frame = frame.copy()

    @staticmethod
    def _validate(frame: pd.DataFrame) -> None:
        missing = [c for c in LEDGER_COLUMNS if c not in frame.columns]
        if missing:
            raise LedgerError(f"ledger is missing required column(s): {missing}")

        if frame[["permno", "date", "signal"]].duplicated().any():
            n = int(frame[["permno", "date", "signal"]].duplicated().sum())
            raise LedgerError(
                f"{n} duplicate (permno, date, signal) row(s); each signal contributes once"
            )

        bad = frame.loc[~frame["tier"].isin(_SCORING_TIERS), "tier"].unique()
        if len(bad):
            offenders = frame.loc[frame["tier"].isin(bad), "signal"].unique()[:5].tolist()
            raise FlagLeakError(
                f"non-scoring tier(s) {sorted(bad.tolist())} present in the ledger "
                f"(e.g. {offenders}). Only {sorted(_SCORING_TIERS)} may contribute to a score."
            )

        if frame["contribution"].isna().any():
            raise LedgerError("contribution contains nulls; a missing constituent is not zero")

    # -- the score, and every way of slicing it ---------------------------

    def totals(self) -> pd.Series:
        """The score. Derived from constituents — this is the only way to get one."""
        return self.frame.groupby(["permno", "date"])["contribution"].sum().rename("score")

    def by_signal(self, permno: int, date) -> pd.DataFrame:
        """Finest grain: why does *this* name score what it scores, on this date."""
        d = pd.Timestamp(date)
        sub = self.frame.loc[(self.frame["permno"] == permno) & (self.frame["date"] == d)]
        return sub.sort_values("contribution", ascending=False).reset_index(drop=True)

    def by_family(self) -> pd.Series:
        """insider / institutional / quality / value / momentum / estimates."""
        return self.frame.groupby(["permno", "date", "family"])["contribution"].sum()

    def by_tier(self) -> pd.Series:
        """How much of a score rests on unvalidated exploratory signals."""
        return self.frame.groupby(["permno", "date", "tier"])["contribution"].sum()

    # -- checks ------------------------------------------------------------

    def reconcile(self, external: pd.Series, *, tol: float = _DEFAULT_TOL) -> None:
        """Assert a combiner's own scores match the ledger it emitted.

        The Phase 7 acceptance criterion. A linear composite passes this exactly;
        an ML combiner's SHAP attributions will not, which is precisely the
        trade-off being measured.
        """
        mine = self.totals()
        aligned, other = mine.align(external, join="outer")

        missing = aligned.isna() | other.isna()
        if missing.any():
            raise LedgerError(
                f"{int(missing.sum())} (permno, date) key(s) appear in only one of "
                "the ledger and the external scores"
            )

        gap = (aligned - other).abs()
        if (gap > tol).any():
            worst = gap.idxmax()
            raise LedgerError(
                f"ledger does not reconcile: {int((gap > tol).sum())} score(s) differ by more "
                f"than {tol:g}; worst at {worst} — ledger {aligned[worst]:.10g} vs external "
                f"{other[worst]:.10g} (delta {gap[worst]:.3g})"
            )

    def assert_no_flags(self, flag_signals: set[str]) -> None:
        """Assert no flag-evidence signal reached the composite."""
        leaked = set(self.frame["signal"].dropna().unique()) & flag_signals
        if leaked:
            raise FlagLeakError(
                f"flag signal(s) {sorted(leaked)} contributed to a score; flags are shown "
                "beside the score and never inside it"
            )

    def __len__(self) -> int:
        return len(self.frame)

    def __repr__(self) -> str:
        n_names = self.frame["permno"].nunique()
        n_sig = self.frame["signal"].nunique()
        return f"<AttributionLedger {len(self.frame):,} rows, {n_names:,} names, {n_sig} signals>"
