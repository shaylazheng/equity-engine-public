"""The signal contract.

Two fields carry most of the weight here:

``tier``
    What the signal costs. Deflated Sharpe is only meaningful against a real
    trial count, so every *scored* signal tested is a trial charged against every
    result. ``core`` is the small pre-registered set that carries the headline;
    ``exploratory`` is tested but its trial cost is disclosed; ``risk`` and
    ``flag`` are never scored for alpha and never cost a trial.

``evidence``
    What backs the claim. ``scored`` means a validated return spread — it enters
    the composite. ``flag`` means a filing, an event, a disclosure — a fact worth
    seeing that makes no return claim, and never enters the composite.

The split that matters is not quantitative-vs-qualitative: everything here ends
up numeric, so that line collapses on contact. It is what kind of evidence backs
the signal.
"""

from __future__ import annotations

import datetime as dt
from typing import Literal, Protocol, runtime_checkable

import pandas as pd

from .panel import AsOfView

__all__ = [
    "Evidence",
    "Family",
    "NativeFreq",
    "Signal",
    "SignalContractError",
    "Tier",
    "validate_signal",
]

Tier = Literal["core", "exploratory", "risk", "flag"]
Evidence = Literal["scored", "flag"]
NativeFreq = Literal["daily", "monthly", "quarterly"]
Family = Literal[
    "insider",
    "institutional",
    "quality",
    "value",
    "momentum",
    "estimates",
    "risk",
    "event",
]

SCORED_TIERS: frozenset[str] = frozenset({"core", "exploratory"})


class SignalContractError(ValueError):
    """A signal's declared metadata is internally inconsistent."""


@runtime_checkable
class Signal(Protocol):
    """One measurable property of a security that may carry information.

    ``compute`` receives an :class:`~qe_core.panel.AsOfView`, not the panel. That
    is deliberate and is a change from the original design: a signal handed the
    whole panel could call ``as_of`` with a date of its choosing, so look-ahead
    would be a mistake away. Handed a single view, it structurally cannot see
    another date. The framework owns the date loop.
    """

    name: str
    family: Family
    tier: Tier
    evidence: Evidence

    #: Panel concepts this signal reads. Validated before any compute runs, so a
    #: missing field fails immediately rather than as a NaN column downstream.
    requires: list[str]

    #: Honest coverage start. Form 4 is 2003-08 whatever the panel contains.
    min_date: dt.date

    #: Direction the literature claims. Used to catch sign flips, not to enforce them.
    expected_sign: Literal[1, -1]

    #: Forward-return horizon in trading days. A 12-month signal must not be
    #: rebalanced daily, and its IC decay has to be read on its own horizon.
    horizon: int

    #: Natural compute frequency. Fundamentals change quarterly; recomputing them
    #: daily across a 150M-row panel repeats an unchanged number ~60x per quarter.
    #: The framework forward-fills onto the daily panel carrying knowledge_date
    #: forward unchanged, so the PIT guarantee is untouched.
    native_freq: NativeFreq

    #: Risk factors stripped before scoring.
    neutralize: list[str]

    #: Absolute floors, so statistically odd but trivially small activity does not
    #: fire. Applied in raw units, while thresholds apply to the normalized
    #: statistic — two different scales, deliberately not blended.
    materiality: dict

    #: Set when this signal is a component of a composite (e.g. piotroski_f), so
    #: the trial registry does not count correlated trials as independent.
    component_of: str | None

    #: Config version that produced a value. What makes a historical board reproducible.
    threshold_version: str

    citation: str

    def compute(self, view: AsOfView) -> pd.Series:
        """Raw signal for one date, indexed by permno. Normalization is the framework's job."""
        ...


def validate_signal(sig: Signal) -> None:
    """Check a signal's metadata is self-consistent. Cheap; run it at registration."""
    if sig.evidence == "flag" and sig.tier != "flag":
        raise SignalContractError(
            f"{sig.name}: evidence='flag' requires tier='flag', got tier={sig.tier!r}. "
            "Flags never enter the composite."
        )
    if sig.tier in SCORED_TIERS and sig.evidence != "scored":
        raise SignalContractError(
            f"{sig.name}: tier={sig.tier!r} is a scored tier but evidence={sig.evidence!r}"
        )
    if sig.tier == "risk" and sig.evidence != "scored":
        raise SignalContractError(
            f"{sig.name}: risk exposures are computed like scored signals but excluded from "
            f"alpha; got evidence={sig.evidence!r}"
        )
    if sig.horizon < 1:
        raise SignalContractError(f"{sig.name}: horizon must be >= 1 trading day, got {sig.horizon}")
    if sig.expected_sign not in (1, -1):
        raise SignalContractError(f"{sig.name}: expected_sign must be 1 or -1")
    if not sig.requires:
        raise SignalContractError(f"{sig.name}: requires must name at least one panel concept")
    if not sig.citation.strip():
        raise SignalContractError(f"{sig.name}: citation is mandatory")
