"""Base class and registry for signals.

`BaseSignal` supplies the metadata defaults so a concrete signal declares only
what makes it distinctive. `validate_signal` from qe-core still runs at
registration — a signal whose declared tier and evidence disagree should never
reach a backtest, and finding that out at registration is cheaper than finding it
out in a report.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from typing import ClassVar

import pandas as pd
from qe_core.panel import AsOfView
from qe_core.signal import Evidence, Family, NativeFreq, Tier, validate_signal

__all__ = ["REGISTRY", "BaseSignal", "SignalRegistry"]


class BaseSignal(ABC):
    """One signal. Subclasses set the class attributes and implement `compute`."""

    name: ClassVar[str] = ""
    family: ClassVar[Family] = "quality"
    tier: ClassVar[Tier] = "exploratory"
    evidence: ClassVar[Evidence] = "scored"
    requires: ClassVar[list[str]] = []
    min_date: ClassVar[dt.date] = dt.date(1970, 1, 1)
    expected_sign: ClassVar[int] = 1
    horizon: ClassVar[int] = 21
    native_freq: ClassVar[NativeFreq] = "daily"
    neutralize: ClassVar[list[str]] = []
    materiality: ClassVar[dict] = {}
    component_of: ClassVar[str | None] = None
    threshold_version: ClassVar[str] = "v1"
    citation: ClassVar[str] = ""

    @abstractmethod
    def compute(self, view: AsOfView) -> pd.Series:
        """Raw signal for the view's date, indexed by permno."""

    # -- helpers available to every signal ---------------------------------

    def validate_against_panel(self, panel) -> None:
        """Fail loudly when a required concept never appears in the panel at all.

        This is the misconfiguration check, and it runs once against the panel's
        whole vocabulary. Without it a missing field surfaces as an all-NaN
        column somewhere downstream, which reads as "this signal had no opinion"
        rather than "this signal could not run".
        """
        missing = [c for c in self.requires if c not in panel.vocabulary]
        if missing:
            raise KeyError(
                f"{self.name}: panel never contains required concept(s) {missing}; "
                f"available: {sorted(panel.vocabulary)}"
            )

    def has_inputs(self, view: AsOfView) -> bool:
        """Are this signal's inputs knowable at the view's date?

        Distinct from :meth:`validate_against_panel` on purpose. A fundamental
        signal legitimately has nothing to say before the first 10-K is filed —
        that is the point-in-time layer working, not a fault — so `compute`
        returns an empty result rather than raising. Raising here would make
        every backtest fail on its own opening months.
        """
        return set(self.requires) <= set(view.concepts())

    @staticmethod
    def as_of_label(view: AsOfView) -> str:
        return str(pd.Timestamp(view.as_of).date())

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name!r} tier={self.tier} horizon={self.horizon}d>"


class SignalRegistry:
    """Named collection of signals, validated on the way in."""

    def __init__(self) -> None:
        self._signals: dict[str, BaseSignal] = {}

    def register(self, signal: BaseSignal) -> BaseSignal:
        if not signal.name:
            raise ValueError(f"{type(signal).__name__} has no name")
        if signal.name in self._signals:
            raise ValueError(f"signal {signal.name!r} is already registered")
        validate_signal(signal)
        self._signals[signal.name] = signal
        return signal

    def get(self, name: str) -> BaseSignal:
        if name not in self._signals:
            raise KeyError(f"no signal {name!r}; have {sorted(self._signals)}")
        return self._signals[name]

    def by_tier(self, tier: Tier) -> list[BaseSignal]:
        return [s for s in self._signals.values() if s.tier == tier]

    def by_family(self, family: Family) -> list[BaseSignal]:
        return [s for s in self._signals.values() if s.family == family]

    def scored(self) -> list[BaseSignal]:
        """Signals eligible to enter a composite — core and exploratory only."""
        return [s for s in self._signals.values() if s.tier in ("core", "exploratory")]

    def flags(self) -> list[BaseSignal]:
        """Signals that are shown beside a score and never inside it."""
        return [s for s in self._signals.values() if s.evidence == "flag"]

    def flag_names(self) -> set[str]:
        return {s.name for s in self.flags()}

    def validate_requirements(self, panel) -> None:
        """Check every registered signal's inputs exist in the panel. Run once, at wiring time."""
        for sig in self._signals.values():
            sig.validate_against_panel(panel)

    def names(self) -> list[str]:
        return sorted(self._signals)

    def __len__(self) -> int:
        return len(self._signals)

    def __contains__(self, name: str) -> bool:
        return name in self._signals

    def __iter__(self):
        return iter(self._signals.values())

    def __repr__(self) -> str:
        return f"<SignalRegistry {len(self)} signals: {', '.join(self.names())}>"


#: The default registry the reference signals attach to.
REGISTRY = SignalRegistry()
