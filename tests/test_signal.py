"""Signal metadata must be self-consistent before anything computes."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import pandas as pd
import pytest
from qe_core.panel import AsOfView
from qe_core.signal import Signal, SignalContractError, validate_signal


@dataclass
class Demo:
    """A minimal conforming signal, used as a base to perturb."""

    name: str = "gross_profitability"
    family: str = "quality"
    tier: str = "core"
    evidence: str = "scored"
    requires: list[str] = field(default_factory=lambda: ["revt", "at"])
    min_date: dt.date = dt.date(1970, 1, 1)
    expected_sign: int = 1
    horizon: int = 21
    native_freq: str = "quarterly"
    neutralize: list[str] = field(default_factory=lambda: ["size", "value"])
    materiality: dict = field(default_factory=dict)
    component_of: str | None = None
    threshold_version: str = "v1"
    citation: str = "Novy-Marx (2013)"

    def compute(self, view: AsOfView) -> pd.Series:
        wide = view.pivot(["revt", "at"])
        return (wide["revt"] / wide["at"]).rename(self.name)


def test_a_conforming_signal_validates():
    validate_signal(Demo())


def test_it_satisfies_the_protocol():
    assert isinstance(Demo(), Signal)


def test_compute_receives_a_view_not_a_panel():
    """Structural, not stylistic: given only a view, a signal cannot pick its own date."""
    from qe_core.synthetic import generate

    synth = generate(n_firms=15, start="2016-01-04", end="2017-12-29", seed=2)

    # FY2016 is reported ~2017-03-16, so it is knowable here and not before.
    out = Demo().compute(synth.panel.as_of("2017-06-30"))
    assert isinstance(out, pd.Series)
    assert out.notna().any()
    assert not hasattr(synth.panel.as_of("2017-06-30"), "as_of_panel")


def test_a_signal_sees_nothing_before_its_inputs_are_reported():
    """The same signal, run inside the reporting gap, yields nothing rather than stale truth."""
    from qe_core.synthetic import generate

    synth = generate(n_firms=15, start="2016-01-04", end="2017-12-29", seed=2)
    out = Demo().compute(synth.panel.as_of("2016-12-30"))
    assert out.empty or out.isna().all()


# -- the tier/evidence coupling ------------------------------------------


def test_flag_evidence_requires_flag_tier():
    with pytest.raises(SignalContractError, match="requires tier='flag'"):
        validate_signal(Demo(name="restatement_8k", evidence="flag", tier="core"))


def test_scored_tier_rejects_non_scored_evidence():
    """Defensive: `evidence` is a Literal, but nothing enforces Literals at runtime."""
    with pytest.raises(SignalContractError, match="scored tier"):
        validate_signal(Demo(tier="exploratory", evidence="unknown"))


def test_flag_evidence_on_a_scored_tier_is_caught_by_the_flag_rule():
    with pytest.raises(SignalContractError, match="requires tier='flag'"):
        validate_signal(Demo(tier="exploratory", evidence="flag"))


def test_a_proper_flag_validates():
    validate_signal(
        Demo(
            name="restatement_8k",
            family="event",
            tier="flag",
            evidence="flag",
            requires=["form_type"],
            citation="SEC 8-K item 4.02",
        )
    )


# -- the rest of the contract --------------------------------------------


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"horizon": 0}, "horizon"),
        ({"expected_sign": 0}, "expected_sign"),
        ({"requires": []}, "requires"),
        ({"citation": "  "}, "citation"),
    ],
)
def test_malformed_metadata_is_rejected(kwargs, match):
    with pytest.raises(SignalContractError, match=match):
        validate_signal(Demo(**kwargs))
