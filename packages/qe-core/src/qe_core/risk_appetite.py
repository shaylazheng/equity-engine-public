"""Risk appetite as one named dial, not six literals copied across four scripts.

## Why this exists

Before this module, portfolio aggressiveness was expressed as
``risk_aversion=8.0, max_weight=0.06, max_active_share=0.60, universe_size=200``
retyped by hand in `real_backtest.py`, `live_rebalance.py`, `portfolio_view.py`
and `worked_example.py`. Four copies of a preference is four chances for the
live book to be running a different risk appetite than the one that was
backtested, with nothing in the output saying so.

A :class:`RiskProfile` is that preference, named once, hashable into the trial
registry, and carried into both the optimizer and the backtest config.

## What risk appetite can and cannot buy in *this* engine

Worth being exact, because "high risk appetite" means something narrower here
than it does in a general portfolio conversation.

The book is **long-only, fully invested, unlevered** — ``sum(w) = 1``, ``w >= 0``
is structural in `qe_backtest.optimize`, not a setting. So total exposure to the
equity market is ~100% at *every* profile. Nothing in this file changes how much
market risk is taken in the first-order sense; that decision was made by holding
equities at all.

What the profiles actually move is two things:

1. **Active risk** — `risk_aversion`, `max_weight`, `max_active_share`,
   `universe_size`, `n_positions`. How far the book may sit from a broad
   equal-weighted holding of the same names. This raises the *dispersion* of
   outcomes around the market. It raises expected return only to the extent the
   alpha is real, and it is not free: the cost model and the tax engine both
   charge for the extra turnover it induces.

2. **Market risk, deliberately** — `beta_premium_annual`. The one dial here that
   moves compensated risk. Setting it positive states a belief that high-beta
   names earn a premium and tells the optimizer to tilt toward them, trading
   that premium off against the variance it adds. A long-horizon investor who
   can sit through drawdowns is exactly who should hold a beta > 1 book, and
   unlike concentration it is a risk the market is supposed to pay for.

Two knobs are carried on the profile but held **constant across every shipped
profile**, on purpose:

- `score_ic` — the information coefficient. That is a claim about how good the
  forecast is, not about how much risk is wanted. Raising it because one feels
  aggressive is not risk appetite, it is lying to the optimizer about the
  forecast, and the optimizer will size positions on the lie.
- `trade_cost` — the optimizer's internal view of trading cost. That is an
  estimate of the world. Wanting more risk does not make spreads narrower.

They live on the profile so that one object fully determines portfolio
construction and the trial hash is complete.

## The empirical warning, in this project's own numbers

`docs/real_backtest.md`, 1975-2025: equal weight net of costs returned
**-2.21% CAGR** at 9.2x turnover, while the optimizer returned **6.94%** at
1.2x. The entire difference is trading restraint. Lowering `risk_aversion` and
raising `max_active_share` push turnover back toward the number that produced
the negative run. Aggression in this engine is therefore **not** a monotone
trade of variance for expected return — past some point it buys variance and
gives back return.

That is a claim about a specific engine on specific data, so it is testable
rather than assertable: `scripts/sweep_risk_appetite.py` runs the ladder and
prints where the turn happens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

__all__ = ["PROFILES", "RiskProfile", "load_profile"]

#: A beta premium larger than this is treated as a units error rather than a
#: view. The equity risk premium is ~5-6%/yr in total; an expected excess return
#: of more than 20%/yr for a one-standard-deviation beta tilt is not a bullish
#: assumption, it is a decimal in the wrong place.
MAX_PLAUSIBLE_BETA_PREMIUM = 0.20


@dataclass(frozen=True)
class RiskProfile:
    """One named point on the risk-appetite ladder.

    Every field is a *preference* or a *belief*, never a fact about the data.
    Anything that would change if the world changed rather than if the investor
    changed does not belong here.
    """

    name: str

    # -- active risk ------------------------------------------------------
    #: Risk aversion in the mean-variance objective. Lower = more alpha-chasing,
    #: more concentrated, higher turnover. In the units the scripts actually run
    #: in (a monthly risk model), matching the values this repo backtested with.
    risk_aversion: float
    #: Maximum weight in any one name.
    max_weight: float
    #: Cap on ``sum|w - equal_weight| / 2`` over the optimized universe. **This
    #: is the binding concentration dial in this codebase**, more than
    #: `max_weight` is: at 0.60 against a 200-name universe the book is forced
    #: at least 40% of the way back to holding 200 names equally, no matter what
    #: the alpha says. Raising `max_weight` without raising this changes very
    #: little.
    max_active_share: float | None
    #: How many names the optimizer is allowed to consider — the top slice by
    #: alpha. Also sets the equal-weight reference `max_active_share` is
    #: measured against, so shrinking it concentrates the book twice over.
    universe_size: int
    #: Names held by the *equal-weight fallback* path in the backtest, used when
    #: no risk model is supplied. Kept on the profile so the fallback and the
    #: optimized book express the same appetite.
    n_positions: int

    # -- market risk ------------------------------------------------------
    #: Expected **annual** excess return for a portfolio carrying a +1 standard
    #: deviation exposure to the `beta` factor. Zero means "take no deliberate
    #: view on beta"; positive tilts the book toward high-beta names and lets
    #: the optimizer price that tilt against the variance it adds.
    #:
    #: Units are the trap here. This is per **standardized** exposure, not per
    #: unit of raw beta, because that is how `qe_risk.neutralize.standardize`
    #: leaves the column the optimizer sees. A cross-sectional beta spread of
    #: ~0.35 against a ~6%/yr equity premium puts the honest range at roughly
    #: 0-3%/yr; :data:`MAX_PLAUSIBLE_BETA_PREMIUM` rejects the rest.
    beta_premium_annual: float

    # -- beliefs, deliberately constant across profiles -------------------
    #: One-way trading cost inside the objective. An estimate of the world.
    trade_cost: float = 0.005
    #: Information coefficient used to convert scores to expected returns. A
    #: claim about forecast quality. See the module docstring for why neither of
    #: these moves with appetite.
    score_ic: float = 0.03

    def __post_init__(self) -> None:
        if self.risk_aversion <= 0:
            raise ValueError(f"{self.name}: risk_aversion must be > 0")
        if not 0 < self.max_weight <= 1:
            raise ValueError(f"{self.name}: max_weight must be in (0, 1]")
        if self.max_active_share is not None and not 0 < self.max_active_share <= 1:
            raise ValueError(f"{self.name}: max_active_share must be in (0, 1] or None")
        if self.n_positions < 1:
            raise ValueError(f"{self.name}: n_positions must be >= 1")
        if self.universe_size < self.n_positions:
            raise ValueError(
                f"{self.name}: universe_size {self.universe_size} is below "
                f"n_positions {self.n_positions}; the optimizer would be asked to "
                "pick more names than it is allowed to look at"
            )
        # A book of `n` names each capped at `max_weight` cannot be fully
        # invested unless the caps sum past 1. The optimizer relaxes this at run
        # time rather than returning an unfunded book, but a profile that needs
        # relaxing is a profile that is not saying what it thinks it says.
        if self.max_weight * self.n_positions < 1.0:
            raise ValueError(
                f"{self.name}: {self.n_positions} names capped at "
                f"{self.max_weight:.0%} sum to "
                f"{self.max_weight * self.n_positions:.2f}, so the book cannot be "
                f"fully invested. Raise max_weight above {1 / self.n_positions:.3f}."
            )
        if not math.isfinite(self.beta_premium_annual):
            raise ValueError(f"{self.name}: beta_premium_annual must be finite")
        if abs(self.beta_premium_annual) > MAX_PLAUSIBLE_BETA_PREMIUM:
            raise ValueError(
                f"{self.name}: beta_premium_annual of "
                f"{self.beta_premium_annual:.3g} is {self.beta_premium_annual:.0%} "
                "per year for a one-SD beta tilt. That is above the whole equity "
                "risk premium and is almost certainly a units error — this field "
                "is an annual rate per standardized exposure, not a beta level "
                "and not a monthly figure."
            )
        if self.trade_cost < 0:
            raise ValueError(f"{self.name}: trade_cost must be >= 0")
        if not 0 <= self.score_ic <= 0.2:
            raise ValueError(
                f"{self.name}: score_ic of {self.score_ic} is outside [0, 0.2]. "
                "A genuinely good equity signal runs 0.02-0.05; above 0.10 is "
                "usually a backtest measuring itself."
            )

    def beta_premium_per_period(self, periods_per_year: int) -> float:
        """The beta premium in the alpha's own units.

        `periods_per_year` has no default and never will. Alpha reaches the
        optimizer in per-period return units, and every serious bug this project
        has had was a number of the wrong size that still looked plausible — a
        premium applied twelve times too large would simply produce a
        high-beta book and no error at all.
        """
        if periods_per_year < 1:
            raise ValueError("periods_per_year must be >= 1")
        return self.beta_premium_annual / float(periods_per_year)

    def payload(self) -> dict[str, Any]:
        """Hashable description, for `qe_eval.registry.config_hash`.

        Sweeping risk appetite and keeping the best profile *is* a search over
        portfolio configurations, and a search that is not counted is a search
        that deflates nothing. Every field is included, so changing any of them
        is a new trial.
        """
        return {
            "risk_profile": self.name,
            "risk_aversion": self.risk_aversion,
            "max_weight": self.max_weight,
            "max_active_share": self.max_active_share,
            "universe_size": self.universe_size,
            "n_positions": self.n_positions,
            "beta_premium_annual": self.beta_premium_annual,
            "trade_cost": self.trade_cost,
            "score_ic": self.score_ic,
        }

    def with_(self, **changes: Any) -> RiskProfile:
        """A modified copy, revalidated. Renames itself unless `name` is given."""
        if "name" not in changes:
            changes["name"] = f"{self.name}+custom"
        return replace(self, **changes)

    def describe(self) -> str:
        beta = (
            "beta-neutral"
            if self.beta_premium_annual == 0
            else f"beta tilt {self.beta_premium_annual:+.1%}/yr per SD"
        )
        active = (
            "unbounded" if self.max_active_share is None
            else f"{self.max_active_share:.0%}"
        )
        return (
            f"{self.name}: lambda {self.risk_aversion:g}, cap {self.max_weight:.0%}, "
            f"active share <= {active}, universe {self.universe_size}, {beta}"
        )


#: The shipped ladder. Defined in code rather than YAML so the defaults cannot
#: drift without a test noticing — `tests/test_risk_appetite.py` pins every
#: number here, and `balanced` in particular is pinned to reproduce the
#: `docs/real_backtest.md` run exactly.
PROFILES: dict[str, RiskProfile] = {
    # Wide, heavily diversified, low turnover. Gives up most of the alpha to
    # keep the book close to a broad equal-weighted holding.
    "conservative": RiskProfile(
        name="conservative",
        risk_aversion=20.0,
        max_weight=0.04,
        max_active_share=0.40,
        universe_size=300,
        n_positions=60,
        beta_premium_annual=0.0,
    ),
    # The configuration every existing result in this repo was produced at.
    # Changing these numbers invalidates `docs/real_backtest.md`.
    "balanced": RiskProfile(
        name="balanced",
        risk_aversion=8.0,
        max_weight=0.06,
        max_active_share=0.60,
        universe_size=200,
        n_positions=40,
        beta_premium_annual=0.0,
    ),
    # Concentrated into the strongest alpha, with a deliberate beta tilt. The
    # universe shrinks as well as the caps loosening, because active share is
    # measured against equal weight *over the optimized universe* — leaving it
    # at 200 would let the blend pull a 25-name book back toward 200 names.
    "aggressive": RiskProfile(
        name="aggressive",
        risk_aversion=4.0,
        max_weight=0.09,
        max_active_share=0.80,
        universe_size=120,
        n_positions=25,
        beta_premium_annual=0.015,
    ),
    # The end of the ladder that this codebase's own defences are written
    # against. Read `qe_backtest.optimize`'s "the thing that will bite" before
    # running real money here: the position cap, the cost term and the active
    # share bound are the three things holding mean-variance error maximization
    # in check, and this profile loosens all three at once.
    "max_growth": RiskProfile(
        name="max_growth",
        risk_aversion=2.0,
        max_weight=0.15,
        max_active_share=0.95,
        universe_size=60,
        n_positions=15,
        beta_premium_annual=0.030,
    ),
}


def load_profile(spec: str | Path | RiskProfile) -> RiskProfile:
    """Resolve a profile by name, or from a YAML file for one-off experiments.

    A name must be one of :data:`PROFILES`. Anything else is read as a path to a
    YAML mapping of the dataclass fields — which exists so that sweeping a knob
    does not require editing code, not so that the live book can quietly run a
    configuration nothing was ever backtested at. Loading from a file names the
    profile after the file, so the trial hash and every printed report still say
    where the numbers came from.
    """
    if isinstance(spec, RiskProfile):
        return spec
    key = str(spec)
    if key in PROFILES:
        return PROFILES[key]
    path = Path(key).expanduser()
    if not path.exists():
        raise KeyError(
            f"unknown risk profile {key!r}. Known: {sorted(PROFILES)}; "
            "anything else is read as a path to a YAML profile, and that file "
            "does not exist."
        )
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path}: a risk profile must be a YAML mapping")
    data.setdefault("name", path.stem)
    known = set(RiskProfile.__dataclass_fields__)
    unknown = set(data) - known
    if unknown:
        raise ValueError(
            f"{path}: unknown risk-profile fields {sorted(unknown)}. "
            f"Known fields: {sorted(known)}. Refusing rather than ignoring them, "
            "because an ignored field reads exactly like an applied one."
        )
    return RiskProfile(**data)
