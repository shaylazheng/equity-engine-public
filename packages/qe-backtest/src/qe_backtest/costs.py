"""Trading costs — the part most backtests get wrong.

A flat basis-point assumption is not acceptable here, because the whole
structural edge of a small account is holding names institutions cannot. Those
names have wide spreads, and a cost model that assumes institutional liquidity
will report an edge that does not exist. The cost model has to be *accurate at
small size*, which mostly means accurate about spreads on illiquid names.

Three components:

- **Half-spread** — paid on every trade. Historically estimated from high/low via
  Corwin-Schultz; here a liquidity-scaled approximation until real quotes land in
  Phase 1.
- **Market impact** — the square-root law, ``impact ~ sigma * sqrt(Q / ADV)``.
  Roughly linear in size for small orders and concave for large ones, which is
  what actually happens.
- **Commission** — broker schedule, per share with a per-order minimum.

Plus a **participation cap**: an order larger than a set fraction of daily volume
cannot be filled in one day, so the remainder spills to subsequent sessions.
Ignoring this is how a backtest silently assumes infinite liquidity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["CostBreakdown", "CostModel"]


@dataclass(frozen=True)
class CostBreakdown:
    spread: float
    impact: float
    commission: float

    @property
    def total(self) -> float:
        return self.spread + self.impact + self.commission

    def __repr__(self) -> str:
        return (
            f"<Cost {self.total:,.2f} "
            f"(spread {self.spread:,.2f}, impact {self.impact:,.2f}, "
            f"commission {self.commission:,.2f})>"
        )


@dataclass(frozen=True)
class CostModel:
    """Per-trade cost in dollars."""

    #: Baseline half-spread in basis points for a liquid, large-cap name.
    base_half_spread_bps: float = 2.0
    #: Additional half-spread as size falls. Applied against the reference cap
    #: below, so a $50m name pays substantially more than a $5bn one.
    illiquidity_bps_at_reference: float = 25.0
    #: **In $ millions**, matching `qe_data.panel.PRICE_SCALES` and the synthetic
    #: generator. Both panels carry `mktcap` in millions, and this constant was
    #: originally 1e9 — raw dollars. The mismatch did not raise, it just made
    #: every name in every backtest look like a sub-$1 microcap and pay the
    #: capped 300bp half-spread: a $50bn company was charged 3% one-way. Keep the
    #: unit of this constant and the unit of the `mktcap` you pass identical.
    reference_mktcap: float = 1_000.0
    #: Cap so a microcap does not produce an absurd spread.
    max_half_spread_bps: float = 300.0

    #: Square-root impact coefficient. ~1.0 with daily vol and Q/ADV is the
    #: conventional calibration.
    impact_coefficient: float = 1.0

    commission_per_share: float = 0.005
    commission_minimum: float = 0.0

    #: Maximum fraction of a day's volume a single order may consume.
    participation_cap: float = 0.10

    def half_spread_bps(self, mktcap: float) -> float:
        """Half-spread widens as size falls, on a square-root scaling."""
        if not np.isfinite(mktcap) or mktcap <= 0:
            return self.max_half_spread_bps
        scale = np.sqrt(self.reference_mktcap / mktcap)
        bps = self.base_half_spread_bps + self.illiquidity_bps_at_reference * scale
        return float(min(bps, self.max_half_spread_bps))

    def impact_bps(self, notional: float, adv: float, volatility: float) -> float:
        """Square-root law. `volatility` is a daily standard deviation."""
        if adv <= 0 or notional <= 0 or not np.isfinite(adv):
            return 0.0
        participation = notional / adv
        return float(self.impact_coefficient * volatility * np.sqrt(participation) * 10_000)

    def cost(
        self,
        *,
        shares: float,
        price: float,
        mktcap: float,
        adv: float,
        volatility: float = 0.02,
    ) -> CostBreakdown:
        """Total cost of trading `shares` at `price`. Sign-agnostic — buys and
        sells both pay."""
        shares = abs(shares)
        if shares == 0 or price <= 0:
            return CostBreakdown(0.0, 0.0, 0.0)

        notional = shares * price
        spread = notional * self.half_spread_bps(mktcap) / 10_000
        impact = notional * self.impact_bps(notional, adv, volatility) / 10_000
        commission = max(shares * self.commission_per_share, self.commission_minimum)
        return CostBreakdown(spread, impact, commission)

    def max_shares(self, adv_shares: float) -> float:
        """Largest order fillable in one session under the participation cap."""
        return max(adv_shares, 0.0) * self.participation_cap
