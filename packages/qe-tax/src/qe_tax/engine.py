"""The tax model: headroom, stacking, and a genuinely marginal rate.

The plan's original framing treated tax as pure friction — turnover is costly, so
trade less. At low income that is simply wrong, and wrong in an expensive
direction. Long-term gains have a **0% band** (2026 single: taxable income up to
$49,450 against a $16,100 standard deduction), so a filer on $20,000 has roughly
$45,550 of long-term gain realizable at zero federal tax every year. On a
portfolio in the tens of thousands that can exceed the entire unrealized gain:
rebalancing is free, and each realization steps up basis, shrinking future tax.

Short-term gains get no such band — they are ordinary income. A low earner pays a
low *rate* on them, never zero.

Three interactions make a flat rate unable to express any of this:

1. **Long-term gains stack on ordinary income**, so the rate on the next dollar
   depends on how much has already been realized this year. The marginal rate is
   a step function within the tax year, not a constant.
2. **Short-term gains consume long-term headroom** by raising taxable income.
   Realizing one carries a hidden second cost a flat model cannot see.
3. **Losses net by character** before crossing over, and the unused remainder
   carries forward indefinitely — state that persists across backtest years.

Rather than hand-derive the marginal rate through all of that, :meth:`TaxEngine.
marginal_rate` differentiates the exact total numerically. Every interaction is
then captured for free and stays correct if the bands change.

**Not tax advice.** A CPA validates an actual return.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .brackets import BracketTable, load_brackets, tax_from_bands

__all__ = ["Netting", "TaxBreakdown", "TaxEngine", "TaxProfile", "net_gains"]


@dataclass(frozen=True)
class TaxProfile:
    """Who is paying. Everything here is config the engine reads, never hardcoded."""

    filing_status: str = "single"
    #: Annual wage/other ordinary income, before any deduction.
    ordinary_income: float = 0.0
    #: Current portfolio market value. Sizes headroom against real unrealized gain
    #: and drives the cost model, which behaves very differently on a small account.
    portfolio_value: float = 0.0
    #: Flat approximation of state tax on gains. Most states give no preferential
    #: rate for long-term gains; New Jersey is one of them. Real state rules vary
    #: far too much to model generically, so this is deliberately crude.
    state_rate: float = 0.0
    tax_year: int = 2026
    #: Unused losses carried in, stored as positive magnitudes.
    carryforward_short: float = 0.0
    carryforward_long: float = 0.0

    def __post_init__(self) -> None:
        if self.ordinary_income < 0:
            raise ValueError("ordinary_income cannot be negative")
        if not 0.0 <= self.state_rate < 1.0:
            raise ValueError(f"state_rate must be in [0, 1), got {self.state_rate}")


@dataclass(frozen=True)
class Netting:
    """Result of netting gains and losses by character."""

    net_short: float
    net_long: float
    #: Capital loss applied against ordinary income this year (capped).
    ordinary_offset: float
    carryforward_short: float
    carryforward_long: float


@dataclass(frozen=True)
class TaxBreakdown:
    federal_ordinary: float
    federal_long_term: float
    niit: float
    state: float
    netting: Netting = field(repr=False)

    @property
    def total(self) -> float:
        return self.federal_ordinary + self.federal_long_term + self.niit + self.state


def net_gains(
    short: float,
    long: float,
    *,
    carryforward_short: float = 0.0,
    carryforward_long: float = 0.0,
    offset_cap: float = 3000.0,
) -> Netting:
    """Net gains and losses by character, then across, then against ordinary income.

    Order matters and is not arbitrary: short offsets short and long offsets long
    *first*, because short-term gains are taxed at the higher rate and a loss is
    worth more against them. Only the residual crosses over.
    """
    ns = short - carryforward_short
    nl = long - carryforward_long

    # Cross-character offset, only after same-character netting.
    if ns < 0 < nl:
        absorb = min(-ns, nl)
        ns += absorb
        nl -= absorb
    elif nl < 0 < ns:
        absorb = min(-nl, ns)
        nl += absorb
        ns -= absorb

    total = ns + nl
    if total >= 0:
        return Netting(ns, nl, 0.0, 0.0, 0.0)

    # Net loss: up to `offset_cap` reduces ordinary income, the rest carries
    # forward keeping its character.
    offset = min(-total, offset_cap)

    short_loss = -ns if ns < 0 else 0.0
    long_loss = -nl if nl < 0 else 0.0
    loss_total = short_loss + long_loss

    if loss_total <= 0:
        return Netting(0.0, 0.0, offset, 0.0, 0.0)

    # Short-term losses are consumed first by the ordinary offset — they are the
    # less valuable ones to carry, since a carried long loss can only ever shelter
    # long gains taxed at the lower rate.
    used_short = min(short_loss, offset)
    used_long = offset - used_short
    return Netting(
        net_short=0.0,
        net_long=0.0,
        ordinary_offset=offset,
        carryforward_short=max(short_loss - used_short, 0.0),
        carryforward_long=max(long_loss - used_long, 0.0),
    )


class TaxEngine:
    """Computes tax exactly, then differentiates it for the marginal rate."""

    def __init__(self, profile: TaxProfile, brackets: BracketTable | None = None,
                 *, allow_unverified: bool = False) -> None:
        self.profile = profile
        self.brackets = brackets or load_brackets(profile.tax_year)
        self.table = self.brackets.status(
            profile.filing_status, allow_unverified=allow_unverified
        )

    # -- the exact computation --------------------------------------------

    def total_tax(self, *, short_gain: float = 0.0, long_gain: float = 0.0) -> TaxBreakdown:
        p = self.profile
        netting = net_gains(
            short_gain,
            long_gain,
            carryforward_short=p.carryforward_short,
            carryforward_long=p.carryforward_long,
            offset_cap=self.brackets.ordinary_offset(p.filing_status),
        )

        agi_ordinary = max(p.ordinary_income - netting.ordinary_offset, 0.0)
        ordinary_taxable = max(agi_ordinary - self.table.standard_deduction, 0.0)
        ordinary_taxable += max(netting.net_short, 0.0)

        long_taxable = max(netting.net_long, 0.0)

        fed_ordinary = tax_from_bands(ordinary_taxable, self.table.ordinary)
        # Stacked: long-term gains sit on top of ordinary taxable income.
        fed_long = tax_from_bands(long_taxable, self.table.long_term, floor=ordinary_taxable)

        investment_income = max(netting.net_short, 0.0) + long_taxable
        magi = p.ordinary_income + investment_income
        over = max(magi - self.brackets.niit_threshold(p.filing_status), 0.0)
        niit = self.brackets.niit_rate * min(investment_income, over)

        state = p.state_rate * investment_income

        return TaxBreakdown(fed_ordinary, fed_long, niit, state, netting)

    # -- the derived quantities the optimizer needs -----------------------

    def marginal_rate(
        self, *, long_term: bool, short_gain: float = 0.0, long_gain: float = 0.0,
        eps: float = 1.0,
    ) -> float:
        """Tax on the *next* dollar of gain, given what is already realized this year.

        A numerical derivative of the exact total, so it picks up every
        interaction — including a short-term gain pushing long-term gains out of
        the 0% band, which is the one people miss.
        """
        base = self.total_tax(short_gain=short_gain, long_gain=long_gain).total
        if long_term:
            bumped = self.total_tax(short_gain=short_gain, long_gain=long_gain + eps).total
        else:
            bumped = self.total_tax(short_gain=short_gain + eps, long_gain=long_gain).total
        return (bumped - base) / eps

    def headroom(self, *, short_gain: float = 0.0, long_gain: float = 0.0) -> float:
        """Further long-term gain realizable at 0% federal this year.

        The annual budget the optimizer should be spending deliberately rather
        than avoiding. Ignores state tax and NIIT — it is a federal-band figure,
        and at any income where NIIT bites the headroom is zero anyway.
        """
        p = self.profile
        netting = net_gains(
            short_gain, long_gain,
            carryforward_short=p.carryforward_short,
            carryforward_long=p.carryforward_long,
            offset_cap=self.brackets.ordinary_offset(p.filing_status),
        )
        agi_ordinary = max(p.ordinary_income - netting.ordinary_offset, 0.0)
        ordinary_taxable = max(agi_ordinary - self.table.standard_deduction, 0.0)
        ordinary_taxable += max(netting.net_short, 0.0)

        used = ordinary_taxable + max(netting.net_long, 0.0)
        return max(self.table.zero_ltcg_ceiling - used, 0.0)

    def tax_attributable_to_gains(
        self, *, short_gain: float = 0.0, long_gain: float = 0.0
    ) -> float:
        """Tax caused *by the portfolio*, isolated from tax on everything else.

        `total_tax` is the investor's whole federal bill — it necessarily
        includes tax on ordinary income, because long-term gains stack on top of
        ordinary taxable income and the bands cannot be evaluated separately.
        That makes it the right number for computing a rate and the wrong number
        to charge a portfolio: a backtest that deducts it is billing the account
        for the owner's salary.

        The attributable figure is the difference between the bill with these
        gains and the bill without them, which keeps every interaction the
        stacking creates — including short-term gains pushing long-term gains out
        of the 0% band — while excluding what would have been owed anyway.
        """
        with_gains = self.total_tax(short_gain=short_gain, long_gain=long_gain).total
        without = self.total_tax(short_gain=0.0, long_gain=0.0).total
        return with_gains - without

    def tax_on_realization(
        self, gain: float, *, long_term: bool, short_gain: float = 0.0, long_gain: float = 0.0
    ) -> float:
        """Incremental tax from realizing `gain` on top of what is already realized."""
        before = self.total_tax(short_gain=short_gain, long_gain=long_gain).total
        if long_term:
            after = self.total_tax(short_gain=short_gain, long_gain=long_gain + gain).total
        else:
            after = self.total_tax(short_gain=short_gain + gain, long_gain=long_gain).total
        return after - before

    def next_year(self, netting: Netting, brackets: BracketTable | None = None) -> TaxEngine:
        """An engine for the following year, carrying unused losses forward.

        If `brackets` is omitted the **current table is reused** while the
        profile's year advances. That is a modelling assumption, not a fact —
        real rates change annually — but it is the right default for a
        forward-looking engine, where next year's brackets genuinely do not exist
        yet. Loading them implicitly would be worse: it would either fail on a
        missing file or silently apply the wrong year.

        :attr:`brackets_stale` reports when this has happened, so a tearsheet can
        say "2028 modelled on 2026 rates" rather than implying precision it lacks.
        Pass the real table as soon as it is published.
        """
        return TaxEngine(
            replace(
                self.profile,
                carryforward_short=netting.carryforward_short,
                carryforward_long=netting.carryforward_long,
                tax_year=self.profile.tax_year + 1,
            ),
            brackets or self.brackets,
            allow_unverified=not self.table.verified,
        )

    @property
    def brackets_stale(self) -> bool:
        """True when the bracket table is not for the profile's own tax year."""
        return self.brackets.tax_year != self.profile.tax_year

    def __repr__(self) -> str:
        p = self.profile
        return (
            f"<TaxEngine {p.filing_status} {p.tax_year}, income {p.ordinary_income:,.0f}, "
            f"headroom {self.headroom():,.0f}>"
        )
