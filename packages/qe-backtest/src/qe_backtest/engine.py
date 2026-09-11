"""Event-driven daily backtester with explicit state.

Not a vectorized returns multiplier. Long-only constraints, tax lots,
participation caps, and partial fills are all path-dependent — the answer depends
on the order things happened — so the loop is required rather than preferred.

Two rules are structural rather than configurable:

**Never same-bar.** A signal computed at the close on day t produces an order
that fills on day t+1. Orders live in a pending queue precisely so it is
impossible to accidentally fill at a price the signal already saw.

**Long-only.** Target weights are non-negative and the lot book refuses to sell
what it does not hold, so a short position cannot arise from a bug in weight
construction.

The tax-aware lot policy is where `qe-tax` earns its place. With unused 0%
headroom the engine sells *lowest*-cost lots — realizing the largest long-term
gain, tax-free, and stepping up basis. Once headroom is exhausted it switches to
highest-cost lots to minimize the taxable gain. A flat tax assumption cannot
express that switch, and defaulting to "minimize gains" leaves free basis step-up
on the table every year.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from qe_core.risk_appetite import RiskProfile
from qe_tax.engine import TaxEngine
from qe_tax.lots import LotBook, LotMethod

from .costs import CostModel
from .optimize import OptimizerConfig, RiskInputs, optimize_weights, scores_to_alpha

__all__ = ["BacktestConfig", "BacktestResult", "run_backtest"]


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 100_000.0
    #: Names held at target. A small account can hold fewer, more concentrated.
    n_positions: int = 25
    max_weight: float = 0.10
    #: *Minimum* trading days between rebalances. A rebalance happens when a
    #: score is available and this much time has passed — the score panel drives
    #: the schedule, not an internal counter. Requiring the caller's score dates
    #: to land on a modulo grid the engine picked is a silent way to do nothing.
    rebalance_every: int = 21
    min_price: float = 5.0
    #: Base lot-selection method once 0% headroom is exhausted.
    lot_method: LotMethod = "hifo"
    #: Skip trades whose value is below this — churn that only pays commission.
    min_trade_value: float = 100.0

    @classmethod
    def from_profile(cls, profile: RiskProfile, **overrides) -> BacktestConfig:
        """Take the two appetite-bearing fields from a :class:`qe_core.RiskProfile`.

        Only `n_positions` and `max_weight` come from the profile; cash, the
        rebalance interval, the price floor and the minimum trade are facts about
        the account and the data, not about how much risk is wanted. Pair with
        `OptimizerConfig.from_profile` — feeding the profile to one and not the
        other leaves the equal-weight fallback path expressing a different
        appetite than the optimized path.
        """
        return cls(
            n_positions=profile.n_positions,
            max_weight=profile.max_weight,
            **overrides,
        )


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    weights: pd.DataFrame
    costs: pd.Series
    #: Running *accrued* tax on the year to date — a mark, not a payment. The
    #: money that actually left the account is `tax_paid`; keeping the two named
    #: differently is deliberate, because conflating them is what let a "net of
    #: tax" run out-return its own pre-tax twin.
    realized_tax: pd.Series
    #: Deliberately has no default: an omitted lot book would silently read as an
    #: empty portfolio rather than failing, so the defaulted fields go after it.
    lots: LotBook = field(repr=False)
    #: Total tax actually deducted from the account over the run.
    tax_paid: float = 0.0
    #: One row per tax year settled.
    tax_payments: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def returns(self) -> pd.Series:
        return self.equity.pct_change().dropna()

    def turnover(self) -> float:
        """Annualized one-way turnover as a fraction of average equity."""
        if self.trades.empty:
            return 0.0
        mean_equity = float(self.equity.mean())
        if not np.isfinite(mean_equity) or mean_equity <= 0:
            return float("nan")
        traded = float(self.trades["notional"].abs().sum())
        years = max((self.equity.index[-1] - self.equity.index[0]).days / 365.25, 1e-9)
        return traded / mean_equity / years

    def __repr__(self) -> str:
        total = self.equity.iloc[-1] / self.equity.iloc[0] - 1.0
        return (
            f"<BacktestResult {len(self.equity)} days, total return {total:+.1%}, "
            f"turnover {self.turnover():.1f}x/yr, {len(self.trades)} trades>"
        )


def sale_tax_cost(
    book: LotBook,
    prices: pd.Series,
    date: pd.Timestamp,
    tax_engine: TaxEngine,
    *,
    ytd_short: float = 0.0,
    ytd_long: float = 0.0,
) -> pd.Series:
    """Tax cost of selling each held name, per dollar of position value.

    This is what makes portfolio construction holding-period aware. Lot
    *selection* was already tax-aware -- given that a sale happens, it picks the
    cheapest lots -- but nothing upstream ever asked whether the sale was worth
    making. The optimizer priced every exit at the same spread whether it
    realized a short-term gain at ordinary rates or a long-term one inside the 0%
    band.

    For each lot: gain = (price - basis) * shares, taxed at
    `TaxEngine.marginal_rate` for that lot's holding period. Summed per name and
    divided by position value, which linearizes the bill into the same units as
    the spread so it can enter the objective's linear cost term directly.

    **Losses come back negative and are kept that way.** Realizing a loss offsets
    other gains, so a position underwater is genuinely cheaper to leave than a
    flat one, and harvesting it is a real benefit rather than a cost to be
    floored at zero. The optimizer clamps the total at zero only to keep its
    threshold well-posed.

    The linearization is the approximation here: marginal rates are a step
    function, so selling a large enough position at once moves the rate that
    prices it. At position sizes a taxable account trades, the step is rarely
    crossed inside one name.
    """
    out: dict[int, float] = {}
    for permno, shares in book.holdings().items():
        price = prices.get(permno, np.nan)
        if not np.isfinite(price) or price <= 0 or shares <= 0:
            continue
        value = shares * float(price)
        if value <= 0:
            continue

        bill = 0.0
        for lot in book.lots(permno):
            gain = (float(price) - lot.basis_per_share) * lot.shares
            if gain == 0.0:
                continue
            rate = tax_engine.marginal_rate(
                long_term=lot.is_long_term(date),
                short_gain=ytd_short,
                long_gain=ytd_long,
            )
            bill += rate * gain
        out[permno] = bill / value
    return pd.Series(out, dtype=float)


def _target_weights(
    scores: pd.Series,
    cfg: BacktestConfig,
    eligible: pd.Index,
    *,
    risk: RiskInputs | None = None,
    current: pd.Series | None = None,
    opt_cfg: OptimizerConfig | None = None,
    sell_cost: pd.Series | None = None,
) -> pd.Series:
    """Target book for one rebalance.

    Two constructions, and which one runs depends on whether a risk model was
    supplied for this date:

    - **With a risk model** — the mean-variance optimizer in `optimize.py`.
    - **Without one** — top-N by score, equal weighted, capped.

    The fallback is kept rather than removed because it is the honest thing to do
    when there is no covariance: an optimizer handed a made-up risk model is
    worse than equal weight, not better, since it will confidently concentrate on
    whatever the fiction says is safe. Early backtest dates genuinely have no
    fitted model, and that is a real state, not an error.

    Long-only in both branches by construction.
    """
    valid = scores.reindex(eligible).dropna()

    if risk is not None:
        held = pd.Series(dtype=float) if current is None else current[current > 1e-9]
        # Held names re-enter even when ineligible today, with a no-buy bound at
        # their current weight. Ineligibility (price floor, wash-sale block) is a
        # reason not to *add*, not a reason to dump at any price — and dropping
        # them would force a sale the cost term never got to price.
        universe = valid.index.union(held.index.intersection(scores.dropna().index))
        alpha = scores.reindex(universe).dropna()
        if alpha.empty:
            return pd.Series(dtype=float)

        blocked = alpha.index.intersection(held.index.difference(eligible))
        caps = pd.Series(np.nan, index=alpha.index, dtype=float)
        if len(blocked):
            # Guarded: pandas 3.0 raises on assigning an empty slice into a
            # float column rather than treating it as a no-op.
            caps.loc[blocked] = held.reindex(blocked)

        cfg_opt = opt_cfg or OptimizerConfig(max_weight=cfg.max_weight)
        if cfg_opt.score_ic is not None:
            # Scores are z-scores or percentiles; the optimizer needs expected
            # returns. Without this the alpha term is ~1.0 against a risk term of
            # ~1e-4 and the risk model, the cost term, and risk aversion all stop
            # mattering — silently, since the weights still look fine.
            _, _, spec = risk.align(alpha.index)
            alpha = scores_to_alpha(
                alpha,
                ic=cfg_opt.score_ic,
                specific_var=pd.Series(spec, index=alpha.index),
            )
        extra = None
        if sell_cost is not None and cfg_opt.tax_aware:
            extra = sell_cost.reindex(alpha.index) + float(cfg_opt.trade_cost)
        result = optimize_weights(
            alpha, risk, current=current, max_weights=caps,
            sell_cost=extra, config=cfg_opt,
        )
        return result.weights

    if valid.empty:
        return pd.Series(dtype=float)

    top = valid.nlargest(min(cfg.n_positions, len(valid)))
    if top.empty:
        return pd.Series(dtype=float)

    w = pd.Series(1.0 / len(top), index=top.index)
    return w.clip(upper=cfg.max_weight)


def run_backtest(
    scores: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    mktcap: pd.DataFrame | None = None,
    adv: pd.DataFrame | None = None,
    config: BacktestConfig | None = None,
    cost_model: CostModel | None = None,
    tax_engine: TaxEngine | None = None,
    risk: Callable[[pd.Timestamp], RiskInputs | None] | None = None,
    optimizer: OptimizerConfig | None = None,
    delisting_returns: pd.Series | None = None,
) -> BacktestResult:
    """Run the daily loop.

    Parameters
    ----------
    scores, prices:
        date x permno. `scores` may be sparse — only rebalance dates need values.
    mktcap:
        **In $ millions**, matching `qe_data.panel.PRICE_SCALES`, the synthetic
        generator, and `CostModel.reference_mktcap`. Feeding raw dollars makes
        every name look like a microcap and charges the capped half-spread.
    adv:
        Average daily volume in **shares**, not dollars. Notional is computed as
        `adv * price` and the participation cap is a fraction of share volume.
    tax_engine:
        When supplied, lot selection becomes headroom-aware and realized tax is
        accrued. Without it the run is pre-tax, which is only meaningful for
        replication work.
    risk:
        `date -> RiskInputs | None`. Returning a model switches construction to
        the mean-variance optimizer for that date; returning `None` falls back to
        top-N equal weight. A callable rather than a frame because exposures and
        specific risk are re-estimated every period, and because returning `None`
        for early dates is the correct answer rather than an error -- a model
        fitted on data that did not exist yet is the leak this repo is built to
        prevent.
    optimizer:
        Optimizer settings. Defaults inherit `config.max_weight` so the position
        limit cannot silently differ between the two construction paths.
    delisting_returns:
        permno -> return realized on delisting, applied to the liquidation price.

        Without this the engine closes a delisted holding at its last *quoted*
        price, and a price series simply stops when a company fails — so
        bankruptcies exit the book at their final good mark and the run inherits
        a survivorship bias pointing the flattering way. CRSP carries the real
        number (`crsp.stkdelists`), and where it is missing the Shumway
        convention supplies -30% on NYSE/AMEX and -55% on NASDAQ. Omitting this
        argument is a modelling choice that should be deliberate, not a default.
    """
    cfg = config or BacktestConfig()
    costs = cost_model or CostModel()
    book = LotBook(cfg.lot_method)

    # Coerce to plain float64 up front. CRSP parquet arrives as pandas *nullable*
    # Float64, where a missing price is `pd.NA` rather than `nan` — and
    # `np.isfinite(pd.NA)` returns NA, so `if not np.isfinite(price)` raises
    # "boolean value of NA is ambiguous". The synthetic panel is plain float64,
    # so every test passed while the real panel could not get through the first
    # bar. Done here rather than at each call site because this is the boundary
    # where outside data enters the loop.
    prices = prices.astype("float64")
    scores = scores.astype("float64")
    if mktcap is not None:
        mktcap = mktcap.astype("float64")
    if adv is not None:
        adv = adv.astype("float64")

    dates = prices.index
    cash = cfg.initial_cash
    pending: dict[int, float] = {}

    #: Last observable price per name, so a delisting can be closed out at a
    #: real number rather than marked at NaN or zero.
    last_price: dict[int, float] = {}
    delisted: list[dict] = []

    equity_rows: dict[pd.Timestamp, float] = {}
    weight_rows: dict[pd.Timestamp, pd.Series] = {}
    cost_rows: dict[pd.Timestamp, float] = {}
    tax_rows: dict[pd.Timestamp, float] = {}
    trades: list[dict] = []

    ytd_short = ytd_long = 0.0
    tax_paid = 0.0
    tax_payments: list[dict] = []
    current_year = dates[0].year
    last_rebalance = -(10**9)  # so the first scored date always rebalances

    for i, date in enumerate(dates):
        px = prices.loc[date]

        # New tax year: settle last year's bill, then reset.
        #
        # The bill is **paid out of the account**. Previously it was accrued,
        # reported, and never deducted -- `realized_tax` was a number in a
        # report that no cash movement corresponded to, so a run labelled "net
        # of tax" was not, and could even out-return its own pre-tax twin
        # because tax-aware lot selection changed *which* shares were sold.
        #
        # Deducting from the portfolio is the conservative reading, and the
        # right one for measuring a strategy: if the bill is actually paid from
        # salary the portfolio does better while the investor is poorer by the
        # same amount, so the strategy's economics are unchanged. Settled at the
        # year boundary rather than the April filing date -- earlier, so erring
        # against the strategy.
        if date.year != current_year:
            if tax_engine is not None:
                bill = tax_engine.tax_attributable_to_gains(
                    short_gain=ytd_short, long_gain=ytd_long
                )
                cash -= bill
                tax_paid += bill
                tax_payments.append({"year": current_year, "tax": bill})
            current_year = date.year
            ytd_short = ytd_long = 0.0

        # --- fill yesterday's orders at today's price ---------------------
        day_cost = 0.0
        day_tax = 0.0
        if pending:
            adv_row = adv.loc[date] if adv is not None else None
            cap_row = mktcap.loc[date] if mktcap is not None else None
            unfilled: dict[int, float] = {}

            for permno, shares in pending.items():
                price = px.get(permno, np.nan)
                if not np.isfinite(price) or price <= 0:
                    unfilled[permno] = shares
                    continue

                # Participation cap: the remainder spills to the next session.
                if adv_row is not None and np.isfinite(adv_row.get(permno, np.nan)):
                    limit = costs.max_shares(adv_row[permno])
                    if abs(shares) > limit:
                        fill = np.sign(shares) * limit
                        unfilled[permno] = shares - fill
                        shares = fill
                if abs(shares * price) < cfg.min_trade_value:
                    continue

                cap = cap_row.get(permno, np.nan) if cap_row is not None else np.nan
                adv_notional = (
                    adv_row.get(permno, np.nan) * price if adv_row is not None else np.nan
                )
                breakdown = costs.cost(
                    shares=shares,
                    price=price,
                    mktcap=cap if np.isfinite(cap) else 0.0,
                    adv=adv_notional if np.isfinite(adv_notional) else 0.0,
                )
                day_cost += breakdown.total

                if shares > 0:
                    spend = shares * price + breakdown.total
                    if spend > cash:
                        # Cash-starved: buy what is affordable, which can be
                        # nothing. The zero check is separate from
                        # `min_trade_value` on purpose — that setting is an
                        # economic threshold ("not worth the commission"), and
                        # letting it double as the validity check meant setting
                        # it to 0 passed a zero-share order to the lot book.
                        shares = max((cash - breakdown.total) / price, 0.0)
                        if shares <= 0 or shares * price < cfg.min_trade_value:
                            continue
                        spend = shares * price + breakdown.total
                    book.buy(permno, shares, price, date)
                    cash -= spend
                else:
                    qty = min(-shares, book.shares(permno))
                    if qty <= 0:
                        continue
                    method = _lot_method(cfg, tax_engine, ytd_short, ytd_long)
                    sale = book.sell(permno, qty, price, date, method=method)
                    cash += sale.proceeds - breakdown.total
                    ytd_short += sale.short_gain
                    ytd_long += sale.long_gain

                trades.append(
                    {
                        "date": date, "permno": permno, "shares": shares,
                        "price": price, "notional": shares * price,
                        "cost": breakdown.total,
                    }
                )

            pending = unfilled

        # --- delisted and suspended names ------------------------------------
        # A held name that stops quoting is closed out at its last known price,
        # not carried. Real securities leave the panel — CRSP stops quoting on
        # delisting — and there are only three things the loop can do about it:
        # carry a stale mark forever (a zombie position that never sells), mark
        # it at zero (assert a total loss that usually did not happen), or
        # liquidate at the last observable price. The third is what actually
        # happens to the account, so it is what the engine does.
        #
        # Before this existed the position was simply marked at `NaN`, because
        # `px.get(p, 0.0)` returns the *stored* NaN when the column exists — the
        # default only fires on a missing key. One delisting turned equity into
        # NaN permanently, and every statistic downstream silently described only
        # the period before the first one.
        for permno, shares in list(book.holdings().items()):
            price = px.get(permno, np.nan)
            if np.isfinite(price) and price > 0:
                last_price[permno] = float(price)
                continue
            stale = last_price.get(permno)
            if stale is None or shares <= 0:
                continue
            if delisting_returns is not None:
                dr = delisting_returns.get(permno, np.nan)
                if np.isfinite(dr):
                    stale = max(stale * (1.0 + float(dr)), 0.0)
            method = _lot_method(cfg, tax_engine, ytd_short, ytd_long)
            sale = book.sell(permno, shares, stale, date, method=method)
            cash += sale.proceeds
            ytd_short += sale.short_gain
            ytd_long += sale.long_gain
            delisted.append({"date": date, "permno": permno, "price": stale})
            trades.append(
                {
                    "date": date, "permno": permno, "shares": -shares,
                    "price": stale, "notional": -shares * stale, "cost": 0.0,
                }
            )
            pending.pop(permno, None)

        # --- mark to market ------------------------------------------------
        holdings = book.holdings()
        marks = {p: last_price.get(p, 0.0) for p in holdings}
        for permno in holdings:
            price = px.get(permno, np.nan)
            if np.isfinite(price) and price > 0:
                marks[permno] = float(price)

        market_value = sum(sh * marks[p] for p, sh in holdings.items())
        equity = cash + market_value
        if not np.isfinite(equity):
            raise ValueError(
                f"equity became non-finite on {date:%Y-%m-%d} — cash={cash}, "
                f"holdings={len(holdings)}. This is a marking bug, not a market "
                "event; a silent NaN would make every statistic after this date "
                "describe a shorter sample than it claims."
            )
        equity_rows[date] = equity
        cost_rows[date] = day_cost

        if tax_engine is not None:
            day_tax = tax_engine.tax_attributable_to_gains(
                short_gain=ytd_short, long_gain=ytd_long
            )
        tax_rows[date] = day_tax

        weight_rows[date] = (
            pd.Series({p: sh * marks[p] / equity for p, sh in holdings.items()})
            if equity > 0
            else pd.Series(dtype=float)
        )

        # --- decide tomorrow's orders --------------------------------------
        if (
            date not in scores.index
            or i - last_rebalance < cfg.rebalance_every
            or i == len(dates) - 1
        ):
            continue
        last_rebalance = i

        eligible = px[(px >= cfg.min_price) & np.isfinite(px)].index
        eligible = eligible.difference(pd.Index(sorted(book.blocked_names(date))))

        risk_inputs = risk(date) if risk is not None else None
        tax_cost = None
        if risk_inputs is not None and tax_engine is not None:
            tax_cost = sale_tax_cost(
                book, px, date, tax_engine,
                ytd_short=ytd_short, ytd_long=ytd_long,
            )
        target = _target_weights(
            scores.loc[date],
            cfg,
            eligible,
            risk=risk_inputs,
            current=weight_rows[date],
            opt_cfg=optimizer,
            sell_cost=tax_cost,
        )
        if target.empty:
            continue

        pending = _orders(book, target, px, equity, cfg)

    # The final (partial) tax year still owes. Leaving it unsettled would make a
    # run's last year the only one that looks tax-free.
    if tax_engine is not None and (ytd_short or ytd_long):
        bill = tax_engine.tax_attributable_to_gains(
            short_gain=ytd_short, long_gain=ytd_long
        )
        tax_paid += bill
        tax_payments.append({"year": current_year, "tax": bill})
        last = dates[-1]
        equity_rows[last] = equity_rows[last] - bill

    return BacktestResult(
        equity=pd.Series(equity_rows, name="equity"),
        trades=pd.DataFrame(trades),
        weights=pd.DataFrame(weight_rows).T.fillna(0.0),
        costs=pd.Series(cost_rows, name="cost"),
        realized_tax=pd.Series(tax_rows, name="tax"),
        lots=book,
        tax_paid=tax_paid,
        tax_payments=pd.DataFrame(tax_payments),
    )


def _lot_method(
    cfg: BacktestConfig, tax_engine: TaxEngine | None, ytd_short: float, ytd_long: float
) -> LotMethod:
    """Headroom-aware lot selection.

    With 0% band left, sell the *cheapest* lots: the gain is untaxed and basis
    steps up, so deferring it is strictly worse. Once the band is used, revert to
    minimizing the taxable gain.
    """
    if tax_engine is None:
        return cfg.lot_method
    headroom = tax_engine.headroom(short_gain=ytd_short, long_gain=ytd_long)
    return "lofo" if headroom > 0 else cfg.lot_method


def _orders(
    book: LotBook, target: pd.Series, px: pd.Series, equity: float, cfg: BacktestConfig
) -> dict[int, float]:
    """Share deltas needed to reach the target weights."""
    orders: dict[int, float] = {}
    names = set(target.index) | set(book.holdings())

    for permno in names:
        price = px.get(permno, np.nan)
        if not np.isfinite(price) or price <= 0:
            continue
        want_shares = target.get(permno, 0.0) * equity / price
        delta = want_shares - book.shares(permno)
        if abs(delta * price) >= cfg.min_trade_value:
            orders[permno] = delta
    return orders
