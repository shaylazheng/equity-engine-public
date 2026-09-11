"""Long-only portfolio construction against a structured risk model.

## What this replaces, and why it is not optional

The engine's first construction rule was top-N by score, equal weighted. That is
a defensible baseline and a bad portfolio: it treats a name whose alpha is barely
above the cutoff exactly like the best name in the book, and it is blind to risk
entirely -- twenty-five names that happen to be twenty-five semiconductor
companies is, to that rule, a diversified portfolio. Having fitted a risk model
on real returns, ignoring it at the one step where it would change a decision
would be strange.

## The problem actually solved

    maximize    a'w  -  (lambda/2) w' S w  -  c'|w - w0|
    subject to  sum(w) = 1,   0 <= w <= cap

Every term is there for a reason that shows up in the trade list:

- ``a'w`` -- expected residual alpha. Scores, not raw signal.
- ``w' S w`` -- risk, using the **structured** covariance ``S = X F X' + D``
  rather than a sample matrix. At ~4,000 names a sample covariance is 16M
  entries estimated from a few hundred observations; it is not merely slow, it is
  singular and an optimizer will find "arbitrage" in its null space. The factor
  form is the reason a risk model was fitted at all.
- ``c'|w - w0|`` -- linear trading cost against the *current* book ``w0``. Without
  it the optimizer happily rebuilds the portfolio every rebalance to chase noise
  in the third decimal of alpha. This term is what makes it hold things.
- ``w >= 0`` -- long-only, structurally, not by post-hoc clipping.
- ``w <= cap`` -- position limit.
- ``sum(w) = 1`` -- fully invested.

## Why a hand-written solver

No dependency here beyond numpy, which is deliberate rather than stubborn:

1. The covariance is never formed. ``S w = X (F (X' w)) + d * w`` costs O(n k)
   with k ~ 16 factors, against O(n^2) for a dense product. A general QP solver
   takes a matrix, so handing it this problem means building the thing the factor
   model exists to avoid.
2. The constraint set -- a simplex intersected with a box -- has an exact
   projection, and adding the L1 cost term keeps the proximal step exact via a
   one-dimensional bisection on the budget multiplier. Nothing here is a
   heuristic or a penalty approximation.

The method is FISTA (accelerated proximal gradient). The objective is convex, so
the fixed point is the global optimum and there is no local-minimum story to
worry about.

## The thing that will bite

Mean-variance optimization is famous for turning small alpha errors into large
position swings, because it is an error-maximizer by construction: it
overweights whatever the estimate says is simultaneously high-alpha and
low-risk, which is exactly where estimation error hides. Three defences are wired
in and none is optional -- the position cap, the trading-cost term, and
``max_active_share``, which bounds how far the answer may sit from equal weight.
Turning all three off gives a mathematically correct portfolio that no one should
hold.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from qe_core.risk_appetite import RiskProfile

__all__ = [
    "AlphaScaleError",
    "ExposureScaleError",
    "OptimizerConfig",
    "OptimizerResult",
    "RiskInputs",
    "apply_beta_tilt",
    "optimize_weights",
    "scores_to_alpha",
]

#: Cross-sectional standard deviation of `alpha` above which the input is
#: assumed to be scores rather than expected returns. A genuine cross-sectional
#: spread of expected monthly excess returns is a few percent; 0.5 is 50%, which
#: no honest forecast produces.
MAX_PLAUSIBLE_ALPHA_SD = 0.5


class AlphaScaleError(ValueError):
    """Raised when `alpha` is not plausibly in return units.

    This is a hard error rather than a warning because the failure it prevents is
    invisible. Feed z-scores or ranks straight into a mean-variance objective and
    the alpha term is ~1.0 while the risk term is ~1e-4: the optimizer degenerates
    into "sort by score and buy the top one", and the risk model, the trading-cost
    term, and risk aversion all become decorative. Every number it returns still
    looks reasonable. See :func:`scores_to_alpha`.
    """


class ExposureScaleError(ValueError):
    """Raised when a factor exposure is not on the scale a tilt assumes.

    :attr:`OptimizerConfig.beta_premium` is quoted per **standardized** exposure.
    Fed a column of raw betas -- mean ~1.0, SD ~0.35 -- the same number means
    something roughly three times smaller, applied on top of a near-constant
    offset that tilts toward nothing in particular. Nothing about the resulting
    book would look wrong, which is why this is an error rather than a warning.
    """


@dataclass(frozen=True)
class RiskInputs:
    """Structured covariance: ``S = X F X' + diag(specific_var)``.

    Held as its pieces rather than a matrix. Beyond the cost argument, this keeps
    the risk *decomposition* available at the end -- the caller can ask how much
    of the portfolio's variance is factor and how much is specific, which is a
    question about the answer that a dense matrix cannot answer.
    """

    exposures: pd.DataFrame          # names x factors
    factor_cov: pd.DataFrame         # factors x factors
    specific_var: pd.Series          # names

    def __post_init__(self) -> None:
        missing = set(self.exposures.columns) - set(self.factor_cov.columns)
        if missing:
            raise ValueError(f"exposures reference unknown factors: {sorted(missing)}")
        if not self.factor_cov.index.equals(self.factor_cov.columns):
            raise ValueError("factor_cov must be square and symmetrically labelled")

    def align(self, names: pd.Index) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (X, F, d) as arrays for `names`, in that order.

        A name absent from the exposure frame gets zero factor exposure and the
        **median** specific variance rather than zero. Zero would tell the
        optimizer the name is riskless, which is the single most dangerous thing
        an unknown name could be told it is.
        """
        X = self.exposures.reindex(names).to_numpy(dtype=float)
        X = np.nan_to_num(X, nan=0.0)
        F = self.factor_cov.reindex(
            index=self.exposures.columns, columns=self.exposures.columns
        ).to_numpy(dtype=float)
        F = 0.5 * (F + F.T)

        d = self.specific_var.reindex(names).to_numpy(dtype=float)
        fallback = float(np.nanmedian(self.specific_var.to_numpy(dtype=float)))
        if not np.isfinite(fallback) or fallback <= 0:
            fallback = 1e-4
        d = np.where(np.isfinite(d) & (d > 0), d, fallback)
        return X, F, d


@dataclass(frozen=True)
class OptimizerConfig:
    #: Risk aversion. Higher = more diversified, lower = more alpha-chasing.
    #: Scaled for *daily* variance; a monthly risk model needs this divided by 21.
    risk_aversion: float = 10.0
    #: Maximum weight in any one name.
    max_weight: float = 0.10
    #: One-way trading cost as a fraction of notional, used *inside* the
    #: objective. This is the optimizer's own view of cost, deliberately a little
    #: higher than the true spread: it is the dial that buys portfolio stability.
    trade_cost: float = 0.0020
    #: Cap on sum|w - equal_weight| / 2. Bounds how far the optimizer may wander
    #: from an equal-weighted book -- the blunt defence against error
    #: maximization. `None` disables it.
    max_active_share: float | None = 0.60
    #: Information coefficient used to turn scores into expected returns before
    #: optimizing (see `scores_to_alpha`). `None` means alpha is already in
    #: return units. Default 0.03 is deliberately pessimistic: a genuinely good
    #: equity signal runs 0.02-0.05, and anything above 0.10 is usually a
    #: backtest measuring itself.
    score_ic: float | None = 0.03
    #: Names considered. The optimizer runs on the top `universe_size` by alpha;
    #: below that the alpha estimate is not informative enough to be worth an
    #: optimization variable.
    universe_size: int = 200
    #: Price the tax bill a sale would realize into the objective's sell-side
    #: cost, so a position sitting on a short-term gain is expensive to leave and
    #: one sitting on a loss is cheap. Requires a `TaxEngine` on the backtest;
    #: without one there is no bill to price and this has no effect.
    tax_aware: bool = True
    #: Expected excess return, **per rebalance period**, for a portfolio carrying
    #: +1 standard deviation of `beta_exposure_column`. Added to alpha before
    #: optimizing, so the optimizer prices the tilt against the variance it adds
    #: rather than being forced into it.
    #:
    #: This is the only dial here that raises *compensated* risk: everything else
    #: in this config moves how far the book sits from equal weight, which is a
    #: bet on the alpha being real. Zero means no view on beta, which is what
    #: every result in `docs/` was produced at. Set it from a
    #: `qe_core.RiskProfile` via :meth:`from_profile`, which does the annual ->
    #: per-period conversion in one place.
    beta_premium: float = 0.0
    #: Exposure column the tilt is applied to. Must be standardized; see
    #: :class:`ExposureScaleError`.
    beta_exposure_column: str = "beta"
    max_iter: int = 500
    tol: float = 1e-9

    @classmethod
    def from_profile(
        cls, profile: RiskProfile, *, periods_per_year: int, **overrides
    ) -> OptimizerConfig:
        """Build a config from a :class:`qe_core.RiskProfile`.

        `periods_per_year` is required because the profile quotes its beta
        premium annually while the optimizer needs it in the alpha's own units,
        and a silently defaulted 12 would be wrong by a factor of 21 the first
        time this runs on a daily panel. `overrides` are applied last -- the live
        path uses that to pass `score_ic=None`, having already converted scores
        to returns itself.
        """
        cfg = cls(
            risk_aversion=profile.risk_aversion,
            max_weight=profile.max_weight,
            trade_cost=profile.trade_cost,
            max_active_share=profile.max_active_share,
            score_ic=profile.score_ic,
            universe_size=profile.universe_size,
            beta_premium=profile.beta_premium_per_period(periods_per_year),
        )
        return dataclasses.replace(cfg, **overrides) if overrides else cfg


@dataclass
class OptimizerResult:
    weights: pd.Series
    iterations: int
    converged: bool
    objective: float
    factor_var: float
    specific_var: float
    turnover: float

    @property
    def total_var(self) -> float:
        return self.factor_var + self.specific_var

    def risk_decomposition(self) -> pd.Series:
        """Factor vs specific share of predicted variance.

        Reported rather than merely computed because it is the fastest read on
        whether a portfolio is doing what was intended: a book that is 95%
        factor risk is a sector bet wearing a stock-picking costume.
        """
        total = self.total_var
        if total <= 0:
            return pd.Series({"factor": np.nan, "specific": np.nan})
        return pd.Series(
            {"factor": self.factor_var / total, "specific": self.specific_var / total}
        )

    def __repr__(self) -> str:
        n = int((self.weights > 1e-6).sum())
        return (
            f"<OptimizerResult {n} names, vol {np.sqrt(max(self.total_var, 0)):.2%}, "
            f"{self.factor_var / self.total_var:.0%} factor, "
            f"turnover {self.turnover:.1%}, "
            f"{'converged' if self.converged else 'NOT converged'} "
            f"in {self.iterations} iters>"
        )


def scores_to_alpha(
    scores: pd.Series, *, ic: float, specific_var: pd.Series | float
) -> pd.Series:
    """Convert cross-sectional scores into expected returns (Grinold-Kahn).

        alpha = IC * volatility * z-score

    The "fundamental law" refactoring of a forecast: a standardized score says
    *where* a name ranks, and it takes an information coefficient and a
    volatility to say what that rank is worth in return. Skipping this step is
    the single easiest way to make a risk model decorative -- see
    :class:`AlphaScaleError`.

    `ic` is the correlation between the score and realized residual return. Be
    pessimistic: a genuinely good equity signal runs 0.02-0.05, and values above
    0.10 are usually a backtest measuring itself.

    `specific_var` is a **variance**, matching `RiskInputs.specific_var`, and is
    square-rooted here. Named for the units rather than for the concept because
    "risk" is ambiguous between the two and mixing them is a factor-of-100 error
    that still produces a plausible-looking number.
    """
    z = scores.astype(float)
    sd = z.std(ddof=1)
    z = (z - z.mean()) / sd if sd > 0 else z * 0.0
    var = (
        pd.Series(float(specific_var), index=scores.index)
        if np.isscalar(specific_var)
        else pd.Series(specific_var).reindex(scores.index).astype(float)
    )
    return (float(ic) * np.sqrt(var.clip(lower=0.0)) * z).rename("alpha")


def apply_beta_tilt(
    alpha: pd.Series,
    exposures: pd.DataFrame,
    *,
    premium: float,
    column: str = "beta",
) -> pd.Series:
    """Add a deliberate market-risk tilt to alpha.

        alpha' = alpha + premium * exposure[column]

    This is how "I want more market risk" enters a long-only book. It cannot
    enter through `risk_aversion`, which is symmetric: lowering it concentrates
    the book into whatever the alpha likes, and the alpha has been *neutralized*
    against beta precisely so it does not like high-beta names. Left alone, the
    engine builds a roughly beta-neutral book at every level of aggression.

    Stating a premium instead says the thing that is actually meant -- that beta
    is compensated -- and lets the optimizer trade it against the variance it
    adds. A premium too small to overcome the risk term simply does not move the
    book, which is the correct behaviour and not a silent failure.

    `premium` is **per period**, matching alpha's units, and per **standardized**
    exposure. Both are enforced rather than documented: an absent column raises,
    and an unstandardized one raises too.
    """
    if premium == 0.0:
        return alpha
    if column not in exposures.columns:
        raise ValueError(
            f"beta tilt asked for exposure {column!r}, which is not among "
            f"{sorted(exposures.columns)}. Refusing rather than tilting by zero: "
            "a tilt that silently did nothing would leave the book beta-neutral "
            "while every report said it was not."
        )
    col = exposures[column].astype(float)
    finite = col[np.isfinite(col)]
    if len(finite) > 1:
        mean, sd = float(finite.mean()), float(finite.std(ddof=1))
        if abs(mean) > 0.5 or not 0.5 < sd < 2.0:
            raise ExposureScaleError(
                f"exposure {column!r} has mean {mean:.3g} and SD {sd:.3g}; a "
                "standardized column is ~(0, 1). `beta_premium` is quoted per "
                "standard deviation, so on this column it would mean something "
                "else entirely. Standardize the exposure (qe_risk.neutralize."
                "standardize) or quote the premium against the column you have."
            )
    # A name with no exposure estimate gets no tilt, rather than the column mean:
    # not knowing a name's beta is not a reason to bet on it.
    tilt = col.reindex(alpha.index).astype(float).fillna(0.0)
    return (alpha + float(premium) * tilt).rename(alpha.name)


def _prox(
    z: np.ndarray,
    w0: np.ndarray,
    buy_cost: np.ndarray,
    sell_cost: np.ndarray,
    cap: np.ndarray,
) -> np.ndarray:
    """Exact proximal step: nearest point to `z` under cost, box, and budget.

    Solves

        argmin_w  0.5||w - z||^2
                  + sum_i buy_cost_i (w_i - w0_i)_+
                  + sum_i sell_cost_i (w0_i - w_i)_+
        s.t.      sum(w) = 1,  0 <= w_i <= cap_i

    Buying and selling are priced **separately** because for a taxable account
    they genuinely differ: buying costs a spread, while selling costs a spread
    *and* a tax bill on whatever gain it realizes. A symmetric cost cannot say
    "this position is cheap to add to and expensive to leave", which is the whole
    of holding-period-aware rebalancing.

    `cap` is per name rather than scalar because "you may keep this position but
    may not add to it" is a real constraint -- it is what a wash-sale block
    means -- and it is exactly a per-name upper bound at the current weight.

    The Lagrangian on the budget constraint separates, so each coordinate is a
    soft-threshold toward its current holding followed by a clip:

        w_i(v) = clip( shrink(z_i + v, w0_i, step_cost_i), 0, cap )

    Every ``w_i(v)`` is continuous and non-decreasing in ``v``, so ``sum_i w_i(v)``
    is too and the budget has a unique root -- found by bisection, which is exact
    to machine precision rather than approximately satisfied by a penalty.
    """
    n = z.size

    def at(v: float) -> np.ndarray:
        y = z + v
        # Two-sided soft threshold toward the current holding, with a different
        # width on each side. Inside the band, stay put.
        d = y - w0
        w = w0 + np.where(
            d > buy_cost,
            d - buy_cost,
            np.where(d < -sell_cost, d + sell_cost, 0.0),
        )
        return np.clip(w, 0.0, cap)

    if cap.sum() < 1.0 - 1e-12:
        raise ValueError(
            f"infeasible: {n} names cap out at {cap.sum():.3f} and cannot sum to 1. "
            "Raise max_weight or widen the universe."
        )

    lo, hi = -1.0, 1.0
    while at(lo).sum() > 1.0:
        lo *= 2.0
        if lo < -1e9:  # pragma: no cover - unreachable for a feasible problem
            break
    while at(hi).sum() < 1.0:
        hi *= 2.0
        if hi > 1e9:  # pragma: no cover
            break

    # Bisect to machine precision, then stop. A fixed 200 iterations asks for
    # 2^-200, which is ~1e-60: about 140 iterations past anything a float can
    # represent, on the inner loop of the inner loop. The early exit is what
    # makes a 600-rebalance backtest finish.
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if at(mid).sum() < 1.0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-14 * max(1.0, abs(hi)):
            break
    return at(0.5 * (lo + hi))


def _lipschitz(X: np.ndarray, F: np.ndarray, d: np.ndarray, lam: float) -> float:
    """Largest eigenvalue of lambda*S, by power iteration on the factor form."""
    rng = np.random.default_rng(0)
    v = rng.standard_normal(X.shape[0])
    v /= np.linalg.norm(v)
    top = 1.0
    for _ in range(60):
        u = X @ (F @ (X.T @ v)) + d * v
        top = float(np.linalg.norm(u))
        if top <= 0:
            return 1.0
        v = u / top
    return max(lam * top, 1e-12)


def optimize_weights(
    alpha: pd.Series,
    risk: RiskInputs,
    *,
    current: pd.Series | None = None,
    max_weights: pd.Series | None = None,
    sell_cost: pd.Series | None = None,
    config: OptimizerConfig | None = None,
) -> OptimizerResult:
    """Solve the long-only mean-variance problem for one rebalance date.

    Parameters
    ----------
    alpha:
        Expected residual return per name. Scores after neutralization, not raw
        signal -- feeding raw signal here re-introduces exactly the factor bets
        the risk model was fitted to remove.
    current:
        The book being traded from. Omitted means starting from cash, in which
        case the cost term charges for the whole build.
    max_weights:
        Per-name upper bounds overriding `config.max_weight`. The use that
        matters is a no-buy constraint -- setting a name's bound to its current
        weight lets the optimizer hold or sell it but never add, which is what a
        wash-sale block actually requires. Dropping such a name from the universe
        instead would force a sale that was never priced against its cost.
    sell_cost:
        Per-name cost of *selling*, as a fraction of position value, replacing
        `config.trade_cost` on the sell side. This is where the tax bill enters
        portfolio construction: a position sitting on a short-term gain is
        expensive to leave and a position sitting on a loss is cheap — or better
        than cheap, since realizing the loss offsets other gains. Negative values
        are allowed and meaningful for exactly that reason.
    """
    cfg = config or OptimizerConfig()

    a = alpha.replace([np.inf, -np.inf], np.nan).dropna()
    if len(a) > 1:
        sd = float(a.std(ddof=1))
        if np.isfinite(sd) and sd > MAX_PLAUSIBLE_ALPHA_SD:
            raise AlphaScaleError(
                f"alpha has a cross-sectional SD of {sd:.3g}, which as an expected "
                f"return is {sd:.0%}. This is almost certainly scores rather than "
                "returns; at that scale the risk and trading-cost terms are "
                "numerically irrelevant and the optimizer reduces to sorting by "
                "score. Convert with scores_to_alpha(scores, ic=..., "
                "specific_risk=...) first."
            )
    if a.empty:
        return OptimizerResult(
            pd.Series(dtype=float), 0, True, 0.0, 0.0, 0.0, 0.0
        )

    # Before the universe is chosen, not after: if beta is believed to be
    # compensated then it is part of expected return, and a name should be able
    # to earn its way into the top slice on that basis like any other.
    a = apply_beta_tilt(
        a, risk.exposures,
        premium=cfg.beta_premium, column=cfg.beta_exposure_column,
    )

    held = pd.Series(dtype=float) if current is None else current[current > 0]
    # Held names stay in the universe even if their alpha has decayed out of the
    # top slice. Dropping them would force a sale the optimizer never got to
    # price against its own cost term -- a forced trade is not an optimized one.
    keep = a.nlargest(min(cfg.universe_size, len(a))).index.union(
        held.index.intersection(a.index)
    )
    a = a.reindex(keep)
    names = a.index

    n = len(names)
    if n == 0:
        return OptimizerResult(pd.Series(dtype=float), 0, True, 0.0, 0.0, 0.0, 0.0)

    cap = np.full(n, float(cfg.max_weight))
    if max_weights is not None:
        override = max_weights.reindex(names).to_numpy(dtype=float)
        cap = np.where(np.isfinite(override), np.minimum(cap, override), cap)
    cap = np.maximum(cap, 0.0)
    if cap.sum() < 1.0:
        # An unfunded book is a silent, expensive wrong answer, so relax the
        # binding rather than return one -- but relax the *uniform* cap only,
        # never a per-name override, which was asked for on purpose.
        room = np.full(n, float(cfg.max_weight))
        if max_weights is not None:
            fixed = np.isfinite(max_weights.reindex(names).to_numpy(dtype=float))
        else:
            fixed = np.zeros(n, dtype=bool)
        free = ~fixed
        if free.any():
            deficit = 1.0 - cap[fixed].sum()
            room[free] = max(deficit / free.sum(), float(cfg.max_weight))
            cap = np.where(free, room, cap)
        if cap.sum() < 1.0:
            cap = np.full(n, 1.0 / n)

    X, F, d = risk.align(names)
    lam = float(cfg.risk_aversion)
    avec = a.to_numpy(dtype=float)
    w0 = (
        np.zeros(n)
        if current is None
        else np.nan_to_num(current.reindex(names).to_numpy(dtype=float))
    )

    def sigma_dot(w: np.ndarray) -> np.ndarray:
        return X @ (F @ (X.T @ w)) + d * w

    def smooth(w: np.ndarray) -> float:
        return float(-avec @ w + 0.5 * lam * (w @ sigma_dot(w)))

    def grad(w: np.ndarray) -> np.ndarray:
        return -avec + lam * sigma_dot(w)

    buy_vec = np.full(n, float(cfg.trade_cost))
    sell_vec = np.full(n, float(cfg.trade_cost))
    if sell_cost is not None:
        override = sell_cost.reindex(names).to_numpy(dtype=float)
        sell_vec = np.where(np.isfinite(override), override, sell_vec)
    # A negative *total* threshold would invert the soft-threshold and push the
    # solution away from w0 without bound, so a tax benefit is capped at giving
    # the sale away for free rather than paying to make it.
    sell_vec = np.maximum(sell_vec, 0.0)
    step = 1.0 / _lipschitz(X, F, d, lam)

    zero = np.zeros(n)
    w = np.full(n, 1.0 / n) if current is None else _prox(w0, w0, zero, zero, cap)
    y, t = w.copy(), 1.0
    converged = False
    it = 0

    for it in range(1, cfg.max_iter + 1):
        w_new = _prox(
            y - step * grad(y), w0, step * buy_vec, step * sell_vec, cap
        )
        if np.max(np.abs(w_new - w)) < cfg.tol:
            w = w_new
            converged = True
            break
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = w_new + ((t - 1.0) / t_new) * (w_new - w)
        w, t = w_new, t_new

    if cfg.max_active_share is not None:
        w = _limit_active_share(w, cap, cfg.max_active_share)

    fx = X.T @ w
    factor_var = float(fx @ F @ fx)
    specific_var = float(np.sum(d * w * w))
    delta = w - w0
    obj = float(
        -smooth(w)
        - float(buy_vec @ np.maximum(delta, 0.0))
        - float(sell_vec @ np.maximum(-delta, 0.0))
    )

    return OptimizerResult(
        weights=pd.Series(w, index=names).where(lambda s: s > 1e-8).dropna(),
        iterations=it,
        converged=converged,
        objective=obj,
        factor_var=factor_var,
        specific_var=specific_var,
        turnover=float(np.abs(w - w0).sum() / 2.0),
    )


def _limit_active_share(w: np.ndarray, cap: np.ndarray, limit: float) -> np.ndarray:
    """Pull `w` toward equal weight until active share is within `limit`.

    Active share is measured against equal weight over the optimized universe,
    not against a market benchmark: the question this bounds is "how concentrated
    is this relative to naive diversification", and equal weight is what naive
    means here.

    Blending toward equal weight is a linear path, and active share is a norm
    along it, so the required blend has a closed form -- no search, and the
    resulting weights still satisfy the budget and (since equal weight is inside
    the box) the box.
    """
    n = w.size
    eq = np.full(n, 1.0 / n)
    if (eq > cap + 1e-12).any():
        # Equal weight is outside the box, so the blend target is not feasible
        # and blending would quietly breach a cap. The cap is the harder
        # constraint of the two -- a no-buy block is a rule, active share is a
        # preference -- so the preference yields.
        return w
    active = float(np.abs(w - eq).sum() / 2.0)
    if active <= limit or active <= 0:
        return w
    theta = limit / active
    return theta * w + (1.0 - theta) * eq
