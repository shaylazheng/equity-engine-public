"""Synthetic daily panel with *known* factor structure.

The architecture here — a latent per-firm health state driving fundamentals,
delisting hazard, and returns coherently — is taken from
`pi-model-replication/src/pi_model/data/synthetic.py`, which had the right idea.
Two things are changed, both load-bearing:

1. **Returns carry a common factor structure.** In the original, returns are
   cross-sectionally independent given health, so the covariance matrix it
   produces is spuriously near-diagonal — any beta, covariance, or
   neutralization logic validated against it passes for the wrong reason. Here
   the loadings are *planted and returned*, so the risk model has a ground truth
   to recover rather than merely something to run on.

2. **It is daily and vectorized.** The original is monthly and builds returns in
   a Python double loop appending dicts — roughly 30M appends at daily scale.

Fundamentals are emitted with a realistic reporting lag, so the panel exercises
the point-in-time contract rather than merely satisfying its schema: a signal
that reads `period_end` instead of `knowledge_date` sees ~75 days of the future.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .calendar import trading_days
from .panel import AsOfPanel

__all__ = ["FACTORS", "SyntheticPanel", "generate"]

#: Planted factors. Deliberately small — enough for a risk model to recover, not
#: a pretend replica of a commercial model.
FACTORS: tuple[str, ...] = ("market", "size", "value")

_TRADING_DAYS = 252

# Annualized factor means and vols, then scaled to daily.
_FACTOR_MU = {"market": 0.08, "size": 0.02, "value": 0.03}
_FACTOR_VOL = {"market": 0.16, "size": 0.10, "value": 0.10}

#: Fundamentals report ~75 calendar days after fiscal year end. This gap is the
#: whole point: using `period_end` as the knowable date leaks a quarter of future.
_REPORT_LAG_DAYS = 75

_SOURCE = "synthetic"
_URL = "synthetic://qe-core/generate"


@dataclass(frozen=True)
class SyntheticPanel:
    """A generated panel plus the ground truth used to build it."""

    panel: AsOfPanel
    #: permno x factor. What the risk model must recover.
    betas: pd.DataFrame
    #: date x factor.
    factor_returns: pd.DataFrame
    #: Latent per-firm quality in [0, 1].
    health: pd.Series
    #: Annualized idiosyncratic vol per firm.
    specific_vol: pd.Series
    #: permno -> first date the firm is absent (NaT if it survives).
    delist_date: pd.Series

    def __repr__(self) -> str:
        return (
            f"<SyntheticPanel {len(self.panel):,} facts, {len(self.betas):,} firms, "
            f"{len(self.factor_returns):,} sessions>"
        )


def generate(
    *,
    n_firms: int = 200,
    start: str = "2015-01-02",
    end: str = "2017-12-29",
    seed: int = 11,
    distress_rate: float = 0.15,
    delist_rate: float = 0.0,
) -> SyntheticPanel:
    """Build a synthetic PIT panel.

    Parameters
    ----------
    delist_rate:
        Annual hazard of a firm disappearing, making the panel unbalanced as a
        real one is. Defaults to 0 so beta-recovery tests get a clean balanced
        panel; turn it up to exercise ragged handling.
    """
    rng = np.random.default_rng(seed)
    days = trading_days(start, end)
    n_days = len(days)
    if n_days == 0:
        raise ValueError(f"no trading sessions between {start} and {end}")

    permnos = np.arange(10_001, 10_001 + n_firms, dtype=np.int64)

    # -- latent state --------------------------------------------------------
    health = rng.beta(2.5, 1.5, size=n_firms)
    distressed = rng.random(n_firms) < distress_rate
    health = np.where(distressed, health * rng.uniform(0.15, 0.45, size=n_firms), health)

    # -- planted loadings ----------------------------------------------------
    betas = np.column_stack(
        [
            np.clip(rng.normal(1.0, 0.35, n_firms), 0.20, 2.50),  # market
            rng.normal(0.0, 0.60, n_firms),  # size
            rng.normal(0.0, 0.60, n_firms),  # value
        ]
    )

    # -- factor returns ------------------------------------------------------
    f_mu = np.array([_FACTOR_MU[f] for f in FACTORS]) / _TRADING_DAYS
    f_sd = np.array([_FACTOR_VOL[f] for f in FACTORS]) / np.sqrt(_TRADING_DAYS)
    factor_rets = rng.normal(f_mu, f_sd, size=(n_days, len(FACTORS)))

    # -- idiosyncratic returns ----------------------------------------------
    # Student-t so the tails are real; rescaled so the sample vol still matches
    # the target (t with df=5 has variance df/(df-2) = 5/3).
    spec_vol_ann = 0.15 + 0.35 * (1.0 - health)
    spec_vol_d = spec_vol_ann / np.sqrt(_TRADING_DAYS)
    df = 5
    eps = rng.standard_t(df, size=(n_days, n_firms)) / np.sqrt(df / (df - 2))
    eps *= spec_vol_d

    # (T,K) @ (K,N) -> (T,N). This is what makes the covariance non-diagonal.
    returns = factor_rets @ betas.T + eps

    # -- prices and size -----------------------------------------------------
    prc0 = np.exp(rng.normal(np.log(30.0), 0.8, n_firms))
    prices = prc0 * np.cumprod(1.0 + returns, axis=0)
    shares_m = np.exp(rng.normal(np.log(50.0), 1.0, n_firms))  # millions
    mktcap = prices * shares_m

    # -- delisting -----------------------------------------------------------
    alive = np.ones((n_days, n_firms), dtype=bool)
    delist_idx = np.full(n_firms, -1, dtype=np.int64)
    if delist_rate > 0:
        daily_hazard = (delist_rate / _TRADING_DAYS) * (0.4 + 1.8 * (1.0 - health))
        draws = rng.random((n_days, n_firms)) < daily_hazard
        first = np.argmax(draws, axis=0)
        hit = draws.any(axis=0)
        delist_idx = np.where(hit, first, -1)
        for j in np.flatnonzero(hit):
            alive[delist_idx[j] :, j] = False

    delist_date = pd.Series(
        [days[i] if i >= 0 else pd.NaT for i in delist_idx], index=permnos, name="delist_date"
    )

    # -- assemble price-side facts ------------------------------------------
    frames = [
        _long(values, days, permnos, alive, concept)
        for values, concept in (
            (returns, "ret"),
            (prices, "prc"),
            (mktcap, "mktcap"),
        )
    ]

    # Prices are knowable the same session, post-close: date == reported == known.
    price_facts = pd.concat(frames, ignore_index=True)
    price_facts["period_start"] = pd.NaT
    price_facts["reported_at"] = price_facts["period_end"]
    price_facts["knowledge_date"] = price_facts["period_end"]

    # -- fundamentals, with a reporting lag ---------------------------------
    fundamentals = _fundamentals(days, permnos, health, mktcap, alive, rng)

    facts = pd.concat([price_facts, fundamentals], ignore_index=True)
    facts["raw_concept"] = facts["concept"]
    facts["source"] = _SOURCE
    facts["source_url"] = _URL
    facts = facts.astype(
        {"permno": "int64", "concept": "string", "value": "float64",
         "raw_concept": "string", "source": "string", "source_url": "string"}
    )

    return SyntheticPanel(
        panel=AsOfPanel(facts),
        betas=pd.DataFrame(betas, index=permnos, columns=list(FACTORS)),
        factor_returns=pd.DataFrame(factor_rets, index=days, columns=list(FACTORS)),
        health=pd.Series(health, index=permnos, name="health"),
        specific_vol=pd.Series(spec_vol_ann, index=permnos, name="specific_vol"),
        delist_date=delist_date,
    )


def _long(
    values: np.ndarray,
    days: pd.DatetimeIndex,
    permnos: np.ndarray,
    alive: np.ndarray,
    concept: str,
) -> pd.DataFrame:
    """Melt a (T, N) array into long form, dropping post-delisting cells.

    The `dropna` is load-bearing and drops exactly the not-alive cells: generated
    values are finite by construction, so the only NaN present are the ones
    `where(alive)` just introduced. (pandas 3.0's `stack` retains NaN rather than
    dropping them, which would otherwise push nulls into the panel.)
    """
    wide = pd.DataFrame(values, index=days, columns=permnos).where(alive)
    out = wide.stack().dropna().reset_index()
    out.columns = ["period_end", "permno", "value"]
    out["concept"] = concept
    return out


def _fundamentals(
    days: pd.DatetimeIndex,
    permnos: np.ndarray,
    health: np.ndarray,
    mktcap: np.ndarray,
    alive: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Annual fundamentals, internally consistent, reported ~75 days after year end."""
    years = sorted({d.year for d in days})
    rows = []

    # Balance-sheet composition is a persistent firm characteristic, not an
    # annual coin flip. Drawing these fresh each year would make every
    # year-over-year delta pure redraw noise — which silently breaks *every*
    # delta-based signal (accruals, asset growth, net share issuance), not just
    # this one. Draw the ratios once per firm, then drift them slowly.
    n_all = len(permnos)
    prior_wc: dict[int, float] = {}
    ratio = {
        "act": np.clip(rng.normal(0.42 + 0.10 * health, 0.10), 0.05, 0.95),
        "che": np.clip(rng.normal(0.25 + 0.15 * health, 0.08), 0.01, 0.90),
        "lct": np.clip(rng.normal(0.28 - 0.05 * health, 0.08), 0.02, 0.90),
        "dlc": np.clip(rng.normal(0.20, 0.08, n_all), 0.0, 0.80),
        "txp": np.clip(rng.normal(0.05, 0.02, n_all), 0.0, 0.40),
    }

    for year in years:
        fye = pd.Timestamp(year=year, month=12, day=31)
        rdq = fye + pd.Timedelta(days=_REPORT_LAG_DAYS)

        # Only firms still alive at year end report.
        mask_day = days <= fye
        if not mask_day.any():
            continue
        last_idx = int(np.flatnonzero(mask_day)[-1])
        live = alive[last_idx]
        if not live.any():
            continue

        n = int(live.sum())
        h = health[live]

        at = mktcap[last_idx][live] * np.exp(rng.normal(0.1, 0.5, n))
        roa = rng.normal(0.10 * h - 0.04, 0.06)
        ni = roa * at
        lev = np.clip(rng.normal(0.85 - 0.45 * h, 0.12), 0.05, 1.60)
        lt = lev * at
        ceq = at - lt
        revt = at * np.clip(rng.normal(0.6 + 0.5 * h, 0.25), 0.05, None)
        dp = 0.05 * at

        # Accrual wedge: weak firms report earnings that cash flow does not back.
        accrual_wedge = at * rng.normal(0.06 * (1.0 - h) - 0.01, 0.03)

        # Balance-sheet accrual components, so the *default* (Sloan's own)
        # construction is exercisable offline. Ratios persist per firm with a
        # small drift, so the year-over-year delta reflects the accrual wedge
        # rather than a fresh draw.
        def drift(k: str, _live: np.ndarray = live, _n: int = n) -> np.ndarray:
            """Persistent per-firm ratio with a small year-to-year wobble.

            Loop variables are bound as defaults rather than captured: a bare
            closure would resolve them at call time and use the final year's
            mask for every year.
            """
            return ratio[k][_live] * np.exp(rng.normal(0.0, 0.03, _n))

        che = at * drift("act") * drift("che")
        lct = at * drift("lct")
        dlc = lct * drift("dlc")
        txp = lct * drift("txp")

        # --- working capital ACCUMULATES the accrual wedge ------------------
        # The wedge is a *flow*, not a level: a firm booking revenue it has not
        # collected grows receivables year after year. So WC_t = WC_{t-1} + wedge,
        # which makes dWC ~ wedge — the quantity both accrual constructions are
        # trying to measure. Adding the wedge to a redrawn level instead would
        # make dWC the *change* in the wedge, which for an i.i.d. wedge is noise
        # uncorrelated with anything, and no accrual signal would work at all.
        prev_wc = np.array([prior_wc.get(int(pn), np.nan) for pn in permnos[live]])
        base_wc = at * (drift("act") - drift("act") * drift("che")) - (lct - dlc - txp)
        wc = np.where(np.isfinite(prev_wc), prev_wc, base_wc) + accrual_wedge
        d_wc = wc - prev_wc

        # Invert wc = (act - che) - (lct - dlc - txp) to recover current assets,
        # so the balance sheet and the cash-flow statement stay consistent.
        act = wc + che + (lct - dlc - txp)

        # CFO = NI + Dep - dWC, the accounting identity. Deriving it from the
        # *actual* change in working capital is what makes the two accrual
        # constructions measure the same thing, as they do in reality.
        oancf = ni + dp - np.where(np.isfinite(d_wc), d_wc, 0.0)
        for pn, w in zip(permnos[live], wc, strict=True):
            prior_wc[int(pn)] = w

        for concept, values in (
            ("at", at), ("ni", ni), ("ceq", ceq), ("revt", revt),
            ("dp", dp), ("oancf", oancf), ("lt", lt),
            ("act", act), ("che", che), ("lct", lct), ("dlc", dlc), ("txp", txp),
        ):
            rows.append(
                pd.DataFrame(
                    {
                        "permno": permnos[live],
                        "concept": concept,
                        "value": values,
                        "period_start": pd.Timestamp(year=year, month=1, day=1),
                        "period_end": fye,
                        "reported_at": rdq,
                        "knowledge_date": rdq,
                    }
                )
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "permno", "concept", "value", "period_start",
                "period_end", "reported_at", "knowledge_date",
            ]
        )
    return pd.concat(rows, ignore_index=True)
