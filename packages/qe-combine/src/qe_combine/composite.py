"""Linear signal combination that emits its own decomposition.

Method A of the Phase 7 horse race, and — following the decision that exact
additivity is a hard requirement — the default. The reason is structural rather
than aesthetic: a linear composite's score *is* a sum of per-signal
contributions, so the attribution ledger is not a reconstruction of the model,
it is the model. `total == sum(contribution)` holds to floating point, always,
with nothing to verify after the fact.

A gradient-boosted alternative can only offer SHAP values, which are estimates,
sum to the prediction minus a baseline rather than to the prediction, cost real
compute across a 150M-row panel, and move when the model is retrained. That is
the trade the pre-registered margin has to be worth.

Weights shrink toward equal weight, which is a famously hard benchmark to beat —
estimated weights carry estimation error, and on noisy financial data that error
routinely swamps whatever the estimation was supposed to buy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from qe_core.ledger import AttributionLedger

__all__ = ["combine", "orthogonalize_signals", "shrink_weights"]


def shrink_weights(weights: dict[str, float], shrinkage: float = 0.5) -> dict[str, float]:
    """Pull weights toward equal weight.

    `shrinkage=1.0` is pure equal weight; `0.0` trusts the supplied weights
    completely. The default splits the difference, on the view that a weight
    estimated from historical performance is worth about as much as no estimate
    at all.
    """
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError(f"shrinkage must be in [0, 1], got {shrinkage}")
    if not weights:
        raise ValueError("no weights supplied")

    n = len(weights)
    total = sum(abs(w) for w in weights.values())
    if total == 0:
        return dict.fromkeys(weights, 1.0 / n)

    normalized = {k: v / total for k, v in weights.items()}
    return {k: (1.0 - shrinkage) * v + shrinkage / n for k, v in normalized.items()}


def orthogonalize_signals(signals: pd.DataFrame, order: list[str] | None = None) -> pd.DataFrame:
    """Sequentially residualize each signal against those before it.

    Order matters and is a real choice: whatever comes first keeps its full
    variance, and later signals keep only what is new. Put the signals you trust
    most first — they should not be residualized against ones you trust less.
    """
    cols = order or list(signals.columns)
    missing = [c for c in cols if c not in signals.columns]
    if missing:
        raise KeyError(f"orthogonalization order names unknown signals: {missing}")

    out = pd.DataFrame(index=signals.index, columns=cols, dtype=float)
    kept: list[str] = []

    for col in cols:
        y = signals[col]
        if not kept:
            out[col] = y
            kept.append(col)
            continue

        valid = y.dropna().index.intersection(out[kept].dropna().index)
        if len(valid) < len(kept) + 2:
            out[col] = y
            kept.append(col)
            continue

        X = np.column_stack([np.ones(len(valid)), out.loc[valid, kept].to_numpy(dtype=float)])
        beta, *_ = np.linalg.lstsq(X, y.loc[valid].to_numpy(dtype=float), rcond=None)
        resid = pd.Series(y.loc[valid].to_numpy() - X @ beta, index=valid)
        out[col] = resid.reindex(signals.index)
        kept.append(col)

    return out


def combine(
    signals: dict[str, pd.Series],
    metadata: dict[str, dict],
    date,
    *,
    weights: dict[str, float] | None = None,
    shrinkage: float = 0.5,
    orthogonalize: bool = True,
    order: list[str] | None = None,
) -> AttributionLedger:
    """Combine normalized signals into an attribution ledger.

    Returns the ledger, not a score. The score is `ledger.totals()` — deriving it
    is the only way to obtain one, which is what makes it impossible to display a
    number without its constituents.

    Parameters
    ----------
    signals:
        name -> Series indexed by permno, already normalized and neutralized.
    metadata:
        name -> {"family": ..., "tier": ...}. Non-scoring tiers are rejected by
        the ledger itself, so a flag reaching here fails loudly.
    """
    if not signals:
        raise ValueError("no signals to combine")

    date = pd.Timestamp(date)
    frame = pd.DataFrame(signals)

    missing_meta = [n for n in frame.columns if n not in metadata]
    if missing_meta:
        raise KeyError(f"no metadata for signal(s) {missing_meta}")

    if orthogonalize and frame.shape[1] > 1:
        frame = orthogonalize_signals(frame, order=order)

    w = shrink_weights(weights or dict.fromkeys(frame.columns, 1.0), shrinkage)

    rows = []
    for name in frame.columns:
        z = frame[name]
        valid = z.dropna()
        if valid.empty:
            continue
        meta = metadata[name]
        rows.append(
            pd.DataFrame(
                {
                    "permno": valid.index.astype("int64"),
                    "date": date,
                    "signal": name,
                    "family": meta["family"],
                    "tier": meta["tier"],
                    "weight": w[name],
                    "z_value": valid.to_numpy(),
                    "contribution": w[name] * valid.to_numpy(),
                }
            )
        )

    if not rows:
        raise ValueError(f"every signal was empty at {date.date()}")

    ledger = pd.concat(rows, ignore_index=True)
    ledger["signal"] = ledger["signal"].astype("string")
    ledger["family"] = ledger["family"].astype("string")
    ledger["tier"] = ledger["tier"].astype("string")
    return AttributionLedger(ledger)
