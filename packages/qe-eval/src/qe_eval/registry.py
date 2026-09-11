"""The trial registry.

Deflated Sharpe needs to know how many things you tried. Almost nobody tracks it
— the inherited implementation read a hand-written integer out of a per-paper
YAML, falling back to a hardcoded 100 — and without a real count DSR is theatre:
you supply a made-up number and get a made-up answer.

This records every backtest run, so the count is *observed*. It also stores each
trial's Sharpe, which matters more than the count: Bailey & López de Prado's
expected-maximum-Sharpe term wants the variance of Sharpes across trials, and
that is only available if you kept them.

SQLite via the standard library — a registry that is a single file, survives
crashes, and needs no service.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd

__all__ = ["TrialRegistry", "config_hash"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trials (
    trial_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at  TEXT    NOT NULL,
    scope        TEXT    NOT NULL,          -- e.g. 'core', 'exploratory', a phase name
    config_hash  TEXT    NOT NULL,
    signals      TEXT    NOT NULL,          -- JSON array
    universe     TEXT    NOT NULL,
    start_date   TEXT    NOT NULL,
    end_date     TEXT    NOT NULL,
    sr_period    REAL    NOT NULL,          -- per-period, NOT annualized
    n_obs        INTEGER NOT NULL,
    note         TEXT
);
CREATE INDEX IF NOT EXISTS idx_trials_scope ON trials (scope);
CREATE UNIQUE INDEX IF NOT EXISTS uq_trials_config ON trials (config_hash, scope);
"""


def config_hash(payload: dict) -> str:
    """Stable hash of a run configuration. Key-order independent."""
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


class TrialRegistry:
    """Append-mostly log of every backtest run."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(self.path)
        self._con.row_factory = sqlite3.Row
        self._con.executescript(_SCHEMA)
        self._con.commit()

    # -- writing -----------------------------------------------------------

    def record(
        self,
        *,
        signals: Sequence[str],
        universe: str,
        start: str,
        end: str,
        sr_period: float,
        n_obs: int,
        scope: str = "default",
        config: dict | None = None,
        note: str = "",
    ) -> int:
        """Record one trial. Re-recording the same config within a scope is a no-op.

        Idempotence is deliberate: re-running an identical backtest is not a new
        trial, and counting it as one would inflate the deflation and make results
        look worse than they are. Changing *anything* in the config is a new trial.
        """
        payload = config if config is not None else {
            "signals": sorted(signals), "universe": universe, "start": start, "end": end
        }
        h = config_hash(payload)

        cur = self._con.execute(
            "SELECT trial_id FROM trials WHERE config_hash = ? AND scope = ?", (h, scope)
        )
        existing = cur.fetchone()
        if existing is not None:
            return int(existing["trial_id"])

        cur = self._con.execute(
            "INSERT INTO trials (recorded_at, scope, config_hash, signals, universe, "
            "start_date, end_date, sr_period, n_obs, note) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                datetime.now(UTC).isoformat(timespec="seconds"),
                scope,
                h,
                json.dumps(sorted(signals)),
                universe,
                str(start),
                str(end),
                float(sr_period),
                int(n_obs),
                note,
            ),
        )
        self._con.commit()
        return int(cur.lastrowid)

    # -- reading -----------------------------------------------------------

    def n_trials(self, scope: str | None = None) -> int:
        """The count Deflated Sharpe should be charged."""
        if scope is None:
            row = self._con.execute("SELECT COUNT(*) AS n FROM trials").fetchone()
        else:
            row = self._con.execute(
                "SELECT COUNT(*) AS n FROM trials WHERE scope = ?", (scope,)
            ).fetchone()
        return int(row["n"])

    def sharpe_variance(self, scope: str | None = None) -> float | None:
        """Variance of per-period Sharpes across trials.

        This is what :func:`qe_eval.stats.deflated_sharpe` wants for
        ``var_sr_trials``. Returns None below two trials, where the variance is
        undefined — the caller should then say so rather than substitute a
        placeholder silently.
        """
        sr = self.sharpes(scope)
        return float(np.var(sr, ddof=1)) if len(sr) >= 2 else None

    def sharpes(self, scope: str | None = None) -> np.ndarray:
        sql = "SELECT sr_period FROM trials"
        args: tuple = ()
        if scope is not None:
            sql += " WHERE scope = ?"
            args = (scope,)
        return np.array([r["sr_period"] for r in self._con.execute(sql, args)], dtype=float)

    def trials(self, scope: str | None = None) -> pd.DataFrame:
        sql = "SELECT * FROM trials"
        args: tuple = ()
        if scope is not None:
            sql += " WHERE scope = ?"
            args = (scope,)
        sql += " ORDER BY trial_id"
        df = pd.DataFrame([dict(r) for r in self._con.execute(sql, args)])
        if not df.empty:
            df["signals"] = df["signals"].map(json.loads)
        return df

    def scopes(self) -> list[str]:
        return [r["scope"] for r in self._con.execute("SELECT DISTINCT scope FROM trials")]

    # -- deflation helper --------------------------------------------------

    def deflation_inputs(self, scope: str | None = None) -> dict:
        """Everything :func:`qe_eval.stats.deflated_sharpe` needs, in one call."""
        var = self.sharpe_variance(scope)
        return {
            "n_trials": max(self.n_trials(scope), 2),
            "var_sr_trials": var,
            "var_sr_known": var is not None,
        }

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __len__(self) -> int:
        return self.n_trials()

    def __repr__(self) -> str:
        return f"<TrialRegistry {self.path!r}, {self.n_trials()} trials>"


def record_many(reg: TrialRegistry, rows: Iterable[dict]) -> list[int]:
    """Convenience for backfilling a registry from a sweep."""
    return [reg.record(**row) for row in rows]
