"""Bracket tables, loaded from year-keyed YAML rather than hardcoded.

Rates and thresholds move every year, and the OBBBA moved the standard deduction
mid-stream in 2025. Anything baked into Python is wrong within twelve months and
wrong silently, so the tables live in `configs/tax/<year>.yaml` with their source
URL and the date they were checked.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from qe_core.config import repo_root

__all__ = ["Band", "BracketTable", "StatusTable", "UnverifiedStatusError", "load_brackets"]


class UnverifiedStatusError(RuntimeError):
    """A filing status whose figures were derived rather than sourced."""


@dataclass(frozen=True)
class Band:
    """One rate band. `upto` is the upper bound of taxable income; None = open-ended."""

    upto: float | None
    rate: float

    @property
    def ceiling(self) -> float:
        return float("inf") if self.upto is None else float(self.upto)


@dataclass(frozen=True)
class StatusTable:
    status: str
    standard_deduction: float
    ordinary: tuple[Band, ...]
    long_term: tuple[Band, ...]
    verified: bool

    @property
    def zero_ltcg_ceiling(self) -> float:
        """Top of the 0% long-term band — the headroom calculation's whole basis."""
        for band in self.long_term:
            if band.rate == 0.0:
                return band.ceiling
        return 0.0


@dataclass(frozen=True)
class BracketTable:
    tax_year: int
    statuses: dict[str, StatusTable]
    niit_rate: float
    niit_thresholds: dict[str, float]
    loss_offset: float
    loss_offset_mfs: float
    sources: tuple[str, ...]
    checked_on: str

    def status(self, name: str, *, allow_unverified: bool = False) -> StatusTable:
        if name not in self.statuses:
            raise KeyError(
                f"no bracket table for filing status {name!r} in {self.tax_year}; "
                f"have {sorted(self.statuses)}"
            )
        table = self.statuses[name]
        if not table.verified and not allow_unverified:
            raise UnverifiedStatusError(
                f"{name!r} figures for {self.tax_year} were derived, not sourced. Confirm them "
                "against the IRS publication, set `verified: true` in the config, or pass "
                "allow_unverified=True to proceed knowing the numbers may be wrong."
            )
        return table

    def niit_threshold(self, status: str) -> float:
        return float(self.niit_thresholds.get(status, self.niit_thresholds["single"]))

    def ordinary_offset(self, status: str) -> float:
        return self.loss_offset_mfs if status == "mfs" else self.loss_offset


def _bands(raw: list[dict]) -> tuple[Band, ...]:
    bands = tuple(Band(upto=b["upto"], rate=float(b["rate"])) for b in raw)
    if not bands:
        raise ValueError("empty bracket list")
    ceilings = [b.ceiling for b in bands]
    if ceilings != sorted(ceilings):
        raise ValueError(f"bracket ceilings are not ascending: {ceilings}")
    if ceilings[-1] != float("inf"):
        raise ValueError("the final band must be open-ended (upto: null)")
    rates = [b.rate for b in bands]
    if rates != sorted(rates):
        raise ValueError(f"bracket rates are not ascending: {rates}")
    return bands


def load_brackets(tax_year: int, path: str | Path | None = None) -> BracketTable:
    """Load a year's table. Defaults to `configs/tax/<year>.yaml` under the repo root."""
    p = Path(path) if path is not None else repo_root() / "configs" / "tax" / f"{tax_year}.yaml"
    if not p.exists():
        raise FileNotFoundError(
            f"no bracket table for {tax_year} at {p}. Rates change annually — add the file "
            "with its source URL rather than reusing another year's."
        )
    with open(p) as fh:
        raw = yaml.safe_load(fh)

    meta = raw.get("meta", {})
    if int(meta.get("tax_year", tax_year)) != tax_year:
        raise ValueError(f"{p} declares tax_year {meta.get('tax_year')}, expected {tax_year}")

    statuses = {
        name: StatusTable(
            status=name,
            standard_deduction=float(cfg["standard_deduction"]),
            ordinary=_bands(cfg["ordinary"]),
            long_term=_bands(cfg["long_term"]),
            verified=bool(cfg.get("verified", False)),
        )
        for name, cfg in raw["filing_status"].items()
    }

    return BracketTable(
        tax_year=tax_year,
        statuses=statuses,
        niit_rate=float(raw["niit"]["rate"]),
        niit_thresholds={k: float(v) for k, v in raw["niit"]["magi_threshold"].items()},
        loss_offset=float(raw["capital_loss"]["annual_ordinary_offset"]),
        loss_offset_mfs=float(raw["capital_loss"]["offset_mfs"]),
        sources=tuple(meta.get("sources", ())),
        checked_on=str(meta.get("checked_on", "unknown")),
    )


def tax_from_bands(taxable: float, bands: tuple[Band, ...], floor: float = 0.0) -> float:
    """Tax on income in (floor, floor + taxable], stacked through `bands`.

    `floor` is what already sits underneath — this is how long-term gains stack on
    top of ordinary income rather than being taxed from zero.
    """
    if taxable <= 0:
        return 0.0
    total = 0.0
    lo, hi = floor, floor + taxable
    prev = 0.0
    for band in bands:
        seg_lo, seg_hi = max(lo, prev), min(hi, band.ceiling)
        if seg_hi > seg_lo:
            total += (seg_hi - seg_lo) * band.rate
        prev = band.ceiling
        if prev >= hi:
            break
    return total
