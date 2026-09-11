"""Self-contained HTML reports.

The Phase 10 requirement is blunt: **a report that shows only a total fails**. A
score of 2.1 is uninterpretable. "2.1, of which 1.4 is insider cluster buying and
0.9 is profitability, against -0.2 of momentum, with no unusual size exposure" is
a claim you can argue with, and being able to disagree with the engine is the
point of being able to see inside it.

So every score renders with its constituents — by signal, by family, by tier —
and risk exposures render *alongside but outside* the sum, so "scores well" can
be told apart from "is a small-cap value tilt wearing a hat". Flags render on
their own board and never inside a score.

No external assets: charts are inline SVG, styles are inline CSS. A report that
needs a CDN is one that stops working the moment you open it on a plane.
"""

from __future__ import annotations

import html
from dataclasses import dataclass

import numpy as np
import pandas as pd
from qe_core.ledger import AttributionLedger

__all__ = ["TearsheetInputs", "render_tearsheet", "sparkline_svg", "summary_stats"]


@dataclass
class TearsheetInputs:
    title: str
    equity: pd.Series
    ledger: AttributionLedger
    #: permno -> {flag_name: bool}, rendered on its own board, never summed.
    flags: pd.DataFrame | None = None
    #: permno x factor risk exposures, shown beside the score and outside it.
    exposures: pd.DataFrame | None = None
    benchmark: pd.Series | None = None
    notes: tuple[str, ...] = ()


def summary_stats(equity: pd.Series, periods: int = 252) -> dict[str, float]:
    """Headline numbers. Deliberately few — a wall of statistics hides the answer."""
    rets = equity.pct_change().dropna()
    if rets.empty:
        return {}

    years = max(len(rets) / periods, 1e-9)
    total = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    cagr = float((1.0 + total) ** (1.0 / years) - 1.0) if total > -1 else -1.0
    vol = float(rets.std(ddof=1) * np.sqrt(periods))
    sharpe = float(rets.mean() / rets.std(ddof=1) * np.sqrt(periods)) if rets.std(ddof=1) else 0.0

    curve = equity / equity.cummax()
    return {
        "total_return": total,
        "cagr": cagr,
        "volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": float(curve.min() - 1.0),
        "n_days": float(len(equity)),
    }


def sparkline_svg(series: pd.Series, width: int = 720, height: int = 180) -> str:
    """Inline SVG line chart. No library, no network."""
    vals = series.dropna().to_numpy(dtype=float)
    if len(vals) < 2:
        return "<p class='muted'>not enough data to plot</p>"

    lo, hi = float(vals.min()), float(vals.max())
    span = hi - lo or 1.0
    pad = 8
    xs = np.linspace(pad, width - pad, len(vals))
    ys = height - pad - (vals - lo) / span * (height - 2 * pad)
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys, strict=True))

    baseline = height - pad - (vals[0] - lo) / span * (height - 2 * pad)
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'role="img" aria-label="equity curve">'
        f'<line x1="{pad}" y1="{baseline:.1f}" x2="{width - pad}" y2="{baseline:.1f}" '
        f'stroke="var(--rule)" stroke-dasharray="3 3" stroke-width="1"/>'
        f'<polyline points="{pts}" fill="none" stroke="var(--accent)" stroke-width="2" '
        f'stroke-linejoin="round" stroke-linecap="round"/>'
        f"</svg>"
    )


def _esc(x: object) -> str:
    return html.escape(str(x))


def _table(df: pd.DataFrame, *, floatfmt: str = "{:,.4f}", classes: str = "") -> str:
    if df.empty:
        return "<p class='muted'>nothing to show</p>"

    head = "".join(f"<th>{_esc(c)}</th>" for c in df.columns)
    body = []
    for idx, row in df.iterrows():
        cells = []
        for v in row:
            if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool):
                cells.append(f"<td class='num'>{floatfmt.format(v)}</td>")
            else:
                cells.append(f"<td>{_esc(v)}</td>")
        body.append(f"<tr><th class='rowhead'>{_esc(idx)}</th>{''.join(cells)}</tr>")

    return (
        f"<div class='scroll'><table class='{classes}'>"
        f"<thead><tr><th></th>{head}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def _constituent_sections(ledger: AttributionLedger, exposures: pd.DataFrame | None) -> str:
    """The part that makes a total insufficient on its own."""
    totals = ledger.totals()
    if totals.empty:
        return "<p class='muted'>empty ledger</p>"

    last_date = ledger.frame["date"].max()
    top = (
        totals.xs(last_date, level="date")
        .sort_values(ascending=False)
        .head(10)
        .rename("score")
        .to_frame()
    )

    blocks = [
        "<h3>Top names, with their constituents</h3>",
        _table(top, floatfmt="{:,.3f}"),
    ]

    for permno in top.index[:5]:
        detail = ledger.by_signal(int(permno), last_date)[
            ["signal", "family", "tier", "weight", "z_value", "contribution"]
        ].set_index("signal")
        total = detail["contribution"].sum()
        blocks.append(
            f"<h4>permno {permno} &mdash; score {total:,.3f} "
            f"<span class='muted'>= sum of {len(detail)} contributions</span></h4>"
        )
        blocks.append(_table(detail, floatfmt="{:,.4f}"))

        if exposures is not None and permno in exposures.index:
            exp = exposures.loc[[permno]].T
            exp.columns = ["exposure"]
            blocks.append(
                "<p class='muted'>Risk exposures &mdash; shown beside the score, "
                "deliberately <em>not</em> part of it:</p>"
            )
            blocks.append(_table(exp, floatfmt="{:,.3f}"))

    fam = ledger.by_family().xs(last_date, level="date").groupby("family").sum()
    tier = ledger.by_tier().xs(last_date, level="date").groupby("tier").sum()

    blocks.append("<h3>Where the score comes from, in aggregate</h3>")
    blocks.append(_table(fam.rename("contribution").to_frame(), floatfmt="{:,.3f}"))
    blocks.append(
        "<p class='muted'>By tier &mdash; how much rests on unvalidated "
        "exploratory signals:</p>"
    )
    blocks.append(_table(tier.rename("contribution").to_frame(), floatfmt="{:,.3f}"))
    return "\n".join(blocks)


def _flag_board(flags: pd.DataFrame | None) -> str:
    """Confluence view: rank by the *count* of active flags, never a weighted sum.

    There is deliberately nowhere to put a weight — that is what keeps the
    constituents visible. The rank always travels with the named set that
    produced it.
    """
    if flags is None or flags.empty:
        return "<p class='muted'>no flags configured</p>"

    counts = flags.astype(bool).sum(axis=1).rename("active_flags")
    active = counts[counts > 0].sort_values(ascending=False).head(20)
    if active.empty:
        return "<p class='muted'>no active flags</p>"

    rows = []
    for permno, n in active.items():
        which = ", ".join(sorted(flags.columns[flags.loc[permno].astype(bool)]))
        rows.append({"permno": permno, "count": int(n), "flags": which})

    board = pd.DataFrame(rows).set_index("permno")
    return (
        "<p class='muted'>Ranked by count of simultaneously active flags. Flags never "
        "enter a score &mdash; they sit beside it.</p>" + _table(board, floatfmt="{:,.0f}")
    )


_CSS = """
:root{color-scheme:light;--plane:#eeeef1;--surface:#fafafc;--surface-2:#f3f3f7;
--ink:#17171f;--ink-2:#5b5b6a;--ink-3:#8b8b9a;--rule:#e0e0e6;--accent:#3f3d9e;}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--plane:#0f0f13;
--surface:#17171d;--surface-2:#1e1e26;--ink:#f3f3f7;--ink-2:#a8a8ba;--ink-3:#74748a;
--rule:#2a2a34;--accent:#8b87f0;}}
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--ink);font:15px/1.55 system-ui,
-apple-system,"Helvetica Neue",Arial,sans-serif;-webkit-font-smoothing:antialiased}
.wrap{max-width:1000px;margin:0 auto;padding:40px 24px 80px}
h1{font-size:30px;margin:0 0 6px;letter-spacing:-.01em}
h2{font-size:22px;margin:38px 0 10px}
h3{font-size:17px;margin:26px 0 8px}
h4{font-size:14.5px;margin:20px 0 6px;font-weight:640}
.muted{color:var(--ink-3);font-size:13px}
.card{background:var(--surface);border:1px solid var(--rule);border-radius:12px;
padding:20px 22px;margin:16px 0}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px}
.stat{background:var(--surface);border:1px solid var(--rule);border-radius:10px;padding:14px 16px}
.stat .v{font-size:24px;font-weight:650;letter-spacing:-.02em}
.stat .k{font-size:12px;color:var(--ink-2);margin-top:5px}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px;margin:6px 0}
th,td{padding:6px 10px;text-align:left;border-bottom:1px solid var(--rule)}
thead th{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-3)}
.rowhead{font-weight:600;white-space:nowrap}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.note{font-size:13px;color:var(--ink-2);border-left:3px solid var(--accent);
padding-left:12px;margin:10px 0}
"""


def render_tearsheet(inputs: TearsheetInputs) -> str:
    """Render a complete, self-contained HTML report."""
    stats = summary_stats(inputs.equity)
    tiles = "".join(
        f"<div class='stat'><div class='v'>{v}</div><div class='k'>{k}</div></div>"
        for k, v in [
            ("Total return", f"{stats.get('total_return', 0):+.1%}"),
            ("CAGR", f"{stats.get('cagr', 0):+.1%}"),
            ("Volatility", f"{stats.get('volatility', 0):.1%}"),
            ("Sharpe", f"{stats.get('sharpe', 0):.2f}"),
            ("Max drawdown", f"{stats.get('max_drawdown', 0):.1%}"),
        ]
    )

    notes = "".join(f"<p class='note'>{_esc(n)}</p>" for n in inputs.notes)

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(inputs.title)}</title><style>{_CSS}</style></head><body>
<div class="wrap">
  <h1>{_esc(inputs.title)}</h1>
  <p class="muted">Every number here expands into its inputs. A total on its own is not a result.</p>
  {notes}

  <h2>Performance</h2>
  <div class="stats">{tiles}</div>
  <div class="card">{sparkline_svg(inputs.equity)}</div>

  <h2>Score attribution</h2>
  <div class="card">{_constituent_sections(inputs.ledger, inputs.exposures)}</div>

  <h2>Flag board</h2>
  <div class="card">{_flag_board(inputs.flags)}</div>
</div></body></html>"""
