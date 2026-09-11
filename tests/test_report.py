"""Reporting — a report that shows only a total fails Phase 10."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest
from qe_core.ledger import AttributionLedger
from qe_report.tearsheet import (
    TearsheetInputs,
    render_tearsheet,
    sparkline_svg,
    summary_stats,
)

DATE = pd.Timestamp("2024-06-28")


@pytest.fixture
def equity():
    rng = np.random.default_rng(2)
    idx = pd.date_range("2020-01-01", periods=500, freq="B")
    return pd.Series(100_000 * np.cumprod(1 + rng.normal(0.0004, 0.01, 500)), index=idx)


@pytest.fixture
def ledger():
    rows = []
    for permno in range(10_001, 10_009):
        for signal, family, tier, contrib in [
            ("insider_cluster_buy", "insider", "core", 1.4 - permno % 3),
            ("gross_profitability", "quality", "core", 0.9),
            ("momentum_12_1", "momentum", "exploratory", -0.2),
        ]:
            rows.append(
                {
                    "permno": permno, "date": DATE, "signal": signal, "family": family,
                    "tier": tier, "weight": 0.33, "z_value": contrib / 0.33,
                    "contribution": contrib,
                }
            )
    frame = pd.DataFrame(rows)
    for col in ("signal", "family", "tier"):
        frame[col] = frame[col].astype("string")
    return AttributionLedger(frame)


@pytest.fixture
def exposures():
    rng = np.random.default_rng(5)
    return pd.DataFrame(
        rng.normal(0, 1, size=(8, 3)),
        index=range(10_001, 10_009),
        columns=["market", "size", "value"],
    )


@pytest.fixture
def flags():
    return pd.DataFrame(
        {
            "restatement_8k": [True, False, False, True, False, False, False, False],
            "auditor_change": [True, True, False, False, False, False, False, False],
            "going_concern": [True, False, False, False, False, False, False, False],
        },
        index=range(10_001, 10_009),
    )


@pytest.fixture
def html(equity, ledger, exposures, flags):
    return render_tearsheet(
        TearsheetInputs(
            title="Equity Alpha Engine — smoke",
            equity=equity,
            ledger=ledger,
            exposures=exposures,
            flags=flags,
            notes=("Synthetic panel. No synthetic dataset connection.",),
        )
    )


# -- self-contained -------------------------------------------------------


def test_no_external_requests(html):
    """A report that needs a CDN stops working the moment you open it on a plane."""
    for pattern in ("http://", "https://", "src=", "href="):
        assert pattern not in html, f"report reaches out via {pattern!r}"


def test_it_is_a_complete_document(html):
    assert html.lstrip().startswith("<!doctype html>")
    assert html.rstrip().endswith("</html>")
    assert "<style>" in html


def test_styles_are_inline(html):
    assert "--accent" in html and "prefers-color-scheme" in html


# -- the constituents, which are the point --------------------------------


def test_per_signal_breakdown_is_present(html):
    for signal in ("insider_cluster_buy", "gross_profitability", "momentum_12_1"):
        assert signal in html, f"{signal} missing — the score is not decomposed"


def test_family_and_tier_aggregates_are_present(html):
    assert "insider" in html and "quality" in html and "momentum" in html
    assert "exploratory" in html and "core" in html


def test_a_score_is_shown_as_a_sum_of_its_parts(html):
    assert "= sum of" in html, "a total was rendered without saying what it sums"


def test_risk_exposures_render_outside_the_score(html):
    assert "not</em> part of it" in html or "not part of it" in html
    assert "market" in html and "size" in html


def test_the_tier_view_is_labelled_by_what_it_means(html):
    assert "unvalidated" in html


# -- flag board -----------------------------------------------------------


def test_flag_board_ranks_by_count(html):
    assert "count of simultaneously active flags" in html
    assert "restatement_8k" in html


def test_flags_are_stated_to_be_outside_the_score(html):
    assert "never" in html and "score" in html


def test_the_highest_confluence_name_leads(flags, equity, ledger):
    out = render_tearsheet(TearsheetInputs("t", equity, ledger, flags=flags))
    # permno 10001 has three active flags, more than any other.
    body = out[out.index("Flag board"):]
    assert body.index("10001") < body.index("10004")


def test_missing_flags_degrade_gracefully(equity, ledger):
    out = render_tearsheet(TearsheetInputs("t", equity, ledger, flags=None))
    assert "no flags configured" in out


def test_all_inactive_flags_say_so(equity, ledger):
    quiet = pd.DataFrame({"restatement_8k": [False] * 8}, index=range(10_001, 10_009))
    out = render_tearsheet(TearsheetInputs("t", equity, ledger, flags=quiet))
    assert "no active flags" in out


# -- summary statistics ---------------------------------------------------


def test_cagr_matches_a_hand_computed_value():
    idx = pd.date_range("2020-01-01", periods=253, freq="B")
    eq = pd.Series(np.linspace(100.0, 110.0, 253), index=idx)
    stats = summary_stats(eq)
    assert stats["total_return"] == pytest.approx(0.10)
    # 252 return observations = one year, so CAGR is the total return.
    assert stats["cagr"] == pytest.approx(0.10, rel=0.02)


def test_max_drawdown_matches_a_known_path():
    eq = pd.Series([100.0, 120.0, 60.0, 90.0])
    assert summary_stats(eq)["max_drawdown"] == pytest.approx(-0.5)


def test_a_flat_curve_has_zero_drawdown():
    eq = pd.Series([100.0] * 50)
    assert summary_stats(eq)["max_drawdown"] == pytest.approx(0.0)


def test_summary_of_an_empty_series_is_empty():
    assert summary_stats(pd.Series([100.0])) == {}


# -- chart ----------------------------------------------------------------


def test_sparkline_is_valid_svg(equity):
    svg = sparkline_svg(equity)
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    assert "polyline" in svg
    assert re.search(r'viewBox="0 0 \d+ \d+"', svg)


def test_sparkline_handles_too_little_data():
    assert "not enough data" in sparkline_svg(pd.Series([1.0]))


def test_sparkline_has_no_external_reference(equity):
    assert "http" not in sparkline_svg(equity)


# -- escaping -------------------------------------------------------------


def test_titles_are_escaped(equity, ledger):
    out = render_tearsheet(TearsheetInputs("<script>alert(1)</script>", equity, ledger))
    assert "<script>alert(1)</script>" not in out
    assert "&lt;script&gt;" in out


def test_notes_are_escaped(equity, ledger):
    out = render_tearsheet(
        TearsheetInputs("t", equity, ledger, notes=("<img onerror=x>",))
    )
    assert "<img onerror=x>" not in out
