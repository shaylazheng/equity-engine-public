"""The leak suite — the most important tests in the repo.

The point of these is not that `as_of` filters correctly. It is that a *failure*
to filter is loud. A PIT layer that silently returns the right answer looks
exactly like one that was never wired in, which is why the guard raises and why
these tests deliberately bypass the filter to prove the guard fires.
"""

from __future__ import annotations

import pandas as pd
import pytest
from qe_core.panel import AsOfPanel, AsOfView, LookAheadError, PanelSchemaError
from qe_core.synthetic import generate


def _fact(permno, concept, value, period_end, reported_at, knowledge_date, period_start=None):
    return {
        "permno": permno,
        "concept": concept,
        "value": value,
        "period_start": pd.Timestamp(period_start) if period_start else pd.NaT,
        "period_end": pd.Timestamp(period_end),
        "reported_at": pd.Timestamp(reported_at),
        "knowledge_date": pd.Timestamp(knowledge_date),
        "raw_concept": concept,
        "source": "test",
        "source_url": "test://fixture",
    }


@pytest.fixture(scope="module")
def synth():
    return generate(n_firms=40, start="2015-01-02", end="2016-12-30", seed=3)


# -- the core guarantee ---------------------------------------------------


def test_as_of_never_returns_future_knowledge(synth):
    for t in ("2015-06-15", "2016-01-04", "2016-11-30"):
        view = synth.panel.as_of(t)
        assert len(view) > 0, f"no data at {t} — fixture is degenerate, not passing"
        assert view.frame["knowledge_date"].max() <= pd.Timestamp(t)


def test_the_guard_fires_when_the_filter_is_bypassed(synth):
    """If this test passes trivially, the PIT layer is decorative.

    Hand an AsOfView the *unfiltered* frame. The guard must reject it — that is
    what proves `as_of`'s correctness is enforced rather than merely intended.
    """
    everything = synth.panel.as_of("2016-12-30").frame
    early = pd.Timestamp("2015-03-02")
    assert everything["knowledge_date"].max() > early

    with pytest.raises(LookAheadError, match="not knowable"):
        AsOfView(everything, early)


def test_guard_message_names_the_damage(synth):
    everything = synth.panel.as_of("2016-12-30").frame
    with pytest.raises(LookAheadError) as exc:
        AsOfView(everything, pd.Timestamp("2015-03-02"))
    msg = str(exc.value)
    assert "row(s)" in msg and "day(s) in the future" in msg


# -- the reporting-lag trap ----------------------------------------------


def test_fundamentals_are_gated_on_report_date_not_period_end(synth):
    """The classic backtest killer, in miniature.

    A 2015 fiscal year ends 2015-12-31 but is not reported until ~2016-03-16.
    Between those dates the number exists and must be invisible.
    """
    inside_the_gap = pd.Timestamp("2016-02-01")
    view = synth.panel.as_of(inside_the_gap)
    fund = view.frame.loc[view.frame["concept"] == "at"]

    # Nothing from the 2015 fiscal year is knowable yet.
    fy2015 = fund.loc[fund["period_end"] == pd.Timestamp("2015-12-31")]
    assert fy2015.empty, "FY2015 fundamentals leaked before their report date"

    # ...and after the report date they are.
    later = synth.panel.as_of("2016-04-01")
    lf = later.frame
    got = lf.loc[(lf["concept"] == "at") & (lf["period_end"] == pd.Timestamp("2015-12-31"))]
    assert not got.empty, "FY2015 fundamentals never became visible"


def test_breaking_pit_changes_the_answer(synth):
    """The mirror of the guard test: if breaking PIT is harmless, PIT isn't binding.

    Compare the honest view against one that (wrongly) treats period_end as the
    knowable date. The broken variant must see strictly more.
    """
    t = pd.Timestamp("2016-02-01")

    honest = len(synth.panel.as_of(t).frame.query("concept == 'at'"))

    raw = synth.panel.as_of("2016-12-30", revisions=True).frame
    broken = raw.assign(knowledge_date=raw["period_end"])
    broken_count = len(
        AsOfPanel(broken, validate=False).as_of(t).frame.query("concept == 'at'")
    )

    assert broken_count > honest, (
        "treating period_end as knowable changed nothing — the PIT layer is not "
        "actually gating fundamentals"
    )


# -- restatements ---------------------------------------------------------


def test_restatement_filed_later_is_invisible():
    panel = AsOfPanel(
        pd.DataFrame(
            [
                _fact(1, "ni", 100.0, "2020-12-31", "2021-02-15", "2021-02-15"),
                _fact(1, "ni", 90.0, "2020-12-31", "2021-08-01", "2021-08-01"),  # restated down
            ]
        )
    )

    before = panel.as_of("2021-06-01").field("ni")
    assert before.loc[1] == 100.0, "saw a restatement before it was filed"

    after = panel.as_of("2021-09-01").field("ni")
    assert after.loc[1] == 90.0, "restatement did not supersede the original"


def test_revisions_flag_exposes_full_lineage():
    panel = AsOfPanel(
        pd.DataFrame(
            [
                _fact(1, "ni", 100.0, "2020-12-31", "2021-02-15", "2021-02-15"),
                _fact(1, "ni", 90.0, "2020-12-31", "2021-08-01", "2021-08-01"),
            ]
        )
    )
    assert len(panel.as_of("2021-09-01")) == 1
    assert len(panel.as_of("2021-09-01", revisions=True)) == 2


# -- schema gates ---------------------------------------------------------


def test_empty_source_url_is_rejected():
    """P3 was a CHECK constraint in Postgres. Parquet enforces nothing, so we do."""
    bad = pd.DataFrame([_fact(1, "ni", 1.0, "2020-12-31", "2021-02-15", "2021-02-15")])
    bad.loc[0, "source_url"] = "   "
    with pytest.raises(PanelSchemaError, match="source_url"):
        AsOfPanel(bad)


def test_knowing_before_asserting_is_rejected():
    bad = pd.DataFrame([_fact(1, "ni", 1.0, "2020-12-31", "2021-02-15", "2021-01-01")])
    with pytest.raises(PanelSchemaError, match="knowledge_date < reported_at"):
        AsOfPanel(bad)


def test_missing_column_is_rejected():
    df = pd.DataFrame([_fact(1, "ni", 1.0, "2020-12-31", "2021-02-15", "2021-02-15")])
    with pytest.raises(PanelSchemaError, match="missing required column"):
        AsOfPanel(df.drop(columns=["reported_at"]))


def test_nulls_in_required_columns_are_rejected():
    df = pd.DataFrame([_fact(1, "ni", 1.0, "2020-12-31", "2021-02-15", "2021-02-15")])
    df.loc[0, "value"] = None
    with pytest.raises(PanelSchemaError, match="null values"):
        AsOfPanel(df)


def test_field_returns_one_row_per_name_not_per_period(synth):
    """A view holds every fiscal period, so a cross-sectional read must collapse.

    Before this was explicit, `field()` returned one row per (permno, period) —
    duplicating the index and leaving which period you got to incidental row
    ordering. Any signal reading a fundamental was relying on that.
    """
    view = synth.panel.as_of("2016-12-30")
    dp = view.field("dp")
    assert dp.index.is_unique, "field() returned multiple periods per permno"

    # ...and it is the LATEST knowable period, not an arbitrary one.
    hist = view.series("dp")
    latest = hist.iloc[-1].dropna()
    common = dp.index.intersection(latest.index)
    assert len(common) > 0
    pd.testing.assert_series_equal(
        dp.reindex(common).astype(float),
        latest.reindex(common).astype(float),
        check_names=False,
    )


def test_pivot_is_also_one_row_per_name(synth):
    view = synth.panel.as_of("2016-12-30")
    wide = view.pivot(["at", "ni", "dp"])
    assert wide.index.is_unique
    assert list(wide.columns) == ["at", "ni", "dp"]
