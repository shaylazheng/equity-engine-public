"""qe-data transforms, tested without a synthetic dataset connection.

The laptop cannot reach synthetic dataset at all, so the pull layer is untestable here by
construction. Everything it *hands off to* is pure and is tested: the delisting
fill, the CCM tiebreak, the YTD differencing, the report-date gate, and panel
assembly. Those are where the silent biases live.
"""

from __future__ import annotations

import pandas as pd
import pytest
from qe_core.panel import PanelSchemaError
from qe_data.delisting import (
    SHUMWAY_NASDAQ,
    SHUMWAY_NYSE_AMEX,
    apply_delisting_returns,
    delisting_fill_summary,
    fill_missing_delisting_returns,
)
from qe_data.fundamentals import assign_knowledge_date, rdq_coverage, ytd_to_quarterly
from qe_data.linking import attach_permno, link_coverage
from qe_data.panel import build_panel, prices_to_facts

# -- SQL builders ---------------------------------------------------------












# -- delisting ------------------------------------------------------------


def _delist_frame():
    return pd.DataFrame(
        {
            "permno": [1, 2, 3, 4, 5],
            "delistingdt": pd.to_datetime(
                ["2010-03-01"] * 5
            ),
            "delret": [-0.42, None, None, None, None],
            "delreasontype": ["BKPY", "BKPY", "BKPY", "UNAV", "DELQ"],
            "primaryexch": ["Q", "Q", "N", "Q", "A"],
        }
    )


def test_present_delisting_returns_are_left_alone():
    out = fill_missing_delisting_returns(_delist_frame())
    assert out.loc[0, "delret_filled"] == pytest.approx(-0.42)
    assert out.loc[0, "delret_source"] == "crsp"


def test_nasdaq_performance_delisting_gets_minus_55():
    """The leg the sibling project could not implement — `exchcd` was never pulled."""
    out = fill_missing_delisting_returns(_delist_frame())
    assert out.loc[1, "delret_filled"] == pytest.approx(SHUMWAY_NASDAQ)
    assert out.loc[1, "delret_source"] == "shumway_nasdaq"


def test_listed_performance_delisting_gets_minus_30():
    out = fill_missing_delisting_returns(_delist_frame())
    assert out.loc[2, "delret_filled"] == pytest.approx(SHUMWAY_NYSE_AMEX)
    assert out.loc[4, "delret_filled"] == pytest.approx(SHUMWAY_NYSE_AMEX)  # AMEX


def test_non_performance_delisting_is_not_filled_with_a_loss():
    """A merger pays the shareholder. Inventing -55% there would be a large
    fabricated loss, which is a worse error than inventing nothing."""
    out = fill_missing_delisting_returns(_delist_frame())
    assert out.loc[3, "delret_filled"] == pytest.approx(0.0)
    assert out.loc[3, "delret_source"] == "zero"


def test_the_two_exchange_legs_actually_differ():
    out = fill_missing_delisting_returns(_delist_frame())
    assert out.loc[1, "delret_filled"] != out.loc[2, "delret_filled"]


def test_fill_summary_counts_the_assumptions():
    summary = delisting_fill_summary(_delist_frame())
    assert summary["shumway_nasdaq"] == 1
    assert summary["crsp"] == 1


def test_delisting_return_is_compounded_not_substituted():
    """The security may have traded on its last day *and then* delisted."""
    prices = pd.DataFrame(
        {
            "permno": [1],
            "dlycaldt": pd.to_datetime(["2010-03-01"]),
            "dlyret": [0.10],
        }
    )
    delist = pd.DataFrame(
        {
            "permno": [1],
            "delistingdt": pd.to_datetime(["2010-03-01"]),
            "delret": [-0.50],
            "delreasontype": ["BKPY"],
            "primaryexch": ["Q"],
        }
    )
    out = apply_delisting_returns(prices, delist)
    assert out.loc[0, "dlyret"] == pytest.approx(1.10 * 0.50 - 1.0)


def test_missing_required_column_is_rejected():
    with pytest.raises(KeyError, match="missing"):
        fill_missing_delisting_returns(pd.DataFrame({"delret": [1.0]}))


# -- CCM linking ----------------------------------------------------------


def test_primary_link_beats_consolidated():
    """The inherited bug: sorting `linkprim` ascending made 'C' win over 'P'."""
    fundamentals = pd.DataFrame(
        {"gvkey": ["001"], "datadate": pd.to_datetime(["2015-06-30"])}
    )
    link = pd.DataFrame(
        {
            "gvkey": ["001", "001"],
            "permno": [111, 222],
            "linkprim": ["C", "P"],
            "linktype": ["LC", "LU"],
            "linkdt": pd.to_datetime(["2000-01-01", "2000-01-01"]),
            "linkenddt": pd.to_datetime(["2030-01-01", "2030-01-01"]),
        }
    )
    out = attach_permno(fundamentals, link)
    assert out.loc[0, "permno"] == 222, "consolidated link won over primary"


def test_link_window_is_respected():
    fundamentals = pd.DataFrame(
        {"gvkey": ["001"], "datadate": pd.to_datetime(["2015-06-30"])}
    )
    link = pd.DataFrame(
        {
            "gvkey": ["001"],
            "permno": [111],
            "linkprim": ["P"],
            "linktype": ["LU"],
            "linkdt": pd.to_datetime(["2020-01-01"]),
            "linkenddt": pd.to_datetime(["2030-01-01"]),
        }
    )
    out = attach_permno(fundamentals, link)
    assert out["permno"].isna().all(), "linked outside the validity window"


def test_unlinked_rows_are_kept_and_counted():
    """Dropping them silently shrinks the universe in a way nobody notices."""
    fundamentals = pd.DataFrame(
        {"gvkey": ["001", "999"], "datadate": pd.to_datetime(["2015-06-30"] * 2)}
    )
    link = pd.DataFrame(
        {
            "gvkey": ["001"],
            "permno": [111],
            "linkprim": ["P"],
            "linktype": ["LU"],
            "linkdt": pd.to_datetime(["2000-01-01"]),
            "linkenddt": pd.to_datetime(["2030-01-01"]),
        }
    )
    out = attach_permno(fundamentals, link)
    assert len(out) == 2
    assert link_coverage(out)["share"] == pytest.approx(0.5)

    dropped = attach_permno(fundamentals, link, keep_unlinked=False)
    assert len(dropped) == 1


# -- year-to-date differencing -------------------------------------------


def _ytd_frame():
    return pd.DataFrame(
        {
            "gvkey": ["001"] * 4,
            "fyearq": [2015] * 4,
            "fqtr": [1, 2, 3, 4],
            "datadate": pd.to_datetime(
                ["2015-03-31", "2015-06-30", "2015-09-30", "2015-12-31"]
            ),
            "oancfy": [100.0, 250.0, 400.0, 600.0],
        }
    )


def test_ytd_differencing_recovers_quarterly_flows():
    """oancfy accumulates within the fiscal year — verified on local demo, mean |value|
    runs 146/313/406/610 across Q1-Q4. Using it raw corrupts any accruals signal."""
    out = ytd_to_quarterly(_ytd_frame(), ["oancfy"])
    assert list(out["oancfy_q"]) == pytest.approx([100.0, 150.0, 150.0, 200.0])


def test_q1_is_taken_as_is():
    out = ytd_to_quarterly(_ytd_frame(), ["oancfy"])
    assert out.loc[out["fqtr"] == 1, "oancfy_q"].iloc[0] == pytest.approx(100.0)


def test_a_gap_in_the_quarter_sequence_yields_nan():
    """Differencing Q4 against Q2 would double-count a quarter."""
    frame = _ytd_frame().drop(index=2)  # drop Q3
    out = ytd_to_quarterly(frame, ["oancfy"])
    assert pd.isna(out.loc[out["fqtr"] == 4, "oancfy_q"].iloc[0])


def test_differencing_resets_across_fiscal_years():
    a = _ytd_frame()
    b = _ytd_frame().assign(fyearq=2016, oancfy=[80.0, 160.0, 240.0, 320.0])
    b["datadate"] = pd.to_datetime(
        ["2016-03-31", "2016-06-30", "2016-09-30", "2016-12-31"]
    )
    out = ytd_to_quarterly(pd.concat([a, b], ignore_index=True), ["oancfy"])
    q1_2016 = out[(out["fyearq"] == 2016) & (out["fqtr"] == 1)]["oancfy_q"].iloc[0]
    assert q1_2016 == pytest.approx(80.0), "Q1 differenced against the prior year"


# -- report date gating ---------------------------------------------------


def _fundq_frame():
    return pd.DataFrame(
        {
            "gvkey": ["001", "002", "003"],
            "datadate": pd.to_datetime(["2015-03-31"] * 3),
            "rdq": pd.to_datetime(["2015-05-15", None, "2015-02-01"]),
        }
    )


def test_rdq_is_used_when_present():
    out = assign_knowledge_date(_fundq_frame())
    assert out.loc[0, "reported_at"] == pd.Timestamp("2015-05-15")
    assert out.loc[0, "report_date_source"] == "rdq"


def test_missing_rdq_falls_back_to_a_lag_and_says_so():
    out = assign_knowledge_date(_fundq_frame())
    assert out.loc[1, "reported_at"] == pd.Timestamp("2015-03-31") + pd.Timedelta(days=90)
    assert out.loc[1, "report_date_source"] == "assumed_lag"


def test_an_rdq_before_period_end_is_rejected():
    """You cannot report a quarter before it ends. Trusting it would be exactly
    the look-ahead the whole layer exists to prevent."""
    out = assign_knowledge_date(_fundq_frame())
    assert out.loc[2, "report_date_source"] == "assumed_lag"
    assert bool(out.loc[2, "report_date_implausible"])
    assert out.loc[2, "reported_at"] > pd.Timestamp("2015-03-31")


def test_knowledge_date_never_precedes_reported_at():
    out = assign_knowledge_date(_fundq_frame())
    assert (out["knowledge_date"] >= out["reported_at"]).all()


def test_coverage_reports_by_decade():
    frame = pd.DataFrame(
        {
            "datadate": pd.to_datetime(["1985-03-31", "1985-06-30", "2015-03-31"]),
            "rdq": pd.to_datetime([None, "1985-08-01", "2015-05-15"]),
        }
    )
    cov = rdq_coverage(assign_knowledge_date(frame))
    assert cov.loc[1980, "share"] == pytest.approx(0.5)
    assert cov.loc[2010, "share"] == pytest.approx(1.0)


# -- panel assembly -------------------------------------------------------


def _prices():
    return pd.DataFrame(
        {
            "permno": [1, 1, 2],
            "dlycaldt": pd.to_datetime(["2015-01-02", "2015-01-05", "2015-01-02"]),
            "dlyret": [0.01, -0.02, 0.03],
            "dlyprc": [10.0, 9.8, 25.0],
            "dlycap": [1e9, 9.8e8, 2.5e9],
        }
    )


def test_prices_have_all_three_axes_coinciding():
    facts = prices_to_facts(_prices())
    assert (facts["reported_at"] == facts["period_end"]).all()
    assert (facts["knowledge_date"] == facts["period_end"]).all()


def test_price_concepts_are_renamed_for_signals():
    facts = prices_to_facts(_prices())
    # `shrout` is derived from dlycap/dlyprc rather than pulled — see the note in
    # panel.py on why a share-class-level count is needed alongside Compustat's
    # company-wide `csho`.
    assert set(facts["concept"]) == {"ret", "prc", "mktcap", "shrout"}
    assert {"dlyret", "dlyprc", "dlycap"} <= set(facts["raw_concept"])


def test_build_panel_produces_a_valid_asof_panel():
    panel = build_panel(_prices())
    view = panel.as_of("2015-01-05")
    assert len(view) > 0
    assert view.frame["knowledge_date"].max() <= pd.Timestamp("2015-01-05")


def test_fundamentals_are_gated_on_the_report_date_end_to_end():
    """The whole point: a Q1 fact must be invisible until it is announced."""
    fundq = pd.DataFrame(
        {
            "gvkey": ["001"],
            "permno": [1],
            "datadate": pd.to_datetime(["2015-03-31"]),
            "fyearq": [2015],
            "fqtr": [1],
            "rdq": pd.to_datetime(["2015-05-15"]),
            "atq": [500.0],
            "niq": [25.0],
        }
    )
    fundq = assign_knowledge_date(fundq)
    panel = build_panel(_prices(), fundq)

    before = panel.as_of("2015-04-30").frame
    assert (before["concept"] == "at").sum() == 0, "Q1 leaked before its report date"

    after = panel.as_of("2015-05-20").frame
    assert (after["concept"] == "at").sum() == 1


def test_panel_rejects_a_knowledge_date_before_reported_at():
    facts = prices_to_facts(_prices())
    facts.loc[0, "knowledge_date"] = pd.Timestamp("2000-01-01")
    with pytest.raises(PanelSchemaError, match="knowledge_date < reported_at"):
        from qe_core.panel import AsOfPanel

        AsOfPanel(facts)


def test_unlinked_fundamentals_are_dropped_from_the_panel():
    """A fact with no permno cannot be attached to a security."""
    fundq = pd.DataFrame(
        {
            "gvkey": ["001"],
            "permno": [None],
            "datadate": pd.to_datetime(["2015-03-31"]),
            "rdq": pd.to_datetime(["2015-05-15"]),
            "atq": [500.0],
        }
    )
    fundq = assign_knowledge_date(fundq)
    panel = build_panel(_prices(), fundq)
    assert (panel.as_of("2015-12-31").frame["concept"] == "at").sum() == 0


