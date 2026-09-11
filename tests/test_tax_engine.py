"""Tax mechanics, checked against hand-worked cases.

The plan named five reference cases as the Phase 8 acceptance criterion: a
realization entirely inside the 0% band, one straddling the 0%/15% boundary, a
short-term gain pushing long-term gain out of the 0% band, a loss year with a
carryforward crossing into the next year, and a case above the NIIT threshold.
All five are here.
"""

from __future__ import annotations

import pytest
from qe_tax.brackets import UnverifiedStatusError, load_brackets, tax_from_bands
from qe_tax.engine import TaxEngine, TaxProfile, net_gains

# 2026 single filer, from configs/tax/2026.yaml
STD_DEDUCTION = 16_100
ZERO_LTCG_TOP = 49_450


@pytest.fixture(scope="module")
def brackets():
    return load_brackets(2026)


def engine(income=20_000.0, **kw):
    return TaxEngine(TaxProfile(ordinary_income=income, **kw))


# -- the config itself ----------------------------------------------------


def test_bracket_table_carries_its_provenance(brackets):
    assert brackets.sources, "a bracket table with no source URL is unusable"
    assert brackets.checked_on != "unknown"


def test_unverified_filing_status_refuses_to_load_silently(brackets):
    """MFJ long-term thresholds were derived, not sourced. Using them must be deliberate."""
    with pytest.raises(UnverifiedStatusError, match="derived, not sourced"):
        brackets.status("mfj")
    assert brackets.status("mfj", allow_unverified=True).standard_deduction == 32_200


def test_missing_year_does_not_fall_back_to_another(brackets):
    with pytest.raises(FileNotFoundError, match="change annually"):
        load_brackets(1999)


def test_bands_must_be_ordered():
    from qe_tax.brackets import _bands

    with pytest.raises(ValueError, match="ascending"):
        _bands([{"upto": 100, "rate": 0.2}, {"upto": 50, "rate": 0.3}, {"upto": None, "rate": 0.4}])


def test_tax_from_bands_stacks_on_a_floor(brackets):
    single = brackets.status("single")
    # 10k of long-term gain sitting on top of 45k of ordinary income straddles
    # the 49,450 boundary: 4,450 at 0%, the remaining 5,550 at 15%.
    got = tax_from_bands(10_000, single.long_term, floor=45_000)
    assert got == pytest.approx(5_550 * 0.15)


# -- case 1: entirely inside the 0% band ----------------------------------


def test_headroom_on_a_student_income():
    """$20k income: taxable ordinary 3,900, so 45,550 of long-term gain is free."""
    eng = engine(20_000)
    assert eng.headroom() == pytest.approx(ZERO_LTCG_TOP - (20_000 - STD_DEDUCTION))
    assert eng.headroom() == pytest.approx(45_550)


def test_realizing_inside_the_headroom_is_untaxed():
    eng = engine(20_000)
    assert eng.tax_on_realization(40_000, long_term=True) == pytest.approx(0.0)
    assert eng.marginal_rate(long_term=True) == pytest.approx(0.0)


def test_headroom_shrinks_as_it_is_used():
    eng = engine(20_000)
    start = eng.headroom()
    assert eng.headroom(long_gain=10_000) == pytest.approx(start - 10_000)
    assert eng.headroom(long_gain=start + 5_000) == 0.0


def test_income_below_the_deduction_gets_the_whole_band():
    eng = engine(10_000)
    assert eng.headroom() == pytest.approx(ZERO_LTCG_TOP)


# -- case 2: straddling the 0%/15% boundary -------------------------------


def test_marginal_rate_steps_at_the_boundary():
    """The reason a flat rate cannot express this: the rate is a step function."""
    eng = engine(20_000)
    head = eng.headroom()
    assert eng.marginal_rate(long_term=True, long_gain=head - 1_000) == pytest.approx(0.0)
    assert eng.marginal_rate(long_term=True, long_gain=head + 1_000) == pytest.approx(0.15)


def test_a_straddling_realization_is_taxed_only_on_the_excess():
    eng = engine(20_000)
    head = eng.headroom()
    tax = eng.tax_on_realization(head + 10_000, long_term=True)
    assert tax == pytest.approx(10_000 * 0.15)


# -- case 3: short-term gains eat long-term headroom ----------------------


def test_short_term_gain_consumes_long_term_headroom():
    """The interaction a flat model cannot see at all."""
    eng = engine(20_000)
    before = eng.headroom()
    after = eng.headroom(short_gain=15_000)
    assert after == pytest.approx(before - 15_000)


def test_short_term_gains_are_taxed_as_ordinary_income():
    eng = engine(20_000)
    # Taxable ordinary is 3,900, inside the 10% band which runs to 12,400.
    assert eng.marginal_rate(long_term=False) == pytest.approx(0.10)
    # Push well past 50,400 and the ordinary rate steps to 22%.
    assert eng.marginal_rate(long_term=False, short_gain=60_000) == pytest.approx(0.22)


def test_there_is_no_zero_band_for_short_term_gains():
    """The premise this model corrects: a low earner pays a low rate, never zero."""
    eng = engine(20_000)
    assert eng.marginal_rate(long_term=False) > 0.0


def test_the_hidden_second_cost_of_a_short_term_gain():
    """Realizing short-term costs its own rate *plus* the long-term gain it displaces."""
    eng = engine(20_000)
    head = eng.headroom()
    # Long gains exactly fill the band, then a short gain pushes some out.
    without = eng.total_tax(long_gain=head).total
    with_short = eng.total_tax(short_gain=10_000, long_gain=head).total
    direct = 10_000 * 0.10
    assert with_short - without > direct, "the displacement cost was not captured"


# -- case 4: losses, netting, and carryforward ----------------------------


def test_same_character_nets_before_crossing_over():
    n = net_gains(short=5_000, long=-2_000)
    assert n.net_short == pytest.approx(3_000)
    assert n.net_long == pytest.approx(0.0)


def test_net_loss_offsets_ordinary_income_up_to_the_cap():
    n = net_gains(short=-10_000, long=0.0)
    assert n.ordinary_offset == pytest.approx(3_000)
    assert n.carryforward_short == pytest.approx(7_000)


def test_carryforward_preserves_character():
    n = net_gains(short=-4_000, long=-5_000)
    assert n.ordinary_offset == pytest.approx(3_000)
    # Short losses are consumed by the offset first.
    assert n.carryforward_short == pytest.approx(1_000)
    assert n.carryforward_long == pytest.approx(5_000)


def test_carryforward_crosses_into_the_next_year():
    eng = engine(50_000)
    loss_year = eng.total_tax(short_gain=-12_000)
    assert loss_year.netting.carryforward_short == pytest.approx(9_000)

    nxt = eng.next_year(loss_year.netting)
    assert nxt.profile.tax_year == 2027
    assert nxt.profile.carryforward_short == pytest.approx(9_000)


def test_rolling_forward_flags_that_the_brackets_are_stale():
    """Reusing this year's rates for next year is an assumption, and it is visible."""
    eng = engine(50_000)
    assert not eng.brackets_stale

    nxt = eng.next_year(eng.total_tax().netting)
    assert nxt.brackets_stale, "2027 modelled on 2026 rates without saying so"


def test_carryforward_survives_several_years():
    """A multi-year backtest must not fail merely because future tables are absent."""
    eng = engine(50_000)
    netting = eng.total_tax(short_gain=-30_000).netting
    for _ in range(5):
        eng = eng.next_year(netting)
        netting = eng.total_tax().netting
    assert eng.profile.tax_year == 2031
    # 30,000 loss in 2026 leaves 27,000 entering 2027. Each of 2027-2030 absorbs
    # 3,000, so 15,000 *enters* 2031 and 12,000 leaves it. The profile holds the
    # opening balance; the netting holds the closing one.
    assert eng.profile.carryforward_short == pytest.approx(15_000)
    assert netting.carryforward_short == pytest.approx(12_000)


def test_a_carried_loss_shelters_the_next_years_gain():
    eng = TaxEngine(TaxProfile(ordinary_income=50_000, carryforward_short=9_000))
    sheltered = eng.total_tax(short_gain=9_000)
    assert sheltered.netting.net_short == pytest.approx(0.0)


# -- case 5: NIIT ---------------------------------------------------------


def test_niit_does_not_apply_below_the_threshold():
    assert engine(20_000).total_tax(long_gain=30_000).niit == pytest.approx(0.0)


def test_niit_applies_above_the_threshold():
    eng = engine(300_000)
    out = eng.total_tax(long_gain=100_000)
    assert out.niit == pytest.approx(100_000 * 0.038)


def test_top_marginal_long_term_rate_includes_niit():
    eng = engine(700_000)
    assert eng.marginal_rate(long_term=True) == pytest.approx(0.20 + 0.038)


# -- the shelf-life point Phase 8.5 turns on ------------------------------


def test_the_same_gain_costs_wildly_different_amounts_by_income():
    """Why 8.5 reports across profiles instead of one number."""
    gain = 40_000
    low = engine(20_000).tax_on_realization(gain, long_term=True)
    mid = engine(120_000).tax_on_realization(gain, long_term=True)
    high = engine(700_000).tax_on_realization(gain, long_term=True)

    assert low == pytest.approx(0.0)
    assert mid == pytest.approx(gain * 0.15)
    assert high == pytest.approx(gain * (0.20 + 0.038))
    assert high > mid > low


def test_state_tax_is_a_flat_add_on():
    eng = engine(20_000, state_rate=0.05)
    # Federal is zero inside the band; the state still takes its cut.
    out = eng.total_tax(long_gain=30_000)
    assert out.federal_long_term == pytest.approx(0.0)
    assert out.state == pytest.approx(30_000 * 0.05)


def test_profile_rejects_nonsense():
    with pytest.raises(ValueError, match="ordinary_income"):
        TaxProfile(ordinary_income=-1)
    with pytest.raises(ValueError, match="state_rate"):
        TaxProfile(state_rate=1.5)
