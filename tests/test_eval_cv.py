"""Purging and embargo must actually remove the leaking observations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from qe_eval.cv import cpcv_splits, label_spans, n_cpcv_paths, purged_kfold


@pytest.fixture
def spans():
    idx = pd.date_range("2015-01-01", periods=600, freq="B")
    return label_spans(idx, horizon=21)


# -- label spans ----------------------------------------------------------


def test_label_spans_drop_the_unresolved_tail(spans):
    """The last `horizon` observations have no knowable outcome yet."""
    assert len(spans) == 600 - 21


def test_label_spans_end_after_they_start(spans):
    assert (spans.to_numpy() > spans.index.to_numpy()).all()


def test_horizon_must_be_positive():
    idx = pd.date_range("2015-01-01", periods=50, freq="B")
    with pytest.raises(ValueError, match="horizon"):
        label_spans(idx, horizon=0)


# -- purging --------------------------------------------------------------


def test_no_training_label_overlaps_the_test_window(spans):
    """The core guarantee. A training label that resolves inside the test window
    describes an overlapping future, so it leaks."""
    for train, test in purged_kfold(spans, n_splits=5, embargo_pct=0.01):
        t_start = spans.index[test[0]]
        t_end = spans.iloc[test].max()
        train_starts = spans.index[train].to_numpy()
        train_ends = spans.iloc[train].to_numpy()
        overlap = (train_ends >= t_start) & (train_starts <= t_end)
        assert not overlap.any(), f"{overlap.sum()} training labels overlap the test window"


def test_train_and_test_never_intersect(spans):
    for train, test in purged_kfold(spans, n_splits=5):
        assert not set(train) & set(test)


def test_purging_actually_costs_something(spans):
    """If purging removed nothing, it is not wired in."""
    splits = purged_kfold(spans, n_splits=5, embargo_pct=0.0)
    n = len(spans)
    for train, test in splits:
        assert len(train) < n - len(test), "purge removed no observations at all"


def test_embargo_is_forward_only(spans):
    """López de Prado's embargo follows the test set; the inherited code applied
    it symmetrically, which discards usable history for no reason."""
    no_emb = purged_kfold(spans, n_splits=5, embargo_pct=0.0)
    emb = purged_kfold(spans, n_splits=5, embargo_pct=0.05)

    # Pick a middle fold so there is history on both sides.
    train_no, test = no_emb[2]
    train_emb, _ = emb[2]

    dropped = set(train_no) - set(train_emb)
    assert dropped, "embargo removed nothing"
    assert min(dropped) > test[-1], "embargo removed observations *before* the test block"


def test_embargo_is_a_fraction_not_a_raw_count(spans):
    """The inherited bug: `cv_embargo_days: 21` applied to a monthly index meant
    21 months. Expressing it as a fraction of the sample removes the unit."""
    small = purged_kfold(spans, n_splits=5, embargo_pct=0.01)
    large = purged_kfold(spans, n_splits=5, embargo_pct=0.10)
    assert len(large[2][0]) < len(small[2][0])


def test_splits_cover_every_observation_exactly_once_in_test(spans):
    covered = np.concatenate([test for _, test in purged_kfold(spans, n_splits=5)])
    assert sorted(covered) == list(range(len(spans)))


def test_rejects_too_few_splits(spans):
    with pytest.raises(ValueError, match="n_splits"):
        purged_kfold(spans, n_splits=1)


# -- CPCV -----------------------------------------------------------------


def test_cpcv_generates_the_expected_number_of_splits(spans):
    from math import comb

    splits = cpcv_splits(spans, n_groups=6, n_test=2)
    assert len(splits) == comb(6, 2)


def test_cpcv_path_count():
    # C(6,2) = 15 splits, each testing 2 of 6 groups -> 5 distinct backtest paths.
    assert n_cpcv_paths(6, 2) == 5


def _contiguous_runs(idx: np.ndarray) -> list[np.ndarray]:
    """Split an index array into runs of consecutive positions."""
    if idx.size == 0:
        return []
    return np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)


def test_cpcv_also_purges(spans):
    """Checked per contiguous block, not against the whole test set's envelope.

    A CPCV test set is non-contiguous by construction — groups 0 and 3, say — so
    its min-to-max span covers the untested groups in between. Training points
    living in that gap are perfectly legitimate, and an envelope check would
    wrongly flag every one of them.
    """
    for train, test in cpcv_splits(spans, n_groups=6, n_test=2, embargo_pct=0.01):
        assert not set(train) & set(test)
        starts = spans.index[train].to_numpy()
        ends = spans.iloc[train].to_numpy()
        for block in _contiguous_runs(test):
            b_start = spans.index[block[0]]
            b_end = spans.iloc[block].max()
            overlap = (ends >= b_start) & (starts <= b_end)
            assert not overlap.any(), f"{overlap.sum()} training labels overlap a test block"


def test_cpcv_rejects_bad_configuration(spans):
    with pytest.raises(ValueError, match="n_test"):
        cpcv_splits(spans, n_groups=4, n_test=4)
