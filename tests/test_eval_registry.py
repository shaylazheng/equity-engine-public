"""The trial registry — without a real count, Deflated Sharpe is theatre."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from qe_eval.registry import TrialRegistry, config_hash
from qe_eval.stats import deflated_sharpe


@pytest.fixture
def reg():
    with TrialRegistry() as r:
        yield r


def _add(reg, name, sr, scope="default"):
    return reg.record(
        signals=[name], universe="us_all", start="2000-01-01", end="2020-12-31",
        sr_period=sr, n_obs=5000, scope=scope,
    )


# -- counting -------------------------------------------------------------


def test_counts_what_was_actually_tried(reg):
    for i in range(7):
        _add(reg, f"sig_{i}", 0.01 * i)
    assert reg.n_trials() == 7
    assert len(reg) == 7


def test_rerunning_an_identical_config_is_not_a_new_trial(reg):
    a = _add(reg, "momentum_12_1", 0.05)
    b = _add(reg, "momentum_12_1", 0.05)
    assert a == b
    assert reg.n_trials() == 1, "re-running the same backtest inflated the trial count"


def test_changing_the_config_is_a_new_trial(reg):
    _add(reg, "momentum_12_1", 0.05)
    reg.record(
        signals=["momentum_12_1"], universe="us_large", start="2000-01-01",
        end="2020-12-31", sr_period=0.05, n_obs=5000,
    )
    assert reg.n_trials() == 2


def test_scopes_are_counted_separately(reg):
    _add(reg, "a", 0.1, scope="core")
    _add(reg, "b", 0.2, scope="core")
    _add(reg, "c", 0.3, scope="exploratory")
    assert reg.n_trials("core") == 2
    assert reg.n_trials("exploratory") == 1
    assert reg.n_trials() == 3
    assert sorted(reg.scopes()) == ["core", "exploratory"]


# -- the variance that DSR actually needs ---------------------------------


def test_sharpe_variance_matches_numpy(reg):
    values = [0.02, 0.11, -0.04, 0.07, 0.15]
    for i, v in enumerate(values):
        _add(reg, f"s{i}", v)
    assert reg.sharpe_variance() == pytest.approx(np.var(values, ddof=1))


def test_sharpe_variance_is_undefined_below_two_trials(reg):
    assert reg.sharpe_variance() is None
    _add(reg, "only", 0.1)
    assert reg.sharpe_variance() is None


def test_deflation_inputs_say_whether_the_variance_is_real(reg):
    assert reg.deflation_inputs()["var_sr_known"] is False
    _add(reg, "a", 0.1)
    _add(reg, "b", 0.3)
    inputs = reg.deflation_inputs()
    assert inputs["var_sr_known"] is True
    assert inputs["n_trials"] == 2


def test_registry_actually_changes_the_deflated_sharpe(reg):
    """End to end: the registry must be able to move the number, or it is decorative."""
    rng = np.random.default_rng(8)
    returns = pd.Series(rng.normal(0.0008, 0.01, 1200))

    for i in range(40):
        _add(reg, f"tried_{i}", float(rng.normal(0.02, 0.25)))

    inputs = reg.deflation_inputs()
    informed = deflated_sharpe(returns, 252, inputs["n_trials"], inputs["var_sr_trials"])
    naive = deflated_sharpe(returns, 252, n_trials=2, var_sr_trials=1e-6)

    assert informed["var_sr_source"] == "registry"
    assert informed["dsr"] < naive["dsr"], "40 recorded trials did not deflate the Sharpe"


# -- housekeeping ---------------------------------------------------------


def test_config_hash_is_key_order_independent():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})


def test_config_hash_is_sensitive_to_values():
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_trials_frame_round_trips_the_signal_list(reg):
    reg.record(
        signals=["b", "a"], universe="u", start="2000-01-01", end="2001-01-01",
        sr_period=0.1, n_obs=10,
    )
    df = reg.trials()
    assert df.loc[0, "signals"] == ["a", "b"]  # stored sorted


def test_registry_persists_to_disk(tmp_path):
    path = tmp_path / "nested" / "trials.sqlite"
    with TrialRegistry(path) as r:
        _add(r, "persisted", 0.4)
    with TrialRegistry(path) as r2:
        assert r2.n_trials() == 1
