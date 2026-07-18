"""Statistical tests behave correctly on constructed signals."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cryptoforecast.evaluate.stats import (
    block_bootstrap_ci,
    deflated_sharpe_ratio,
    diebold_mariano,
    expected_max_sharpe,
    max_drawdown,
    pesaran_timmermann,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)


@pytest.fixture
def returns() -> pd.Series:
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(0.0, 0.03, 400))


@pytest.mark.unit
def test_dm_flags_a_better_model(returns: pd.Series) -> None:
    perfect = returns.to_numpy()
    naive = np.zeros(len(returns))
    result = diebold_mariano(returns, perfect, naive)
    assert result.statistic < 0  # model has lower loss
    assert result.p_value < 0.01


@pytest.mark.unit
def test_dm_is_antisymmetric(returns: pd.Series) -> None:
    a = returns.to_numpy() * 0.5
    b = np.zeros(len(returns))
    forward = diebold_mariano(returns, a, b).statistic
    backward = diebold_mariano(returns, b, a).statistic
    assert forward == pytest.approx(-backward, rel=1e-6)


@pytest.mark.unit
def test_dm_equal_models_is_nan(returns: pd.Series) -> None:
    same = np.zeros(len(returns))
    assert np.isnan(diebold_mariano(returns, same, same).statistic)


@pytest.mark.unit
def test_pt_detects_perfect_sign_forecast(returns: pd.Series) -> None:
    result = pesaran_timmermann(returns, returns.to_numpy())
    assert result.statistic > 0
    assert result.p_value < 0.01


@pytest.mark.unit
def test_pt_degenerate_forecast_is_nan(returns: pd.Series) -> None:
    assert np.isnan(pesaran_timmermann(returns, np.ones(len(returns))).statistic)


@pytest.mark.unit
def test_sharpe_and_drawdown() -> None:
    r = pd.Series([0.02, 0.01, 0.015, 0.005] * 20)  # positive mean, non-zero variance
    assert sharpe_ratio(r, periods_per_year=252) > 0
    assert sharpe_ratio(pd.Series([0.01] * 5), 252) == 0.0  # zero variance -> undefined
    equity = pd.Series([1.0, 1.2, 0.9, 1.1])
    assert max_drawdown(equity) == pytest.approx(0.9 / 1.2 - 1.0)


@pytest.mark.unit
def test_psr_high_for_strong_signal_and_neutral_for_noise() -> None:
    rng = np.random.default_rng(1)
    strong = pd.Series(rng.normal(0.01, 0.01, 500))  # SR ~ 1 per period
    neutral = pd.Series(rng.normal(0.0, 0.02, 500))
    neutral = neutral - neutral.mean()  # exactly zero sample mean -> PSR = 0.5
    assert probabilistic_sharpe_ratio(strong) > 0.99
    assert probabilistic_sharpe_ratio(neutral) == pytest.approx(0.5, abs=1e-6)


@pytest.mark.unit
def test_deflation_penalizes_multiple_trials() -> None:
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0.005, 0.02, 500))
    assert expected_max_sharpe(20, 0.05) > expected_max_sharpe(2, 0.05)
    assert deflated_sharpe_ratio(r, n_trials=50, trials_sr_std=0.05) <= probabilistic_sharpe_ratio(
        r
    )


@pytest.mark.unit
def test_bootstrap_ci_brackets_point_estimate() -> None:
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.01, 0.02, 400))
    point = sharpe_ratio(r, 365)
    lo, hi = block_bootstrap_ci(r, lambda a: sharpe_ratio(a, 365), n_boot=500)
    assert lo < point < hi
