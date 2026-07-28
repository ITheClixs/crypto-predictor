"""Statistical tests behave correctly on constructed signals."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from cryptoforecast.evaluate.stats import (
    benjamini_hochberg_adjusted,
    block_bootstrap_ci,
    clark_west,
    deflated_sharpe_ratio,
    diebold_mariano,
    expected_max_sharpe,
    holm_adjusted,
    max_drawdown,
    newey_west_lrv,
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
def test_dm_matches_a_hand_computed_value_at_horizon_one(returns: pd.Series) -> None:
    """At h=1 there are no Newey-West lags and HLN is exactly 1, so DM is a t-stat.

    This pins the formula to an independently computable number rather than only
    checking its sign.
    """
    model = returns.to_numpy() * 0.5
    bench = np.zeros(len(returns))
    yt = returns.to_numpy()
    d = (yt - model) ** 2 - (yt - bench) ** 2
    n = d.size
    expected = d.mean() / np.sqrt(np.mean((d - d.mean()) ** 2) / n)
    expected *= np.sqrt((n + 1 - 2 + 0.0) / n)  # HLN with h=1
    result = diebold_mariano(returns, model, bench, horizon=1)
    assert result.statistic == pytest.approx(expected, rel=1e-12)
    assert result.p_value == pytest.approx(
        2.0 * scipy_stats.t.cdf(-abs(expected), df=n - 1), rel=1e-12
    )


@pytest.mark.unit
def test_newey_west_reduces_to_the_variance_at_zero_lags() -> None:
    rng = np.random.default_rng(5)
    x = rng.normal(size=200)
    assert newey_west_lrv(x, lags=0) == pytest.approx(np.mean((x - x.mean()) ** 2))


@pytest.mark.unit
def test_newey_west_stays_non_negative_on_strongly_negatively_autocorrelated_data() -> None:
    """The 1/n normalization is what keeps the Bartlett-weighted sum non-negative."""
    x = np.array([1.0, -1.0] * 40)  # autocorrelation of -1 at lag 1
    for lags in range(0, 8):
        assert newey_west_lrv(x, lags=lags) >= 0.0


@pytest.mark.unit
def test_clark_west_favours_a_model_that_nests_the_benchmark(returns: pd.Series) -> None:
    signal = returns.to_numpy() * 0.5
    bench = np.zeros(len(returns))
    result = clark_west(returns, signal, bench)
    assert result.statistic > 0  # positive favours the model
    assert result.p_value < 0.01  # one-sided


@pytest.mark.unit
def test_dm_is_biased_against_a_useless_nested_model_and_clark_west_is_not() -> None:
    """The reason Clark-West exists, demonstrated by simulation.

    The larger model forecasts pure noise, unrelated to the outcome, so its
    *population* MSPE equals the benchmark's — the null is true. But estimating
    coefficients that are really zero costs it sample MSPE, so the DM statistic
    is systematically positive (it reads that cost as evidence the benchmark
    wins). Clark-West subtracts exactly that term, and its statistic is centered
    on zero as a test statistic under the null should be.

    Note the opposite sign conventions: DM positive means the model looks worse,
    Clark-West positive means it looks better.
    """
    rng = np.random.default_rng(4)
    dm_stats, cw_stats = [], []
    for _ in range(200):
        y = pd.Series(rng.normal(0.0, 0.03, 250))
        useless = rng.normal(0.0, 0.01, 250)  # independent of y
        bench = np.zeros(250)
        dm_stats.append(diebold_mariano(y, useless, bench).statistic)
        cw_stats.append(clark_west(y, useless, bench).statistic)

    assert np.mean(dm_stats) > 1.0  # DM leans hard toward the benchmark
    assert abs(np.mean(cw_stats)) < 0.2  # CW is centered under the null
    # DM "significantly worse" far more often than CW claims significant skill.
    dm_rejects = np.mean([s > 1.96 for s in dm_stats])
    cw_rejects = np.mean([s > 1.645 for s in cw_stats])  # one-sided 5%
    assert dm_rejects > 0.25
    assert cw_rejects < 0.10


@pytest.mark.unit
def test_clark_west_identical_forecasts_is_nan(returns: pd.Series) -> None:
    same = np.zeros(len(returns))
    assert np.isnan(clark_west(returns, same, same).statistic)


@pytest.mark.unit
def test_multiple_testing_leaves_a_single_hypothesis_alone() -> None:
    for adjust in (holm_adjusted, benjamini_hochberg_adjusted):
        assert adjust(np.array([0.03])) == pytest.approx([0.03])


@pytest.mark.unit
def test_holm_matches_the_hand_computed_step_down() -> None:
    p = np.array([0.01, 0.04, 0.03])
    # sorted: .01 (x3), .03 (x2), .04 (x1) -> running max -> .03, .06, .06
    assert holm_adjusted(p) == pytest.approx([0.03, 0.06, 0.06])


@pytest.mark.unit
def test_benjamini_hochberg_matches_the_hand_computed_step_up() -> None:
    p = np.array([0.01, 0.04, 0.03])
    # sorted: .01*3/1=.03, .03*3/2=.045, .04*3/3=.04 -> step-up min from the right
    assert benjamini_hochberg_adjusted(p) == pytest.approx([0.03, 0.04, 0.04])


@pytest.mark.unit
def test_adjusted_p_values_are_monotone_and_bounded() -> None:
    rng = np.random.default_rng(8)
    p = rng.uniform(0, 1, 40)
    order = np.argsort(p)
    for adjust in (holm_adjusted, benjamini_hochberg_adjusted):
        adjusted = adjust(p)
        assert adjusted.min() >= 0.0 and adjusted.max() <= 1.0
        assert np.all(np.diff(adjusted[order]) >= -1e-12)  # order-preserving
        assert np.all(adjusted >= p - 1e-12)  # never more lenient than the raw p


@pytest.mark.unit
def test_holm_is_never_more_lenient_than_benjamini_hochberg() -> None:
    """Controlling the family-wise error rate is strictly harder than the FDR."""
    rng = np.random.default_rng(12)
    p = rng.uniform(0, 0.2, 30)
    assert np.all(holm_adjusted(p) >= benjamini_hochberg_adjusted(p) - 1e-12)


@pytest.mark.unit
def test_multiple_testing_ignores_missing_p_values() -> None:
    p = np.array([0.01, np.nan, 0.02])
    for adjust in (holm_adjusted, benjamini_hochberg_adjusted):
        adjusted = adjust(p)
        assert np.isnan(adjusted[1])
        assert np.isfinite(adjusted[[0, 2]]).all()
        # The NaN must not inflate the family size: two hypotheses, not three.
        assert adjusted[0] == pytest.approx(0.02)


@pytest.mark.unit
def test_multiple_testing_deflates_a_family_of_pure_noise() -> None:
    """20 independent tests of true nulls: some raw p<0.05, none should survive."""
    rng = np.random.default_rng(13)
    p = rng.uniform(0, 1, 20)
    assert (p < 0.05).sum() >= 1  # the false positives the correction exists for
    assert (holm_adjusted(p) < 0.05).sum() == 0


@pytest.mark.unit
def test_pt_detects_perfect_sign_forecast(returns: pd.Series) -> None:
    result = pesaran_timmermann(returns, returns.to_numpy())
    assert result.statistic > 0
    assert result.p_value < 0.01


@pytest.mark.unit
def test_pt_degenerate_forecast_is_nan(returns: pd.Series) -> None:
    assert np.isnan(pesaran_timmermann(returns, np.ones(len(returns))).statistic)


@pytest.mark.unit
def test_pt_is_negative_for_a_reliably_wrong_way_forecast(returns: pd.Series) -> None:
    """The sign carries the meaning: a small p alone cannot tell skill from anti-skill."""
    result = pesaran_timmermann(returns, -returns.to_numpy())
    assert result.statistic < 0
    assert result.p_value < 0.01


@pytest.mark.unit
def test_pt_matches_the_published_variance_formula() -> None:
    """Recompute Pesaran-Timmermann (1992) directly and compare."""
    rng = np.random.default_rng(9)
    y = pd.Series(rng.normal(0.0, 0.02, 300))
    pred = y.to_numpy() + rng.normal(0.0, 0.02, 300)
    up_y, up_x = (y.to_numpy() > 0).astype(float), (pred > 0).astype(float)
    n = 300
    p_y, p_x = up_y.mean(), up_x.mean()
    hit = float(np.mean(up_y == up_x))
    hit_indep = p_y * p_x + (1 - p_y) * (1 - p_x)
    var_hit = hit_indep * (1 - hit_indep) / n
    var_indep = (
        ((2 * p_y - 1) ** 2) * p_x * (1 - p_x) / n
        + ((2 * p_x - 1) ** 2) * p_y * (1 - p_y) / n
        + 4 * p_y * p_x * (1 - p_y) * (1 - p_x) / n**2
    )
    expected = (hit - hit_indep) / np.sqrt(var_hit - var_indep)
    assert pesaran_timmermann(y, pred).statistic == pytest.approx(expected, rel=1e-12)


@pytest.mark.unit
def test_sharpe_and_drawdown() -> None:
    r = pd.Series([0.02, 0.01, 0.015, 0.005] * 20)  # positive mean, non-zero variance
    assert sharpe_ratio(r, periods_per_year=252) > 0
    assert sharpe_ratio(pd.Series([0.0] * 5), 252) == 0.0  # never traded -> flat, not lucky
    assert np.isnan(sharpe_ratio(pd.Series([0.01] * 5), 252))  # constant gain -> undefined
    equity = pd.Series([1.0, 1.2, 0.9, 1.1])
    assert max_drawdown(equity) == pytest.approx(0.9 / 1.2 - 1.0)


@pytest.mark.unit
def test_max_drawdown_counts_a_loss_in_the_very_first_period() -> None:
    """Without the implied starting capital the opening loss is invisible."""
    equity = pd.Series([0.8, 0.9, 1.0])  # lost 20% before anything else happened
    assert max_drawdown(equity) == pytest.approx(-0.2)
    assert max_drawdown(equity, initial=None) == pytest.approx(0.0)


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
