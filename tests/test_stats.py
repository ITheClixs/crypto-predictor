"""Statistical tests behave correctly on constructed signals."""

from __future__ import annotations

import math

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
    is_degenerate,
    max_drawdown,
    newey_west_lrv,
    pesaran_timmermann,
    pesaran_timmermann_non_overlapping,
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
def test_clark_west_is_positive_when_the_forecast_covaries_with_the_outcome(
    returns: pd.Series,
) -> None:
    """CW reduces to a t-test on ``2 (y - b)(m - b)``, so it is positive iff the
    forecast's deviation from the benchmark covaries positively with the outcome's."""
    signal = returns.to_numpy() * 0.5
    bench = np.zeros(len(returns))
    result = clark_west(returns, signal, bench)
    assert result.statistic > 0  # positive favours the model
    assert result.p_value < 0.01  # one-sided


@pytest.mark.unit
def test_exogenous_noise_forecast_is_genuinely_worse_and_dm_detects_it() -> None:
    """What an independent-noise forecast actually shows, which is not what we once claimed.

    An earlier version of this test drew ``y ~ N(0, 0.03)`` and an independent
    ``u ~ N(0, 0.01)`` as the larger model's forecast against a zero benchmark, and
    asserted that the null of equal predictive accuracy "holds by construction". It
    does not:

        MSPE_bench = E[(y - 0)^2] = sy^2
        MSPE_model = E[(y - u)^2] = sy^2 + su^2 > sy^2

    The population MSPEs differ by ``su^2``, so the null is *false* by construction and
    DM's high rejection rate is power, not size distortion. Nor does the setup exercise
    the mechanism Clark-West exists to correct: ``u`` is exogenous noise, not an
    estimated forecast, so no coefficient is estimated anywhere and the
    estimation-noise term is absent. Finally, CW appears centered here for an
    algebraic reason that carries no information about its size: the adjusted
    differential reduces to ``f_t = 2 (y_t - b_t)(m_t - b_t)``, which with ``b = 0`` is
    ``2 y_t u_t``, and ``E[2 y u] = 0`` for *any* independent ``u`` however bad a
    forecast it is.

    So this test now asserts only what the construction supports. Size and power of
    either statistic under a nested-*estimation* null require estimated forecasts; see
    ``audit/scripts/mc_null.py``.
    """
    rng = np.random.default_rng(4)
    sy, su, n = 0.03, 0.01, 250
    gaps, dm_stats, cw_stats = [], [], []
    for _ in range(200):
        y = pd.Series(rng.normal(0.0, sy, n))
        useless = rng.normal(0.0, su, n)  # independent of y
        bench = np.zeros(n)
        yt = y.to_numpy()
        gaps.append(np.mean((yt - useless) ** 2) - np.mean((yt - bench) ** 2))
        dm_stats.append(diebold_mariano(y, useless, bench).statistic)
        cw_stats.append(clark_west(y, useless, bench).statistic)

    # The model is worse in population by exactly su^2, and the sample gap recovers it.
    assert np.mean(gaps) == pytest.approx(su**2, rel=0.15)
    # DM therefore has power against a false null. This is not evidence of size distortion.
    assert np.mean(dm_stats) > 1.0
    # CW centres by the algebraic cancellation above, not by holding size.
    assert abs(np.mean(cw_stats)) < 0.2


@pytest.mark.unit
def test_pesaran_timmermann_is_inflated_by_overlapping_labels() -> None:
    """Feeding overlapping h-step labels to a test that divides by ``n`` inflates it.

    Each observation is repeated ``h`` times here, which adds no information but
    multiplies the nominal sample size by ``h``. The statistic grows by roughly
    ``sqrt(h)`` while the underlying hit rate is unchanged. This is the defect behind
    the two ``p = 0.001`` sign results in the original write-up; on non-overlapping
    subsamples they are ``p = 0.24-0.27`` (``audit/scripts/pt_and_exec.py``).
    """
    rng = np.random.default_rng(3)
    h, n = 7, 300
    y = rng.normal(0.0, 0.03, n)
    pred = -0.3 * y + rng.normal(0.0, 0.03, n)  # mildly anti-predictive

    independent = pesaran_timmermann(pd.Series(y), pred)
    overlapping = pesaran_timmermann(pd.Series(np.repeat(y, h)), np.repeat(pred, h))

    assert abs(overlapping.statistic) == pytest.approx(
        abs(independent.statistic) * math.sqrt(h), rel=0.05
    )
    # Same hit rate, no extra information -- yet only the overlapping version "rejects".
    assert independent.p_value > 0.05
    assert overlapping.p_value < 0.01


@pytest.mark.unit
def test_pesaran_timmermann_non_overlapping_averages_the_phases() -> None:
    """The phase-averaged wrapper must equal the mean over the ``h`` disjoint subsamples.

    ``audit/scripts/pt_and_exec.py`` produced the corrected sign-timing numbers this
    way, and the manuscript quotes them; the pipeline must compute the same thing
    rather than a second, subtly different aggregation.
    """
    rng = np.random.default_rng(11)
    h, n = 7, 700
    y = rng.normal(0.0, 0.03, n)
    pred = -0.3 * y + rng.normal(0.0, 0.03, n)

    phases = [pesaran_timmermann(pd.Series(y[k::h]), pred[k::h]) for k in range(h)]
    combined = pesaran_timmermann_non_overlapping(pd.Series(y), pred, horizon=h)

    assert combined.statistic == pytest.approx(np.mean([p.statistic for p in phases]))
    assert combined.p_value == pytest.approx(np.mean([p.p_value for p in phases]))


@pytest.mark.unit
def test_pesaran_timmermann_non_overlapping_is_the_plain_test_at_h1() -> None:
    """With one phase there is nothing to average, so the two must agree exactly."""
    rng = np.random.default_rng(12)
    y = rng.normal(0.0, 0.03, 400)
    pred = 0.2 * y + rng.normal(0.0, 0.03, 400)

    plain = pesaran_timmermann(pd.Series(y), pred)
    combined = pesaran_timmermann_non_overlapping(pd.Series(y), pred, horizon=1)

    assert combined.statistic == pytest.approx(plain.statistic)
    assert combined.p_value == pytest.approx(plain.p_value)


@pytest.mark.unit
def test_pesaran_timmermann_non_overlapping_deflates_the_overlap_artifact() -> None:
    """The whole point: the overlapping statistic rejects, the phase-averaged one does not."""
    rng = np.random.default_rng(13)
    h, n = 7, 300
    y = rng.normal(0.0, 0.03, n)
    pred = -0.3 * y + rng.normal(0.0, 0.03, n)
    y_ovlp, p_ovlp = np.repeat(y, h), np.repeat(pred, h)

    overlapping = pesaran_timmermann(pd.Series(y_ovlp), p_ovlp)
    corrected = pesaran_timmermann_non_overlapping(pd.Series(y_ovlp), p_ovlp, horizon=h)

    assert overlapping.p_value < 0.01
    assert abs(corrected.statistic) < abs(overlapping.statistic)


@pytest.mark.unit
def test_clark_west_identical_forecasts_is_nan(returns: pd.Series) -> None:
    same = np.zeros(len(returns))
    assert np.isnan(clark_west(returns, same, same).statistic)


@pytest.mark.unit
def test_dm_supports_absolute_error_loss(returns: pd.Series) -> None:
    model = returns.to_numpy() * 0.5
    bench = np.zeros(len(returns))
    assert diebold_mariano(returns, model, bench, loss="mae").statistic < 0


@pytest.mark.unit
def test_dm_rejects_an_unknown_loss(returns: pd.Series) -> None:
    with pytest.raises(ValueError, match="loss must be"):
        diebold_mariano(returns, returns.to_numpy(), np.zeros(len(returns)), loss="huber")


@pytest.mark.unit
@pytest.mark.parametrize("test", [diebold_mariano, clark_west])
def test_tests_refuse_to_report_on_a_sample_too_short_to_mean_anything(test) -> None:
    y = pd.Series([0.01, -0.02, 0.03, -0.01])  # 4 points
    result = test(y, y.to_numpy(), np.zeros(4))
    assert np.isnan(result.statistic) and np.isnan(result.p_value)


@pytest.mark.unit
def test_pt_and_bootstrap_and_psr_refuse_a_sample_too_short() -> None:
    short = pd.Series([0.01, -0.02, 0.03, -0.01])
    assert np.isnan(pesaran_timmermann(short, short.to_numpy()).statistic)
    assert np.isnan(probabilistic_sharpe_ratio(short))
    assert all(np.isnan(v) for v in block_bootstrap_ci(short, lambda a: float(a.mean())))


@pytest.mark.unit
def test_degenerate_inputs_return_neutral_values_rather_than_raising() -> None:
    assert sharpe_ratio(pd.Series([], dtype=float), 365) == 0.0
    assert max_drawdown(pd.Series([], dtype=float)) == 0.0
    assert expected_max_sharpe(1, 0.05) == 0.0  # a single trial is not a search
    assert expected_max_sharpe(20, 0.0) == 0.0  # identical trials cannot be deflated
    assert np.isnan(probabilistic_sharpe_ratio(pd.Series([0.01] * 20)))  # zero variance


@pytest.mark.unit
@pytest.mark.parametrize("value", [0.01, 0.1, 0.07, -0.03])
def test_a_constant_return_series_is_undefined_not_astronomically_good(value: float) -> None:
    """The `== 0` variance check these guards used to do is not float-safe.

    Twenty copies of 0.01 have a sample standard deviation of ~1.8e-18, not 0,
    because 0.01 has no exact binary representation. Dividing by that produced a
    Sharpe of ~1e17, a number that would sail straight into a results table.
    """
    r = pd.Series([value] * 20)
    assert r.std(ddof=1) != 0.0  # the residue the old guard missed
    assert is_degenerate(r.to_numpy())
    assert np.isnan(sharpe_ratio(r, 365))
    assert np.isnan(probabilistic_sharpe_ratio(r))


@pytest.mark.unit
def test_a_genuinely_low_volatility_series_is_not_treated_as_degenerate() -> None:
    """The tolerance must sit far below any real strategy's volatility."""
    rng = np.random.default_rng(21)
    r = rng.normal(0.0001, 1e-6, 200)  # tiny but real variation
    assert not is_degenerate(r)
    assert np.isfinite(sharpe_ratio(pd.Series(r), 365))


@pytest.mark.unit
def test_multiple_testing_on_an_empty_family_returns_nothing_to_reject() -> None:
    for adjust in (holm_adjusted, benjamini_hochberg_adjusted):
        assert adjust(np.array([])).size == 0
        assert np.isnan(adjust(np.array([np.nan, np.nan]))).all()


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
