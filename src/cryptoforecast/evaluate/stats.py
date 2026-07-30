"""Statistical tests that separate a real edge from a lucky sample.

A lower RMSE or a positive Sharpe means nothing without an error bar. This module
provides the standard toolkit for forecast and strategy evaluation:

- Diebold-Mariano (1995) with the Harvey-Leybourne-Newbold small-sample
  correction and a Newey-West long-run variance for overlapping h-step forecasts.
- Clark-West (2007), because DM is *not* valid when the models are nested. Under a
  nested null the larger model still estimates coefficients that are truly zero,
  which inflates its sample MSPE; DM reads that cost as evidence for the benchmark.
  Clark-West removes that term.

  **Read the caveat before using this against the zero-forecast benchmark.** The
  models in this package fit an unpenalized intercept, which equals the training-window
  mean of the target. Setting their slopes to zero therefore returns that mean, *not*
  zero. The zero forecast is the restriction "intercept and slopes are all zero", so
  Clark-West against it tests the joint martingale-difference null -- no drift *and* no
  conditional predictability -- rather than "the features add nothing beyond drift".
  With a nonzero drift in the data the two are materially different: the adjusted
  differential reduces to ``f_t = 2 (y_t - b_t)(m_t - b_t)``, which with ``b = 0``
  has expectation ``2 (E[y] E[m] + Cov(y, m))``, and the first term is nonzero
  whatever the features do. To test feature predictability, pass the recursively
  estimated mean as ``pred_bench``. Measured size at a nominal 5% under an estimated
  nested null: 14-22% against the zero forecast, 5-6% against the recursive mean at
  h=1. See ``audit/FORENSIC_REVIEW.md`` and ``audit/scripts/mc_null.py``.
- Pesaran-Timmermann (1992) market-timing / sign-predictability test.
- Sharpe ratio, max drawdown, and the Probabilistic Sharpe Ratio and Deflated
  Sharpe Ratio (Bailey & Lopez de Prado) which correct for short samples,
  non-normal returns, and the number of strategies tried.
- A circular block bootstrap for confidence intervals that respects serial
  dependence.

**Sign conventions** (deliberately spelled out, because a p-value without a direction is
worse than no p-value, because a model that is significantly *worse* than the
benchmark produces exactly the same small number as one that is better):

======================  ===========================================
statistic               a *better-than-benchmark* model gives ...
======================  ===========================================
:func:`diebold_mariano` a **negative** statistic (lower loss)
:func:`clark_west`      a **positive** statistic (lower adjusted MSPE)
:func:`pesaran_timmermann` a **positive** statistic (sign skill)
======================  ===========================================
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import stats

_EULER_MASCHERONI = 0.5772156649015329


class TestResult(NamedTuple):
    statistic: float
    p_value: float


def newey_west_lrv(x: np.ndarray, lags: int) -> float:
    """Bartlett-kernel long-run variance of ``x`` (Newey-West, bandwidth ``lags``).

    Every autocovariance is normalized by ``n`` rather than ``n - k``. That is not
    a typo: the Bartlett weights guarantee a non-negative estimate only under the
    ``1/n`` normalization, and mixing ``1/(n - k)`` into a weighted sum can push
    the estimate negative in small samples.
    """
    n = x.size
    mean = x.mean()
    centered = x - mean
    lrv = float(np.mean(centered**2))
    for k in range(1, lags + 1):
        cov = float(np.sum(centered[k:] * centered[:-k]) / n)
        lrv += 2.0 * (1.0 - k / (lags + 1)) * cov  # Bartlett weights
    return lrv


def _loss_differential(
    y_true: pd.Series, pred_model: np.ndarray, pred_bench: np.ndarray, loss: str
) -> np.ndarray:
    """``loss(model) - loss(bench)`` per observation, non-finite rows removed."""
    yt = np.asarray(y_true, dtype=float)
    e_model = yt - np.asarray(pred_model, dtype=float)
    e_bench = yt - np.asarray(pred_bench, dtype=float)
    if loss == "mse":
        d = e_model**2 - e_bench**2
    elif loss == "mae":
        d = np.abs(e_model) - np.abs(e_bench)
    else:
        raise ValueError("loss must be 'mse' or 'mae'")
    return d[np.isfinite(d)]


def diebold_mariano(
    y_true: pd.Series,
    pred_model: np.ndarray,
    pred_bench: np.ndarray,
    horizon: int = 1,
    loss: str = "mse",
) -> TestResult:
    """Test equal predictive accuracy of ``pred_model`` vs ``pred_bench``.

    The loss differential is ``loss(model) - loss(bench)``, so a *negative*
    statistic means the model has lower loss (is better). The long-run variance
    uses Newey-West with ``horizon - 1`` lags to handle overlapping forecasts, and
    the statistic gets the HLN small-sample correction and a two-sided Student-t
    p-value (the null is *equal* accuracy, so both tails matter).

    Use :func:`clark_west` instead when the benchmark is nested in the model.
    """
    d = _loss_differential(y_true, pred_model, pred_bench, loss)
    n = d.size
    if n < 8 or np.allclose(d, d[0]):
        return TestResult(float("nan"), float("nan"))

    d_bar = d.mean()
    lrv = newey_west_lrv(d, lags=max(0, horizon - 1))
    if lrv <= 0:
        return TestResult(float("nan"), float("nan"))

    dm = d_bar / math.sqrt(lrv / n)
    hln_arg = (n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n
    if hln_arg <= 0:  # sample too short for this horizon to apply the HLN correction
        return TestResult(float("nan"), float("nan"))
    dm_corrected = dm * math.sqrt(hln_arg)
    p_value = 2.0 * stats.t.cdf(-abs(dm_corrected), df=n - 1)
    return TestResult(float(dm_corrected), float(p_value))


def clark_west(
    y_true: pd.Series,
    pred_model: np.ndarray,
    pred_bench: np.ndarray,
    horizon: int = 1,
) -> TestResult:
    """MSPE-adjusted test of a model against a benchmark it *nests* (Clark-West 2007).

    Under the null that the extra coefficients are zero, the larger model still has to
    estimate them; the noise in those estimates raises its sample MSPE even though its
    population MSPE is identical. Diebold-Mariano reads that pure estimation noise as
    evidence *for* the benchmark. Clark-West subtracts it off:

    .. math::
        \\hat f_t = (y_t - \\hat y_{b,t})^2
                   - \\left[(y_t - \\hat y_{m,t})^2 - (\\hat y_{b,t} - \\hat y_{m,t})^2\\right]

    which is algebraically ``f_t = 2 (y_t - \\hat y_{b,t})(\\hat y_{m,t} - \\hat y_{b,t})``.
    The statistic is therefore a t-test of whether the model's *deviation from the
    benchmark* covaries positively with the outcome's -- which is why the choice of
    ``pred_bench`` defines the hypothesis, not merely the baseline. See the module
    docstring for what the zero-forecast benchmark tests and does not test.

    A **positive** statistic favours the model. The test is one-sided against a
    standard normal, as Clark and West recommend: the alternative of interest is
    "the model predicts", and their adjusted statistic is not asymptotically
    normal under a two-sided reading.

    ``horizon - 1`` Bartlett lags is the mechanical MA(h-1) bandwidth for overlapping
    h-step errors, and is *zero* lags at ``horizon = 1``. That is only defensible when
    ``f_t`` is serially uncorrelated, which fails once a persistent component enters
    through drift. Measured size under an estimated nested null is 10% at h=7 even
    against the recursive mean, and 28-43% when the resampling scheme preserves
    volatility clustering; dependence-robust inference on ``f_t`` is needed there.
    """
    yt = np.asarray(y_true, dtype=float)
    pm = np.asarray(pred_model, dtype=float)
    pb = np.asarray(pred_bench, dtype=float)
    f = (yt - pb) ** 2 - ((yt - pm) ** 2 - (pb - pm) ** 2)
    f = f[np.isfinite(f)]
    n = f.size
    if n < 8 or np.allclose(f, f[0]):
        return TestResult(float("nan"), float("nan"))

    lrv = newey_west_lrv(f, lags=max(0, horizon - 1))
    if lrv <= 0:
        return TestResult(float("nan"), float("nan"))

    cw = f.mean() / math.sqrt(lrv / n)
    p_value = 1.0 - stats.norm.cdf(cw)  # one-sided: large positive = model predicts
    return TestResult(float(cw), float(p_value))


def pesaran_timmermann(y_true: pd.Series, y_pred: np.ndarray) -> TestResult:
    """Test whether the sign of the forecast predicts the sign of the outcome.

    A **positive** statistic means the hit rate beats what independence between
    forecast sign and outcome sign would produce; a negative one means the
    forecast is systematically *anti*-predictive. The p-value is two-sided, so a
    small p on its own says only "not independent"; read it with the statistic.

    **Every variance term below divides by ``n``, so the caller must pass
    non-overlapping observations.** Feeding it all ``n`` rows of an ``h``-step forecast
    whose labels overlap ``h - 1`` times inflates the statistic by roughly
    ``sqrt(h)``. On this study's own forecasts that turned two ETH settings at h=7 from
    ``p = 0.001`` into ``p = 0.24-0.27`` once the labels were made non-overlapping
    (``audit/scripts/pt_and_exec.py``). Slice to ``y[k::h]`` for each phase ``k`` and
    aggregate across phases; do not pass the full overlapping series.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    n = yt.size
    if n < 8:
        return TestResult(float("nan"), float("nan"))

    up_true = (yt > 0).astype(float)
    up_pred = (yp > 0).astype(float)
    p_y, p_x = up_true.mean(), up_pred.mean()
    hit = float(np.mean(up_true == up_pred))
    hit_indep = p_y * p_x + (1 - p_y) * (1 - p_x)

    var_hit = hit_indep * (1 - hit_indep) / n
    var_indep = (
        ((2 * p_y - 1) ** 2) * p_x * (1 - p_x) / n
        + ((2 * p_x - 1) ** 2) * p_y * (1 - p_y) / n
        + 4 * p_y * p_x * (1 - p_y) * (1 - p_x) / n**2
    )
    denom = var_hit - var_indep
    if denom <= 0:  # degenerate when the model never changes its sign
        return TestResult(float("nan"), float("nan"))

    pt = (hit - hit_indep) / math.sqrt(denom)
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(pt)))
    return TestResult(float(pt), float(p_value))


def is_degenerate(returns: np.ndarray) -> bool:
    """True when a return series carries no usable variation.

    Testing ``std == 0`` exactly is not enough. Twenty copies of 0.01 have a
    sample standard deviation of ~1.8e-18 rather than 0, because 0.01 is not
    representable in binary; dividing by that produces a Sharpe of ~1e17 instead
    of an honest "undefined". The threshold is relative to the size of the
    returns, so it catches float residue while staying far below any real
    strategy's volatility.
    """
    if returns.size < 2:
        return True
    scale = max(1.0, float(np.abs(returns).max()))
    return bool(returns.std(ddof=1) <= 1e-12 * scale)


def holm_adjusted(p_values: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values: control the family-wise error rate.

    The strict correction. Testing 18 settings at 5% each is not a 5% test; it
    is roughly one expected false positive per study. Holm asks the demanding
    question: is *any* of these real? NaNs pass through unchanged.
    """
    p = np.asarray(p_values, dtype=float)
    finite = np.isfinite(p)
    out = np.full(p.shape, np.nan)
    values = p[finite]
    m = values.size
    if m == 0:
        return out

    order = np.argsort(values)
    scaled = (m - np.arange(m)) * values[order]
    adjusted = np.maximum.accumulate(scaled)  # step-down: monotone in ascending p
    result = np.empty(m)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    out[finite] = result
    return out


def benjamini_hochberg_adjusted(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values: control the false discovery rate.

    The lenient counterpart to Holm. It tolerates a known *fraction* of false
    positives among the rejections rather than forbidding them outright, which is
    the more useful question when screening many candidate signals.
    """
    p = np.asarray(p_values, dtype=float)
    finite = np.isfinite(p)
    out = np.full(p.shape, np.nan)
    values = p[finite]
    m = values.size
    if m == 0:
        return out

    order = np.argsort(values)
    ranks = np.arange(1, m + 1)
    scaled = values[order] * m / ranks
    adjusted = np.minimum.accumulate(scaled[::-1])[::-1]  # step-up, monotone
    result = np.empty(m)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    out[finite] = result
    return out


def sharpe_ratio(returns: pd.Series | np.ndarray, periods_per_year: float) -> float:
    """Annualized Sharpe ratio of a per-period return series (excess over zero).

    ``periods_per_year`` must match the sampling of ``returns``: for the
    non-overlapping ``h``-day schedule used here that is ``365 / h``, not 365.
    A flat, never-traded series (all zeros) scores 0; a constant *non-zero*
    series has an undefined Sharpe and returns NaN rather than a fake 0.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if is_degenerate(r):
        # A never-traded strategy is flat, not brilliant; a constant gain is undefined.
        return 0.0 if r.size == 0 or np.allclose(r, 0.0) else float("nan")
    return float(math.sqrt(periods_per_year) * r.mean() / r.std(ddof=1))


def max_drawdown(equity: pd.Series | np.ndarray, initial: float | None = 1.0) -> float:
    """Most negative peak-to-trough decline of an equity curve (a negative number).

    ``initial`` is prepended as the starting capital so that a loss in the very
    first period counts as drawdown. Without it the first value is its own running
    maximum and the opening loss is invisible. Pass ``None`` when ``equity``
    already contains its starting point.
    """
    e = np.asarray(equity, dtype=float)
    if e.size == 0:
        return 0.0
    if initial is not None:
        e = np.concatenate([[initial], e])
    running_max = np.maximum.accumulate(e)
    return float((e / running_max - 1.0).min())


def probabilistic_sharpe_ratio(returns: pd.Series | np.ndarray, benchmark_sr: float = 0.0) -> float:
    """P(true Sharpe > benchmark), correcting for sample size, skew, and kurtosis."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 8 or is_degenerate(r):
        return float("nan")
    sr = r.mean() / r.std(ddof=1)  # per-period, non-annualized
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))  # non-excess
    denom = math.sqrt(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2)
    if denom <= 0:
        return float("nan")
    return float(stats.norm.cdf((sr - benchmark_sr) * math.sqrt(n - 1) / denom))


def expected_max_sharpe(n_trials: int, trials_sr_std: float) -> float:
    """Expected maximum per-period Sharpe across ``n_trials`` independent strategies."""
    if n_trials < 2 or trials_sr_std <= 0:
        return 0.0
    inv = stats.norm.ppf
    term1 = (1 - _EULER_MASCHERONI) * inv(1 - 1.0 / n_trials)
    term2 = _EULER_MASCHERONI * inv(1 - 1.0 / (n_trials * math.e))
    return float(trials_sr_std * (term1 + term2))


def deflated_sharpe_ratio(
    returns: pd.Series | np.ndarray, n_trials: int, trials_sr_std: float
) -> float:
    """PSR against the expected-best-of-N-trials benchmark (guards against selection)."""
    benchmark = expected_max_sharpe(n_trials, trials_sr_std)
    return probabilistic_sharpe_ratio(returns, benchmark_sr=benchmark)


def block_bootstrap_ci(
    returns: pd.Series | np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    n_boot: int = 2000,
    block: int | None = None,
    alpha: float = 0.05,
    seed: int = 7,
) -> tuple[float, float]:
    """Circular block-bootstrap confidence interval for a serially-dependent stat."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 8:
        return (float("nan"), float("nan"))
    block = block or max(1, round(n ** (1 / 3)))
    n_blocks = math.ceil(n / block)
    rng = np.random.default_rng(seed)

    estimates = np.empty(n_boot)
    offsets = np.arange(block)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        estimates[b] = stat_fn(r[idx])
    lo, hi = np.percentile(estimates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))
