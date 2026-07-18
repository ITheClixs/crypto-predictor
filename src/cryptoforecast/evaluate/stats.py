"""Statistical tests that separate a real edge from a lucky sample.

A lower RMSE or a positive Sharpe means nothing without an error bar. This module
provides the standard toolkit for forecast and strategy evaluation:

- Diebold-Mariano (1995) with the Harvey-Leybourne-Newbold small-sample
  correction and a Newey-West long-run variance for overlapping h-step forecasts.
- Pesaran-Timmermann (1992) market-timing / sign-predictability test.
- Sharpe ratio, max drawdown, and the Probabilistic Sharpe Ratio and Deflated
  Sharpe Ratio (Bailey & Lopez de Prado) which correct for short samples,
  non-normal returns, and the number of strategies tried.
- A circular block bootstrap for confidence intervals that respects serial
  dependence.
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
    the statistic gets the HLN small-sample correction and a Student-t p-value.
    """
    yt = np.asarray(y_true, dtype=float)
    e_model = yt - np.asarray(pred_model, dtype=float)
    e_bench = yt - np.asarray(pred_bench, dtype=float)
    if loss == "mse":
        d = e_model**2 - e_bench**2
    elif loss == "mae":
        d = np.abs(e_model) - np.abs(e_bench)
    else:
        raise ValueError("loss must be 'mse' or 'mae'")

    d = d[np.isfinite(d)]
    n = d.size
    if n < 8 or np.allclose(d, d[0]):
        return TestResult(float("nan"), float("nan"))

    d_bar = d.mean()
    gamma0 = np.mean((d - d_bar) ** 2)
    lags = max(0, horizon - 1)
    lrv = gamma0
    for k in range(1, lags + 1):
        cov = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        lrv += 2.0 * (1.0 - k / (lags + 1)) * cov  # Bartlett weights
    if lrv <= 0:
        return TestResult(float("nan"), float("nan"))

    dm = d_bar / math.sqrt(lrv / n)
    hln = math.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    dm_corrected = dm * hln
    p_value = 2.0 * stats.t.cdf(-abs(dm_corrected), df=n - 1)
    return TestResult(float(dm_corrected), float(p_value))


def pesaran_timmermann(y_true: pd.Series, y_pred: np.ndarray) -> TestResult:
    """Test whether the sign of the forecast predicts the sign of the outcome."""
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


def sharpe_ratio(returns: pd.Series | np.ndarray, periods_per_year: float) -> float:
    """Annualized Sharpe ratio of a per-period return series (excess over zero)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    sd = r.std(ddof=1) if r.size > 1 else 0.0
    if sd == 0:
        return 0.0
    return float(math.sqrt(periods_per_year) * r.mean() / sd)


def max_drawdown(equity: pd.Series | np.ndarray) -> float:
    """Most negative peak-to-trough decline of an equity curve (a negative number)."""
    e = np.asarray(equity, dtype=float)
    if e.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(e)
    return float((e / running_max - 1.0).min())


def probabilistic_sharpe_ratio(returns: pd.Series | np.ndarray, benchmark_sr: float = 0.0) -> float:
    """P(true Sharpe > benchmark), correcting for sample size, skew, and kurtosis."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 8 or r.std(ddof=1) == 0:
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
