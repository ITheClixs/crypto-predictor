"""Forecast metrics compute the textbook quantities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cryptoforecast.evaluate.metrics import (
    directional_accuracy,
    mae,
    r2_oos,
    rank_ic,
    regression_metrics,
    rmse,
)


@pytest.mark.unit
def test_rmse_and_mae() -> None:
    yt = pd.Series([1.0, 2.0, 3.0])
    yp = np.array([1.0, 2.0, 5.0])
    assert mae(yt, yp) == pytest.approx(2.0 / 3.0)
    assert rmse(yt, yp) == pytest.approx(np.sqrt(4.0 / 3.0))


@pytest.mark.unit
def test_r2_oos_reference_points() -> None:
    yt = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert r2_oos(yt, yt.to_numpy()) == pytest.approx(1.0)  # perfect
    assert r2_oos(yt, np.full(4, yt.mean())) == pytest.approx(0.0)  # predicts the mean
    assert r2_oos(yt, np.full(4, 100.0)) < 0.0  # worse than the mean


@pytest.mark.unit
def test_directional_accuracy() -> None:
    yt = pd.Series([0.01, -0.02, 0.03, -0.04])
    yp = np.array([0.5, 0.5, -0.5, -0.5])  # right, wrong, wrong, right
    assert directional_accuracy(yt, yp) == pytest.approx(0.5)
    assert np.isnan(directional_accuracy(yt, np.zeros(4)))  # never takes a side


@pytest.mark.unit
def test_rank_ic_monotonic() -> None:
    yt = pd.Series(np.arange(20, dtype=float))
    assert rank_ic(yt, np.arange(20, dtype=float)) == pytest.approx(1.0)
    assert rank_ic(yt, np.arange(20, dtype=float)[::-1]) == pytest.approx(-1.0)


@pytest.mark.unit
def test_r2_oos_against_an_explicit_benchmark_forecast() -> None:
    """The Campbell-Thompson form: the denominator is the benchmark's error."""
    yt = pd.Series([1.0, 2.0, 3.0, 4.0])
    bench = np.array([1.5, 1.5, 3.5, 3.5])  # errors of -0.5, +0.5, -0.5, +0.5
    assert r2_oos(yt, bench, bench) == pytest.approx(0.0)  # benchmark scores 0 vs itself
    assert r2_oos(yt, yt.to_numpy(), bench) == pytest.approx(1.0)  # perfect forecast
    # SSE = 4 * 1.0 vs benchmark SSE = 4 * 0.25 -> 1 - 4/1 = -3.
    worse = yt.to_numpy() + 1.0
    assert r2_oos(yt, worse, bench) == pytest.approx(-3.0)


@pytest.mark.unit
def test_r2_oos_benchmark_choice_changes_the_answer() -> None:
    """The hindsight sample mean is an easier benchmark than an ex-ante forecast."""
    yt = pd.Series([0.01, -0.02, 0.03, 0.04, -0.01])
    pred = np.full(5, 0.01)
    drift = np.full(5, 0.03)  # a poor ex-ante drift estimate
    assert r2_oos(yt, pred, drift) > r2_oos(yt, pred)


@pytest.mark.unit
def test_r2_oos_ignores_rows_the_benchmark_cannot_score() -> None:
    yt = pd.Series([1.0, 2.0, 3.0])
    pred = np.array([1.0, 2.0, 3.0])
    bench = np.array([1.5, 1.5, np.nan])
    assert r2_oos(yt, pred, bench) == pytest.approx(1.0)  # NaN row dropped, not propagated


@pytest.mark.unit
def test_regression_metrics_keys() -> None:
    yt = pd.Series([0.1, -0.1, 0.2])
    keys = set(regression_metrics(yt, np.array([0.1, -0.1, 0.2])))
    assert keys == {"rmse", "mae", "r2_oos", "r2_vs_sample_mean", "dir_acc", "rank_ic"}
