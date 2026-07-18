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
def test_regression_metrics_keys() -> None:
    yt = pd.Series([0.1, -0.1, 0.2])
    keys = set(regression_metrics(yt, np.array([0.1, -0.1, 0.2])))
    assert keys == {"rmse", "mae", "r2_oos", "dir_acc", "rank_ic"}
