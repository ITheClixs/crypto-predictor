"""Benchmark forecasters behave exactly as specified."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cryptoforecast.models.base import Forecaster
from cryptoforecast.models.baselines import (
    AR1Forecaster,
    HistoricalMeanForecaster,
    RandomWalkForecaster,
    benchmark_factories,
)


@pytest.fixture
def xy() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    ret = rng.normal(0.0, 0.02, 200)
    X = pd.DataFrame({"ret_1": ret})
    y = pd.Series(0.5 + 2.0 * ret + rng.normal(0.0, 1e-9, 200))  # near-exact linear
    return X, y


@pytest.mark.unit
def test_all_conform_to_protocol() -> None:
    for ctor in benchmark_factories().values():
        assert isinstance(ctor(), Forecaster)


@pytest.mark.unit
def test_random_walk_predicts_zero(xy: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = xy
    preds = RandomWalkForecaster().fit(X, y).predict(X)
    assert np.allclose(preds, 0.0)


@pytest.mark.unit
def test_historical_mean_is_constant_train_mean(xy: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = xy
    preds = HistoricalMeanForecaster().fit(X, y).predict(X)
    assert np.allclose(preds, y.mean())


@pytest.mark.unit
def test_ar1_recovers_linear_coefficients(xy: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = xy
    model = AR1Forecaster().fit(X, y)
    assert model._intercept == pytest.approx(0.5, abs=1e-3)
    assert model._slope == pytest.approx(2.0, abs=1e-3)
