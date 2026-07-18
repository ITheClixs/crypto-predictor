"""ML forecasters: finite, deterministic, and able to recover a real signal."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cryptoforecast.models.base import Forecaster
from cryptoforecast.models.linear import ElasticNetForecaster, RidgeForecaster
from cryptoforecast.models.registry import (
    BENCHMARK_NAMES,
    PRIMARY_BENCHMARK,
    default_models,
)
from cryptoforecast.models.trees import GBMForecaster


@pytest.fixture
def signal_xy() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(1)
    n = 400
    f0 = rng.normal(0, 1, n)
    f1 = rng.normal(0, 1, n)
    X = pd.DataFrame({"ret_1": f0, "vol_10": f1, "rsi_14": rng.normal(0, 1, n)})
    y = pd.Series(0.4 * f0 - 0.2 * f1 + rng.normal(0, 0.5, n))
    return X, y


@pytest.mark.unit
@pytest.mark.parametrize("ctor", [RidgeForecaster, ElasticNetForecaster])
def test_linear_recovers_signal(ctor: type, signal_xy: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = signal_xy
    model = ctor().fit(X, y)
    preds = model.predict(X)
    assert np.isfinite(preds).all()
    assert np.corrcoef(preds, y)[0, 1] > 0.5


@pytest.mark.unit
def test_gbm_is_deterministic_and_finite(signal_xy: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = signal_xy
    fast = {"n_estimators": 60, "early_stopping_rounds": 10}
    p1 = GBMForecaster(**fast).fit(X, y).predict(X)
    p2 = GBMForecaster(**fast).fit(X, y).predict(X)
    assert np.isfinite(p1).all()
    assert np.array_equal(p1, p2)
    assert np.corrcoef(p1, y)[0, 1] > 0.5


@pytest.mark.unit
def test_gbm_small_data_fallback() -> None:
    X = pd.DataFrame({"ret_1": np.linspace(-1, 1, 20)})
    y = pd.Series(np.linspace(-1, 1, 20))
    preds = GBMForecaster(n_estimators=20).fit(X, y).predict(X)
    assert np.isfinite(preds).all()


@pytest.mark.unit
def test_registry_wiring() -> None:
    models = default_models()
    assert PRIMARY_BENCHMARK in models
    assert set(BENCHMARK_NAMES) <= set(models)
    for factory in models.values():
        assert isinstance(factory(), Forecaster)
