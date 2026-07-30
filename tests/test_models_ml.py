"""ML forecasters: finite, deterministic, and able to recover a real signal."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cryptoforecast.models.base import Forecaster
from cryptoforecast.models.linear import ElasticNetForecaster, RidgeForecaster
from cryptoforecast.models.registry import (
    BENCHMARK_NAMES,
    ML_NAMES,
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
def test_gbm_never_sees_the_rows_it_purges() -> None:
    """The purged gap must reach neither the fit slice nor the validation slice.

    Corrupting exactly those rows is the direct test: if the model is genuinely
    blind to them, its predictions are bit-identical either way. A leaky
    implementation would train or early-stop on the garbage and diverge.
    """
    rng = np.random.default_rng(3)
    n, purge = 300, 10
    f = rng.normal(0, 1, n)
    X = pd.DataFrame({"ret_1": f})
    y = pd.Series(0.5 * f + rng.normal(0, 0.5, n))

    val = int(n * 0.15)
    gap = slice(n - val - purge, n - val)  # the rows the purge is meant to drop
    corrupted_X = X.copy()
    corrupted_y = y.copy()
    corrupted_X.iloc[gap, 0] = 1e6
    corrupted_y.iloc[gap] = -1e6

    clean = GBMForecaster(n_estimators=40, purge=purge).fit(X, y).predict(X)
    dirty = GBMForecaster(n_estimators=40, purge=purge).fit(corrupted_X, corrupted_y).predict(X)
    np.testing.assert_array_equal(clean, dirty)

    # Sanity: without the purge those same rows *do* reach the model.
    leaky = GBMForecaster(n_estimators=40, purge=0).fit(corrupted_X, corrupted_y).predict(X)
    assert not np.array_equal(clean, leaky)


@pytest.mark.unit
def test_registry_wiring() -> None:
    models = default_models()
    assert PRIMARY_BENCHMARK in models
    assert set(BENCHMARK_NAMES) <= set(models)
    assert set(ML_NAMES) <= set(models)
    assert set(BENCHMARK_NAMES).isdisjoint(ML_NAMES)
    assert set(BENCHMARK_NAMES) | set(ML_NAMES) == set(models)
    for factory in models.values():
        assert isinstance(factory(), Forecaster)


@pytest.mark.unit
def test_registry_passes_the_horizon_to_models_with_an_internal_holdout() -> None:
    gbm = default_models(horizon=7)["gbm"]()
    assert gbm.purge == 7
    assert default_models()["gbm"]().purge == 1


@pytest.mark.unit
@pytest.mark.parametrize("ctor", [RidgeForecaster, ElasticNetForecaster])
def test_zeroing_the_slopes_returns_the_training_mean_not_zero(ctor: type) -> None:
    """The linear models do **not** nest the zero forecast, and this pins down why.

    Both fit an unpenalized intercept on standardized features, so the intercept is the
    training-window mean of the target and the slopes-zero restriction of the estimator
    is the historical-mean forecaster -- never the random walk. Clark-West against a
    zero benchmark therefore tests the joint null "no drift and no conditional
    predictability", not "the features add nothing beyond drift". See
    ``audit/FORENSIC_REVIEW.md`` Issue 1.
    """
    rng = np.random.default_rng(11)
    n, drift = 300, 0.002
    X = pd.DataFrame({c: rng.normal(0, 1, n) for c in ("ret_1", "vol_10", "rsi_14")})
    y = pd.Series(drift + rng.normal(0, 0.03, n))  # pure drift, no conditional signal

    estimator = ctor().fit(X, y)._pipe.named_steps["model"]
    assert estimator.fit_intercept is True
    assert float(estimator.intercept_) == pytest.approx(float(y.mean()), rel=1e-9)
    assert abs(float(estimator.intercept_)) > 1e-4  # emphatically not the zero forecast
