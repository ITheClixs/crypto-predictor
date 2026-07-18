"""The model roster used by the study, plus which names are benchmarks."""

from __future__ import annotations

from .base import ModelFactory
from .baselines import AR1Forecaster, HistoricalMeanForecaster, RandomWalkForecaster
from .linear import ElasticNetForecaster, RidgeForecaster
from .trees import GBMForecaster

#: The primary null every model is measured against.
PRIMARY_BENCHMARK = "random_walk"

#: All benchmark names (non-ML reference points).
BENCHMARK_NAMES: tuple[str, ...] = ("random_walk", "historical_mean", "ar1")


def default_models() -> dict[str, ModelFactory]:
    """Ordered name -> factory map: benchmarks first, then the ML models."""
    return {
        "random_walk": RandomWalkForecaster,
        "historical_mean": HistoricalMeanForecaster,
        "ar1": AR1Forecaster,
        "ridge": RidgeForecaster,
        "elastic_net": ElasticNetForecaster,
        "gbm": GBMForecaster,
    }
