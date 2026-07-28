"""The model roster used by the study, plus which names are benchmarks."""

from __future__ import annotations

from functools import partial

from .base import ModelFactory
from .baselines import AR1Forecaster, HistoricalMeanForecaster, RandomWalkForecaster
from .linear import ElasticNetForecaster, RidgeForecaster
from .trees import GBMForecaster

#: The primary null every model is measured against.
PRIMARY_BENCHMARK = "random_walk"

#: All benchmark names (non-ML reference points).
BENCHMARK_NAMES: tuple[str, ...] = ("random_walk", "historical_mean", "ar1")

#: Names of the models that are *not* benchmarks — the ones the study is testing.
ML_NAMES: tuple[str, ...] = ("ridge", "elastic_net", "gbm")


def default_models(horizon: int = 1) -> dict[str, ModelFactory]:
    """Ordered name -> factory map: benchmarks first, then the ML models.

    ``horizon`` is passed to models that hold out their own validation slice, so
    they can purge the same number of overlapping labels the outer walk-forward
    purges. Models without an internal holdout ignore it.
    """
    return {
        "random_walk": RandomWalkForecaster,
        "historical_mean": HistoricalMeanForecaster,
        "ar1": AR1Forecaster,
        "ridge": RidgeForecaster,
        "elastic_net": ElasticNetForecaster,
        "gbm": partial(GBMForecaster, purge=horizon),
    }
