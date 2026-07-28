"""Benchmark forecasters: the bar every ML model must clear to be interesting.

- ``RandomWalkForecaster``: predicts zero forward return. Under the efficient-market
  or martingale hypothesis this is the honest null, and it is notoriously hard to
  beat out of sample.
- ``HistoricalMeanForecaster``: predicts the in-sample average forward return
  (unconditional drift).
- ``AR1Forecaster``: OLS of the forward return on the most recent daily return.
  The simplest conditional (momentum/reversal) model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class RandomWalkForecaster:
    name = "random_walk"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RandomWalkForecaster:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X), dtype=float)


class HistoricalMeanForecaster:
    name = "historical_mean"

    def __init__(self) -> None:
        self._mean = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> HistoricalMeanForecaster:
        self._mean = float(np.mean(y.to_numpy()))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self._mean, dtype=float)


class AR1Forecaster:
    name = "ar1"

    def __init__(self, lag_col: str = "ret_1") -> None:
        self.lag_col = lag_col
        self._intercept = 0.0
        self._slope = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> AR1Forecaster:
        x = X[self.lag_col].to_numpy(dtype=float)
        design = np.column_stack([np.ones_like(x), x])
        coef, *_ = np.linalg.lstsq(design, y.to_numpy(dtype=float), rcond=None)
        self._intercept, self._slope = float(coef[0]), float(coef[1])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        x = X[self.lag_col].to_numpy(dtype=float)
        return self._intercept + self._slope * x


def benchmark_factories() -> dict[str, type]:
    """Name -> zero-arg constructor for the standard benchmark set."""
    return {
        RandomWalkForecaster.name: RandomWalkForecaster,
        HistoricalMeanForecaster.name: HistoricalMeanForecaster,
        AR1Forecaster.name: AR1Forecaster,
    }
