"""Regularized linear forecasters.

Standardization lives *inside* the fit so the scaler only ever sees training rows
Fitting a scaler on the full sample is a classic and easy-to-miss leak. Ridge
and elastic-net shrink coefficients, which matters because the features are
correlated and the signal-to-noise ratio of return prediction is tiny.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class RidgeForecaster:
    name = "ridge"

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self._pipe: Pipeline | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RidgeForecaster:
        self._pipe = Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=self.alpha))])
        self._pipe.fit(X.to_numpy(dtype=float), y.to_numpy(dtype=float))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self._pipe is not None, "predict called before fit"
        return np.asarray(self._pipe.predict(X.to_numpy(dtype=float)), dtype=float)


class ElasticNetForecaster:
    name = "elastic_net"

    def __init__(self, alpha: float = 1e-3, l1_ratio: float = 0.5) -> None:
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self._pipe: Pipeline | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> ElasticNetForecaster:
        self._pipe = Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=5000)),
            ]
        )
        self._pipe.fit(X.to_numpy(dtype=float), y.to_numpy(dtype=float))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self._pipe is not None, "predict called before fit"
        return np.asarray(self._pipe.predict(X.to_numpy(dtype=float)), dtype=float)
