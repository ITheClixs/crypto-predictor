"""Gradient-boosted trees (XGBoost) with a temporal early-stopping holdout.

The model is deliberately shrunk (shallow trees, low learning rate, subsampling,
L2 penalty) because return prediction is close to noise and boosting will happily
memorize it. Early stopping uses the *last* slice of the training window as
validation, never a random slice — a random split would let the model peek across
time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


class GBMForecaster:
    name = "gbm"

    def __init__(
        self,
        n_estimators: int = 400,
        max_depth: int = 3,
        learning_rate: float = 0.03,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        min_child_weight: float = 5.0,
        val_fraction: float = 0.15,
        early_stopping_rounds: int = 30,
        random_state: int = 7,
    ) -> None:
        self.params = {
            "objective": "reg:squarederror",
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_lambda": reg_lambda,
            "min_child_weight": min_child_weight,
            "random_state": random_state,
            "n_jobs": 1,
            "verbosity": 0,
        }
        self.val_fraction = val_fraction
        self.early_stopping_rounds = early_stopping_rounds
        self._model: XGBRegressor | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> GBMForecaster:
        xs = X.to_numpy(dtype=float)
        ys = y.to_numpy(dtype=float)
        n = len(xs)
        k = int(n * self.val_fraction)

        if k >= 10 and n - k >= 30:
            model = XGBRegressor(early_stopping_rounds=self.early_stopping_rounds, **self.params)
            model.fit(xs[:-k], ys[:-k], eval_set=[(xs[-k:], ys[-k:])], verbose=False)
        else:  # too little data to hold out a temporal validation slice
            model = XGBRegressor(**self.params)
            model.fit(xs, ys, verbose=False)

        self._model = model
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self._model is not None, "predict called before fit"
        return np.asarray(self._model.predict(X.to_numpy(dtype=float)), dtype=float)
