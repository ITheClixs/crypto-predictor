"""The minimal forecaster interface shared by benchmarks and ML models."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class Forecaster(Protocol):
    """A point forecaster of the forward-return target.

    ``fit`` sees only in-sample (X, y) and returns ``self``; ``predict`` returns a
    1-D array of predicted forward returns aligned to ``X``'s rows.
    """

    name: str

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Forecaster: ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


ModelFactory = Callable[[], Forecaster]
