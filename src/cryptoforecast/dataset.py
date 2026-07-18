"""Assemble aligned (features, target) supervised datasets.

Glues :mod:`cryptoforecast.features` and :mod:`cryptoforecast.targets` together on
a single index. Warmup rows (any NaN feature) are dropped. Trailing rows whose
label is not yet observable are *kept* — they carry valid features and no target,
which is precisely the state a live forecaster is in on the most recent bar.
"""

from __future__ import annotations

from typing import NamedTuple

import pandas as pd

from .features import make_features
from .targets import make_target


class Dataset(NamedTuple):
    """Time-aligned supervised learning data for one asset/horizon."""

    X: pd.DataFrame  # features known at time t
    y: pd.Series  # forward return over (t, t+h]; NaN on the trailing h rows
    close: pd.Series  # close price at time t, same index as X

    @property
    def labeled(self) -> Dataset:
        """The subset with an observable target (safe for fitting/scoring)."""
        mask = self.y.notna()
        return Dataset(self.X.loc[mask], self.y.loc[mask], self.close.loc[mask])

    def __len__(self) -> int:
        return len(self.X)


def build_supervised(ohlcv: pd.DataFrame, horizon: int, target: str = "logret") -> Dataset:
    """Build features and a forward target, aligned and warmup-trimmed."""
    features = make_features(ohlcv)
    close = ohlcv["Close"].astype(float)
    y = make_target(close, horizon, kind=target)

    finite_features = features.dropna(how="any")
    index = finite_features.index
    return Dataset(
        X=finite_features,
        y=y.reindex(index),
        close=close.reindex(index),
    )
