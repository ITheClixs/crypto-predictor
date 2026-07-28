"""The walk-forward engine produces genuinely out-of-sample predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cryptoforecast.backtest.engine import OOS_COLUMNS, walk_forward
from cryptoforecast.config import WalkForwardConfig
from cryptoforecast.dataset import build_supervised
from cryptoforecast.models.baselines import AR1Forecaster

WF = WalkForwardConfig(train_size=150, test_size=30, embargo=3, min_train=100, mode="expanding")


@pytest.mark.unit
def test_output_schema_and_coverage(synthetic_ohlcv: pd.DataFrame) -> None:
    ds = build_supervised(synthetic_ohlcv, horizon=1)
    oos = walk_forward(ds, AR1Forecaster, WF, horizon=1)
    assert list(oos.columns) == list(OOS_COLUMNS)
    assert oos.index.is_monotonic_increasing
    assert not oos.index.has_duplicates
    assert oos.index.isin(ds.labeled.X.index).all()


@pytest.mark.unit
def test_raises_when_too_short() -> None:
    short = build_supervised(_tiny_ohlcv(120), horizon=1)
    with pytest.raises(ValueError, match="not enough data"):
        walk_forward(short, AR1Forecaster, WF, horizon=1)


@pytest.mark.unit
def test_end_to_end_no_lookahead(synthetic_ohlcv: pd.DataFrame) -> None:
    """Perturbing future bars must not change any past OOS *prediction*.

    ``y_true`` is deliberately excluded: the realized label at date ``t`` is the
    return over ``(t, t+h]`` and so legitimately reads the (perturbed) future — that
    is the target, not a leak. The leak-free claim is about ``y_pred`` and the
    decision-time ``close``, both of which must be untouched for every past date.
    """
    ds = build_supervised(synthetic_ohlcv, horizon=1)
    oos = walk_forward(ds, AR1Forecaster, WF, horizon=1)

    cutoff = ds.labeled.X.index[250]
    perturbed = synthetic_ohlcv.copy()
    future = perturbed.index > cutoff
    rng = np.random.default_rng(7)
    for col in ("Open", "High", "Low", "Close", "Volume"):
        perturbed.loc[future, col] *= rng.uniform(0.5, 1.5, size=int(future.sum()))

    oos_perturbed = walk_forward(build_supervised(perturbed, horizon=1), AR1Forecaster, WF, 1)
    for col in ("y_pred", "close"):
        pd.testing.assert_series_equal(oos[col].loc[:cutoff], oos_perturbed[col].loc[:cutoff])


def _tiny_ohlcv(n: int) -> pd.DataFrame:
    index = pd.date_range("2021-01-01", periods=n, freq="D")
    close = pd.Series(np.linspace(100, 120, n), index=index)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": 1000.0,
        },
        index=index,
    )
