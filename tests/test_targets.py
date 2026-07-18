"""Forward-target correctness and boundary behavior."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cryptoforecast.targets import (
    forward_direction,
    forward_log_return,
    forward_simple_return,
    make_target,
)


@pytest.fixture
def close() -> pd.Series:
    index = pd.date_range("2021-01-01", periods=6, freq="D")
    return pd.Series([100.0, 110.0, 121.0, 133.1, 120.0, 132.0], index=index)


@pytest.mark.unit
def test_simple_return_value_and_tail(close: pd.Series) -> None:
    fwd = forward_simple_return(close, 1)
    assert fwd.iloc[0] == pytest.approx(0.10)
    assert np.isnan(fwd.iloc[-1])


@pytest.mark.unit
def test_trailing_nan_count_equals_horizon(close: pd.Series) -> None:
    for horizon in (1, 2, 3):
        assert int(forward_simple_return(close, horizon).isna().sum()) == horizon


@pytest.mark.unit
def test_log_return_matches_definition(close: pd.Series) -> None:
    fwd = forward_log_return(close, 2)
    assert fwd.iloc[0] == pytest.approx(np.log(121.0 / 100.0))


@pytest.mark.unit
def test_direction_encoding(close: pd.Series) -> None:
    direction = forward_direction(close, 1)
    assert direction.iloc[0] == 1.0  # 100 -> 110 up
    assert direction.iloc[3] == 0.0  # 133.1 -> 120 down
    assert np.isnan(direction.iloc[-1])


@pytest.mark.unit
def test_make_target_dispatch_and_errors(close: pd.Series) -> None:
    pd.testing.assert_series_equal(make_target(close, 1, "simple"), forward_simple_return(close, 1))
    with pytest.raises(ValueError, match="unknown target"):
        make_target(close, 1, "nope")
    with pytest.raises(ValueError, match="horizon must be"):
        forward_simple_return(close, 0)
