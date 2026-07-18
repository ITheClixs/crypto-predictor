"""Dataset alignment: features, target, and price share one index."""

from __future__ import annotations

import pandas as pd
import pytest

from cryptoforecast.dataset import build_supervised


@pytest.mark.unit
def test_alignment_and_no_feature_nans(synthetic_ohlcv: pd.DataFrame) -> None:
    ds = build_supervised(synthetic_ohlcv, horizon=1)
    assert ds.X.index.equals(ds.y.index)
    assert ds.X.index.equals(ds.close.index)
    assert not ds.X.isna().to_numpy().any()


@pytest.mark.unit
def test_labeled_drops_exactly_the_horizon_tail(synthetic_ohlcv: pd.DataFrame) -> None:
    horizon = 7
    ds = build_supervised(synthetic_ohlcv, horizon=horizon)
    # Warmup is trimmed at the front, so the only unlabeled rows are the tail.
    assert len(ds) - len(ds.labeled) == horizon
    assert ds.labeled.y.notna().all()


@pytest.mark.unit
def test_close_matches_source(synthetic_ohlcv: pd.DataFrame) -> None:
    ds = build_supervised(synthetic_ohlcv, horizon=1)
    expected = synthetic_ohlcv["Close"].reindex(ds.close.index)
    pd.testing.assert_series_equal(ds.close, expected, check_names=False)
