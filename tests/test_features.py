"""Feature correctness, above all the no-lookahead guarantee."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cryptoforecast.features import FEATURE_COLUMNS, make_features


@pytest.mark.unit
def test_columns_and_order(synthetic_ohlcv: pd.DataFrame) -> None:
    feats = make_features(synthetic_ohlcv)
    assert tuple(feats.columns) == FEATURE_COLUMNS
    assert feats.index.equals(synthetic_ohlcv.index)


@pytest.mark.unit
def test_no_lookahead(synthetic_ohlcv: pd.DataFrame) -> None:
    """Perturbing *future* bars must not change any *past* feature value.

    This is the property that separates an honest forecast from an in-sample fit.
    """
    df = synthetic_ohlcv
    feats = make_features(df)

    cut = df.index[200]
    future = df.index > cut
    rng = np.random.default_rng(123)
    perturbed = df.copy()
    for col in ("Open", "High", "Low", "Close", "Volume"):
        perturbed.loc[future, col] *= rng.uniform(0.5, 1.5, size=int(future.sum()))

    feats_perturbed = make_features(perturbed)
    pd.testing.assert_frame_equal(feats.loc[:cut], feats_perturbed.loc[:cut])


@pytest.mark.unit
def test_scale_invariance(synthetic_ohlcv: pd.DataFrame) -> None:
    """Features are returns/ratios/z-scores, so rescaling price leaves them fixed."""
    df = synthetic_ohlcv
    scaled = df.copy()
    for col in ("Open", "High", "Low", "Close"):  # volume intentionally unchanged
        scaled[col] *= 1000.0

    pd.testing.assert_frame_equal(make_features(df), make_features(scaled), rtol=1e-6, atol=1e-9)


@pytest.mark.unit
def test_finite_after_warmup(synthetic_ohlcv: pd.DataFrame) -> None:
    feats = make_features(synthetic_ohlcv).dropna(how="any")
    assert len(feats) > 300
    assert np.isfinite(feats.to_numpy()).all()


@pytest.mark.unit
def test_missing_column_raises(synthetic_ohlcv: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="missing columns"):
        make_features(synthetic_ohlcv.drop(columns=["Volume"]))


@pytest.mark.unit
def test_unsorted_index_raises(synthetic_ohlcv: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="sorted ascending"):
        make_features(synthetic_ohlcv.iloc[::-1])
