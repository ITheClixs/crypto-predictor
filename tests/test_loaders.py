"""Data loading: column normalization and cache round-trip (no network)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cryptoforecast.data.cache import cache_path, read_cache, request_key, write_cache
from cryptoforecast.data.loaders import load_ohlcv, normalize_ohlcv, to_yf_ticker


@pytest.mark.unit
def test_to_yf_ticker() -> None:
    assert to_yf_ticker("btc") == "BTC-USD"
    assert to_yf_ticker("ETH-USD") == "ETH-USD"


@pytest.mark.unit
def test_normalize_flattens_multiindex_columns() -> None:
    index = pd.date_range("2022-01-01", periods=3, freq="D")
    columns = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Volume"], ["BTC-USD"]],
        names=["Price", "Ticker"],
    )
    raw = pd.DataFrame(
        [[10, 11, 9, 10.5, 100], [11, 12, 10, 11.5, 110], [12, 13, 11, 12.5, 120]],
        index=index,
        columns=columns,
    )
    out = normalize_ohlcv(raw)
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert isinstance(out["Close"], pd.Series)


@pytest.mark.unit
def test_normalize_rejects_non_positive_prices() -> None:
    index = pd.date_range("2022-01-01", periods=2, freq="D")
    raw = pd.DataFrame(
        {
            "Open": [1.0, 1.0],
            "High": [1.0, 1.0],
            "Low": [0.0, 1.0],
            "Close": [1.0, 1.0],
            "Volume": [1.0, 1.0],
        },
        index=index,
    )
    with pytest.raises(ValueError, match="non-positive prices"):
        normalize_ohlcv(raw)


@pytest.mark.unit
def test_cache_round_trip(tmp_path: Path) -> None:
    key = request_key(ticker="BTC-USD", start="2022-01-01", end=None, interval="1d")
    path = cache_path(key, tmp_path)
    df = pd.DataFrame({"Close": [1.0, 2.0]}, index=pd.date_range("2022-01-01", periods=2))
    write_cache(df, path)
    pd.testing.assert_frame_equal(read_cache(path), df, check_freq=False)


@pytest.mark.unit
def test_load_ohlcv_uses_cache_without_network(tmp_path: Path) -> None:
    # Pre-seed the cache so load_ohlcv never imports/calls yfinance.
    key = request_key(ticker="BTC-USD", start="2022-01-01", end=None, interval="1d")
    seeded = pd.DataFrame(
        {c: [1.0, 2.0] for c in ("Open", "High", "Low", "Close", "Volume")},
        index=pd.date_range("2022-01-01", periods=2),
    )
    write_cache(seeded, cache_path(key, tmp_path))
    out = load_ohlcv("BTC", start="2022-01-01", cache_dir=tmp_path)
    pd.testing.assert_frame_equal(out, seeded, check_freq=False)


@pytest.mark.integration
def test_download_real_btc(tmp_path: Path) -> None:
    df = load_ohlcv("BTC", start="2024-01-01", end="2024-01-15", cache_dir=tmp_path)
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(df) > 5
    assert (df["Close"] > 0).all()
