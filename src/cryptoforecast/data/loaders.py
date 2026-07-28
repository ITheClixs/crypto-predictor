"""Load clean daily OHLCV bars from Yahoo Finance, with a local cache.

The only external data dependency in the study. Everything downstream consumes
the validated frame this module returns: columns ``Open, High, Low, Close,
Volume``, a sorted unique ``DatetimeIndex``, and strictly positive prices.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .cache import DEFAULT_CACHE_DIR, cache_path, read_cache, request_key, write_cache

OHLCV_COLUMNS: tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume")


def to_yf_ticker(symbol: str) -> str:
    """Map a bare crypto symbol to a Yahoo Finance USD pair (``BTC`` -> ``BTC-USD``)."""
    symbol = symbol.strip().upper()
    return symbol if "-" in symbol else f"{symbol}-USD"


def normalize_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance's (Price, Ticker) columns and validate the frame."""
    data = raw.copy()

    if getattr(data.columns, "nlevels", 1) > 1:
        for level in range(data.columns.nlevels):
            level_values = list(data.columns.get_level_values(level))
            if all(col in level_values for col in OHLCV_COLUMNS):
                data.columns = level_values
                break

    data = data.loc[:, ~data.columns.duplicated()]
    missing = [c for c in OHLCV_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(f"market data missing required columns: {missing}")

    data = data[list(OHLCV_COLUMNS)].astype(float)
    data = data[~data.index.duplicated(keep="last")].sort_index()
    data = data.dropna(subset=["Close"])
    if (data[["Open", "High", "Low", "Close"]] <= 0).to_numpy().any():
        raise ValueError("market data contains non-positive prices")
    if data.empty:
        raise ValueError("market data is empty after cleaning")
    return data


def load_ohlcv(
    symbol: str,
    start: str,
    end: str | None = None,
    interval: str = "1d",
    cache_dir: Path = DEFAULT_CACHE_DIR,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Return validated daily OHLCV for ``symbol``, downloading only on a cache miss."""
    ticker = to_yf_ticker(symbol)
    key = request_key(ticker=ticker, start=start, end=end, interval=interval)
    path = cache_path(key, cache_dir)

    if use_cache:
        cached = read_cache(path)
        if cached is not None:
            return cached

    import yfinance as yf  # imported lazily so unit tests need no network

    raw = yf.download(
        ticker, start=start, end=end, interval=interval, auto_adjust=True, progress=False
    )
    if raw is None or raw.empty:
        raise ValueError(f"no data returned for {ticker!r}; check the symbol and date range")

    data = normalize_ohlcv(raw)
    if use_cache:
        write_cache(data, path)
    return data
