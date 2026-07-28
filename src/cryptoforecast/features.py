"""Leak-free feature engineering.

Every feature at row ``t`` is a function of OHLCV data on ``[..., t]`` only, and
never of a future bar. That property is what makes the downstream backtest an
honest forecast rather than an in-sample fit, and it is enforced by
``tests/test_features.py::test_no_lookahead`` (perturbing future bars must not
change any past feature value).

Features are also constructed to be scale-free (returns, ratios, z-scores) so a
single model can pool across assets trading at very different price levels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS: tuple[str, ...] = (
    "ret_1",
    "ret_5",
    "ret_10",
    "ret_21",
    "vol_10",
    "vol_21",
    "rsi_14",
    "macd_hist",
    "sma_ratio_7_21",
    "dist_sma_50",
    "range_14",
    "volume_z_21",
)


def _log_return(close: pd.Series, window: int = 1) -> pd.Series:
    """Cumulative log return over ``window`` bars: log(C_t / C_{t-window})."""
    return np.log(close / close.shift(window))


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI, centered to roughly [-1, 1] as (RSI - 50) / 50."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return (rsi - 50.0) / 50.0


def _macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD histogram normalized by price so it is comparable across assets."""
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=slow).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False, min_periods=slow + signal).mean()
    return (macd - sig) / close


def make_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Build the scale-free feature matrix from an OHLCV frame.

    Parameters
    ----------
    ohlcv:
        DataFrame indexed by date with columns ``Open, High, Low, Close, Volume``,
        sorted ascending in time.

    Returns
    -------
    DataFrame with :data:`FEATURE_COLUMNS`, same index as the input. Warmup rows
    that lack enough history are NaN and are meant to be dropped by the caller
    (see :func:`cryptoforecast.dataset.build_supervised`).
    """
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(ohlcv.columns)
    if missing:
        raise ValueError(f"OHLCV frame missing columns: {sorted(missing)}")
    if not ohlcv.index.is_monotonic_increasing:
        raise ValueError("OHLCV index must be sorted ascending in time")

    close = ohlcv["Close"].astype(float)
    high = ohlcv["High"].astype(float)
    low = ohlcv["Low"].astype(float)
    volume = ohlcv["Volume"].astype(float)

    daily_ret = _log_return(close, 1)
    sma_7 = close.rolling(7).mean()
    sma_21 = close.rolling(21).mean()
    sma_50 = close.rolling(50).mean()

    log_volume = np.log1p(volume)
    vol_mean = log_volume.rolling(21).mean()
    vol_std = log_volume.rolling(21).std()

    feats = pd.DataFrame(index=ohlcv.index)
    feats["ret_1"] = daily_ret
    feats["ret_5"] = _log_return(close, 5)
    feats["ret_10"] = _log_return(close, 10)
    feats["ret_21"] = _log_return(close, 21)
    feats["vol_10"] = daily_ret.rolling(10).std()
    feats["vol_21"] = daily_ret.rolling(21).std()
    feats["rsi_14"] = _rsi(close, 14)
    feats["macd_hist"] = _macd_hist(close)
    feats["sma_ratio_7_21"] = np.log(sma_7 / sma_21)
    feats["dist_sma_50"] = np.log(close / sma_50)
    feats["range_14"] = ((high - low) / close).rolling(14).mean()
    feats["volume_z_21"] = (log_volume - vol_mean) / vol_std

    feats = feats.replace([np.inf, -np.inf], np.nan)
    return feats[list(FEATURE_COLUMNS)]
