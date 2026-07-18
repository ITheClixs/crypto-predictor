"""Shared fixtures. All synthetic so the unit suite never touches the network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_synthetic_ohlcv(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """A deterministic geometric-random-walk OHLCV frame with valid bar geometry."""
    rng = np.random.default_rng(seed)
    daily_ret = rng.normal(0.0008, 0.03, n)
    close = 100.0 * np.exp(np.cumsum(daily_ret))
    open_ = close * (1.0 + rng.normal(0.0, 0.004, n))
    hi_raw = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.008, n)))
    lo_raw = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.008, n)))
    volume = rng.lognormal(mean=15.0, sigma=0.5, size=n)
    index = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"Open": open_, "High": hi_raw, "Low": lo_raw, "Close": close, "Volume": volume},
        index=index,
    )


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    return make_synthetic_ohlcv()
