"""Forward-looking prediction targets.

Unlike features, a target legitimately depends on the future: the label for the
decision made at time ``t`` is the return realized over ``(t, t+h]``. The last
``h`` rows therefore have no label (the future has not happened yet) and are NaN;
those rows are exactly the ones a live model would forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def forward_log_return(close: pd.Series, horizon: int) -> pd.Series:
    """Log return realized from ``t`` to ``t + horizon``: log(C_{t+h} / C_t)."""
    _check_horizon(horizon)
    return np.log(close.shift(-horizon) / close)


def forward_simple_return(close: pd.Series, horizon: int) -> pd.Series:
    """Simple return realized from ``t`` to ``t + horizon``: C_{t+h} / C_t - 1."""
    _check_horizon(horizon)
    return close.shift(-horizon) / close - 1.0


def forward_direction(close: pd.Series, horizon: int) -> pd.Series:
    """Sign of the forward return as {0.0, 1.0}; NaN where the return is NaN."""
    fwd = forward_simple_return(close, horizon)
    direction = (fwd > 0).astype(float)
    return direction.where(fwd.notna())


def make_target(close: pd.Series, horizon: int, kind: str = "logret") -> pd.Series:
    """Dispatch to a target constructor by name."""
    builders = {
        "logret": forward_log_return,
        "simple": forward_simple_return,
        "direction": forward_direction,
    }
    if kind not in builders:
        raise ValueError(f"unknown target kind {kind!r}; choose one of {sorted(builders)}")
    return builders[kind](close, horizon)


def _check_horizon(horizon: int) -> None:
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
