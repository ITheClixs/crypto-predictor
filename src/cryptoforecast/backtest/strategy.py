"""Turn forecasts into positions and net-of-cost PnL.

To avoid double-counting overlapping ``h``-day forecasts, trades are taken on a
non-overlapping schedule: one decision every ``horizon`` bars, held to the next.
Positions are unit-sized (``sign`` = long/short, ``long_flat`` = long/flat), which
keeps the mapping transparent — the point of the study is whether the *signal*
survives costs, not leverage engineering.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import CostModel
from .costs import turnover

CRYPTO_DAYS_PER_YEAR = 365.0  # crypto trades every calendar day


@dataclass(frozen=True)
class StrategyResult:
    positions: pd.Series
    gross: pd.Series  # per-trade return before costs
    net: pd.Series  # per-trade return after costs
    equity: pd.Series  # cumulative net equity, starting near 1.0
    turnover: pd.Series
    horizon: int
    periods_per_year: float


def build_positions(y_pred: pd.Series, kind: str = "sign") -> pd.Series:
    """Map a forecast to a unit position."""
    if kind == "sign":
        values = np.sign(y_pred.to_numpy(dtype=float))
    elif kind == "long_flat":
        values = (y_pred.to_numpy(dtype=float) > 0).astype(float)
    else:
        raise ValueError(f"unknown strategy kind {kind!r}; use 'sign' or 'long_flat'")
    return pd.Series(values, index=y_pred.index)


def backtest_strategy(
    oos: pd.DataFrame,
    horizon: int,
    costs: CostModel,
    kind: str = "sign",
    log_target: bool = True,
) -> StrategyResult:
    """Backtest a forecast series into a net-of-cost equity curve.

    ``oos`` must have ``y_true`` and ``y_pred`` columns (as returned by
    :func:`cryptoforecast.backtest.engine.walk_forward`). With ``log_target`` the
    realized return is converted from log to simple space for compounding.
    """
    decisions = oos.iloc[::horizon]
    positions = build_positions(decisions["y_pred"], kind)
    realized = decisions["y_true"]
    fwd_simple = np.expm1(realized) if log_target else realized

    gross = positions * fwd_simple
    turn = turnover(positions)
    net = gross - turn * costs.cost_per_side
    equity = (1.0 + net).cumprod()

    return StrategyResult(
        positions=positions,
        gross=gross,
        net=net,
        equity=equity,
        turnover=turn,
        horizon=horizon,
        periods_per_year=CRYPTO_DAYS_PER_YEAR / horizon,
    )
