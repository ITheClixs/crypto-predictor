"""Turn forecasts into positions and net-of-cost PnL.

To avoid double-counting overlapping ``h``-day forecasts, trades are taken on a
non-overlapping schedule: one decision every ``horizon`` bars, held to the next.
Positions are unit-sized (``sign`` = long/short, ``long_flat`` = long/flat), which
keeps the mapping transparent. The point of the study is whether the *signal*
survives costs, not leverage engineering.

Sampling every ``h``-th bar leaves ``h`` possible starting offsets, and nothing
makes offset 0 special. :func:`phase_sharpes` runs all of them so the reported
number can be checked against the ones the arbitrary choice discarded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import CostModel
from ..evaluate.stats import sharpe_ratio
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
    elif kind == "always_long":
        values = np.ones(len(y_pred), dtype=float)
    else:
        raise ValueError(f"unknown strategy kind {kind!r}; use 'sign', 'long_flat', 'always_long'")
    return pd.Series(values, index=y_pred.index)


def backtest_strategy(
    oos: pd.DataFrame,
    horizon: int,
    costs: CostModel,
    kind: str = "sign",
    log_target: bool = True,
    phase: int = 0,
) -> StrategyResult:
    """Backtest a forecast series into a net-of-cost equity curve.

    ``oos`` must have ``y_true`` and ``y_pred`` columns (as returned by
    :func:`cryptoforecast.backtest.engine.walk_forward`). With ``log_target`` the
    realized return is converted from log to simple space for compounding.
    ``phase`` selects which of the ``horizon`` non-overlapping schedules to trade.
    """
    decisions = oos.iloc[phase % horizon :: horizon]
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


def buy_and_hold(oos: pd.DataFrame, horizon: int, costs: CostModel) -> StrategyResult:
    """Always-long reference on the same schedule and cost model as the primary strategy.

    The point of comparison for every long-biased forecaster. A model whose only
    achievement is staying long during a bull market should be visibly indistinct
    from this line, not quietly credited with the market's return. It is built with
    :func:`staggered_strategy` so that it shares the primary specification's daily
    rebalancing, execution lag and annualisation; comparing a daily-rebalanced strategy
    against a reference sampled every ``h`` bars would compare two different things.
    """
    return staggered_strategy(oos, horizon, costs, kind="always_long")


def staggered_strategy(
    oos: pd.DataFrame,
    horizon: int,
    costs: CostModel,
    kind: str = "sign",
    execution_lag: int = 1,
) -> StrategyResult:
    """Equal-weighted portfolio over all ``horizon`` daily vintages, executed with a lag.

    Two defects of the single-phase backtest are fixed here at once.

    *Phase arbitrariness.* Sampling every ``h``-th bar leaves ``h`` schedules and nothing
    distinguishes them; at ``h = 7`` the resulting Sharpe ratios span about 1.4 points, so the
    headline number was partly a property of where sampling began. No investor would put all
    capital in one weekly vintage. Allocating ``1/h`` to each vintage uses every daily signal
    while keeping the ``h``-day holding period, which makes the aggregate weight the rolling
    mean of the last ``h`` signals.

    *Infeasible execution.* The features at bar ``t`` are computed from the completed close
    ``C_t``, so entering at that same close requires knowing the price before it prints. With
    ``execution_lag = 1`` the weight in force during bar ``t`` depends only on signals through
    ``t - 1``, and the return earned is the realised one-bar return of bar ``t``.

    Because the portfolio rebalances daily, PnL is computed on one-bar returns taken from the
    ``close`` column rather than on the ``h``-bar label, and it annualises at 365.
    """
    if "close" not in oos:
        raise ValueError("staggered_strategy needs a 'close' column to compute one-bar returns")
    signals = build_positions(oos["y_pred"], kind)
    # Weight in force during bar t: the average of the h most recent signals, all of which
    # were observable by t - execution_lag.
    weights = signals.shift(execution_lag).rolling(horizon, min_periods=1).mean()
    close = oos["close"].astype(float)
    one_bar = close.pct_change()

    valid = weights.notna() & one_bar.notna()
    weights, one_bar = weights[valid], one_bar[valid]

    gross = weights * one_bar
    turn = (weights - weights.shift(1)).abs().fillna(weights.abs())
    net = gross - turn * costs.cost_per_side
    equity = (1.0 + net).cumprod()

    return StrategyResult(
        positions=weights,
        gross=gross,
        net=net,
        equity=equity,
        turnover=turn,
        horizon=horizon,
        periods_per_year=CRYPTO_DAYS_PER_YEAR,
    )


def phase_sharpes(
    oos: pd.DataFrame, horizon: int, costs: CostModel, kind: str = "sign"
) -> list[float]:
    """Annualized net Sharpe for each of the ``horizon`` possible start offsets.

    A signal worth anything should not care which offset it is traded on. A wide
    spread here means the headline Sharpe is an artifact of where the sampling
    happened to start.
    """
    out = []
    for phase in range(horizon):
        result = backtest_strategy(oos, horizon, costs, kind=kind, phase=phase)
        out.append(sharpe_ratio(result.net, result.periods_per_year))
    return out
