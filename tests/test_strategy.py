"""Cost model and forecast-to-PnL strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cryptoforecast.backtest.costs import trading_costs, turnover
from cryptoforecast.backtest.strategy import (
    backtest_strategy,
    build_positions,
    buy_and_hold,
    phase_sharpes,
)
from cryptoforecast.config import CostModel


@pytest.mark.unit
def test_turnover_counts_position_changes() -> None:
    pos = pd.Series([1.0, 1.0, -1.0, 0.0])
    np.testing.assert_allclose(turnover(pos).to_numpy(), [1.0, 0.0, 2.0, 1.0])


@pytest.mark.unit
def test_trading_costs_scale_with_turnover() -> None:
    pos = pd.Series([1.0, -1.0])
    costs = trading_costs(pos, cost_per_side=0.001)
    np.testing.assert_allclose(costs.to_numpy(), [0.001, 0.002])


def _oos(y_true: list[float], y_pred: list[float]) -> pd.DataFrame:
    index = pd.date_range("2022-01-01", periods=len(y_true), freq="D")
    return pd.DataFrame(
        {"y_true": y_true, "y_pred": y_pred, "close": 100.0, "fold": 0}, index=index
    )


@pytest.mark.unit
def test_perfect_signal_makes_money() -> None:
    rng = np.random.default_rng(0)
    truth = rng.normal(0, 0.03, 120).tolist()
    oos = _oos(truth, truth)  # predictions have the correct sign everywhere
    result = backtest_strategy(oos, horizon=1, costs=CostModel())
    assert (result.gross >= 0).all()  # long/short in the right direction
    assert result.equity.iloc[-1] > 1.0


@pytest.mark.unit
def test_zero_forecast_takes_no_risk() -> None:
    oos = _oos([0.01, -0.02, 0.03], [0.0, 0.0, 0.0])
    result = backtest_strategy(oos, horizon=1, costs=CostModel())
    assert (result.positions == 0).all()
    assert np.allclose(result.net.to_numpy(), 0.0)
    assert np.allclose(result.equity.to_numpy(), 1.0)


@pytest.mark.unit
def test_costs_reduce_net_return() -> None:
    truth = [0.02, -0.02, 0.02, -0.02, 0.02, -0.02]
    oos = _oos(truth, truth)
    free = backtest_strategy(oos, 1, CostModel(0.0, 0.0, 0.0)).net.sum()
    charged = backtest_strategy(oos, 1, CostModel(20.0, 10.0, 5.0)).net.sum()
    assert charged < free


@pytest.mark.unit
def test_non_overlapping_sampling() -> None:
    oos = _oos(list(np.zeros(70)), list(np.zeros(70)))
    result = backtest_strategy(oos, horizon=7, costs=CostModel())
    assert len(result.net) == 10  # 70 bars / 7-day step


@pytest.mark.unit
def test_long_flat_never_shorts() -> None:
    positions = build_positions(pd.Series([0.1, -0.1, 0.0, 0.2]), kind="long_flat")
    np.testing.assert_allclose(positions.to_numpy(), [1.0, 0.0, 0.0, 1.0])


@pytest.mark.unit
def test_build_positions_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown strategy kind"):
        build_positions(pd.Series([0.1, -0.1]), kind="martingale")


@pytest.mark.unit
def test_annualization_matches_the_sampling_frequency() -> None:
    """h-day decisions are sampled 365/h times a year, not 365."""
    oos = _oos(list(np.zeros(70)), list(np.zeros(70)))
    assert backtest_strategy(oos, horizon=1, costs=CostModel()).periods_per_year == 365.0
    assert backtest_strategy(oos, horizon=7, costs=CostModel()).periods_per_year == 365.0 / 7


@pytest.mark.unit
def test_buy_and_hold_ignores_the_forecast_and_trades_once() -> None:
    truth = [0.02, -0.01, 0.03, -0.02, 0.01, 0.04]
    oos = _oos(truth, [-1.0] * 6)  # forecast says short every single period
    result = buy_and_hold(oos, horizon=1, costs=CostModel())
    assert (result.positions == 1.0).all()
    assert int((result.turnover > 0).sum()) == 1  # enter once, then hold


@pytest.mark.unit
def test_an_always_long_forecast_is_indistinguishable_from_buy_and_hold() -> None:
    """The claim the report makes about `historical_mean`, pinned as a test."""
    truth = [0.02, -0.01, 0.03, -0.02, 0.01, 0.04]
    drift = [0.001] * 6  # a positive constant: always long, by construction
    oos = _oos(truth, drift)
    signal = backtest_strategy(oos, horizon=1, costs=CostModel())
    reference = buy_and_hold(oos, horizon=1, costs=CostModel())
    np.testing.assert_allclose(signal.net.to_numpy(), reference.net.to_numpy())


@pytest.mark.unit
def test_phase_sharpes_covers_every_start_offset() -> None:
    rng = np.random.default_rng(2)
    truth = rng.normal(0, 0.03, 140).tolist()
    oos = _oos(truth, truth)
    assert len(phase_sharpes(oos, horizon=1, costs=CostModel())) == 1
    phases = phase_sharpes(oos, horizon=7, costs=CostModel())
    assert len(phases) == 7
    assert all(s > 0 for s in phases)  # a perfect signal works on every offset


@pytest.mark.unit
def test_phase_offset_selects_a_different_schedule() -> None:
    oos = _oos(list(np.arange(21, dtype=float) / 100), list(np.arange(21, dtype=float) / 100))
    first = backtest_strategy(oos, horizon=7, costs=CostModel(), phase=0)
    second = backtest_strategy(oos, horizon=7, costs=CostModel(), phase=1)
    assert list(first.net.index) != list(second.net.index)
