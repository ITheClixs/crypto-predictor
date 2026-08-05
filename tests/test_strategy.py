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
    staggered_strategy,
)
from cryptoforecast.config import CostModel
from cryptoforecast.evaluate.stats import sharpe_ratio


@pytest.mark.unit
def test_turnover_counts_position_changes() -> None:
    pos = pd.Series([1.0, 1.0, -1.0, 0.0])
    np.testing.assert_allclose(turnover(pos).to_numpy(), [1.0, 0.0, 2.0, 1.0])


@pytest.mark.unit
def test_trading_costs_scale_with_turnover() -> None:
    pos = pd.Series([1.0, -1.0])
    costs = trading_costs(pos, cost_per_side=0.001)
    np.testing.assert_allclose(costs.to_numpy(), [0.001, 0.002])


def _oos_with_path(returns: list[float], preds: list[float]) -> pd.DataFrame:
    """OOS frame whose ``close`` column is the actual path implied by ``returns``."""
    index = pd.date_range("2022-01-01", periods=len(returns), freq="D")
    close = 100.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame(
        {"y_true": returns, "y_pred": preds, "close": close, "fold": 0}, index=index
    )


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
    """The claim the report makes about `historical_mean`, pinned as a test.

    Compared inside the primary specification: a forecaster that is always long and the
    always-long reference must produce identical net returns, or the reference is not a
    reference. Both are staggered with the same execution lag, because comparing a
    daily-rebalanced strategy against one sampled every ``h`` bars compares two things.
    """
    returns = [0.02, -0.01, 0.03, -0.02, 0.01, 0.04, 0.0, -0.03]
    drift = [0.001] * len(returns)  # a positive constant: always long, by construction
    oos = _oos_with_path(returns, drift)
    signal = staggered_strategy(oos, horizon=1, costs=CostModel())
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


@pytest.mark.unit
def test_staggered_weight_is_the_rolling_mean_of_recent_signals() -> None:
    """The portfolio holds 1/h in each vintage, so the aggregate weight averages h signals."""
    preds = [1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0]
    oos = _oos_with_path([0.01] * len(preds), preds)
    result = staggered_strategy(oos, horizon=3, costs=CostModel(), entry_delay=1)

    signs = pd.Series(np.sign(preds), index=oos.index)
    # entry_delay = 1 holds from one bar after the signal, and a weight aligned to bar t is
    # held from close t-1, so the signal is lagged twice in total.
    expected = signs.shift(2).rolling(3, min_periods=1).mean()
    pd.testing.assert_series_equal(
        result.positions, expected[result.positions.index], check_names=False
    )
    assert result.positions.abs().max() <= 1.0


@pytest.mark.unit
def test_staggered_entry_delay_forbids_trading_at_the_signals_own_close() -> None:
    """The feasibility property the single-phase backtest lacked.

    Features at bar t are built from the close of bar t and predict the return over (t, t+1].
    Capturing that return means being positioned from close t -- an order filled at the very
    price the signal was computed from. With ``entry_delay = 1`` that return is forgone, which
    is the whole economic content of the correction.
    """
    returns = [0.0] * 8 + [0.5, 0.5]
    preds = [0.0] * 8 + [1.0, 1.0]  # the forecast only turns positive on bar 8
    oos = _oos_with_path(returns, preds)
    same_close = staggered_strategy(oos, horizon=1, costs=CostModel(), entry_delay=0)
    delayed = staggered_strategy(oos, horizon=1, costs=CostModel(), entry_delay=1)
    # Bar 9's return is (8, 9], which the signal at bar 8 predicts. The same-close convention
    # collects it; the feasible one cannot.
    assert same_close.gross.iloc[-1] > 0.0
    assert delayed.gross.iloc[-1] == pytest.approx(0.0)


@pytest.mark.unit
def test_staggered_constant_signal_stops_trading() -> None:
    """An always-long forecast should pay one entry cost and then nothing."""
    oos = _oos_with_path([0.001] * 40, [1.0] * 40)
    result = staggered_strategy(oos, horizon=7, costs=CostModel(), entry_delay=1)
    # Weight ramps to 1 over the first h bars, then stops moving.
    assert result.turnover.iloc[11:].sum() == pytest.approx(0.0, abs=1e-12)
    assert result.positions.iloc[-1] == pytest.approx(1.0)


@pytest.mark.unit
def test_staggered_annualises_daily_because_it_rebalances_daily() -> None:
    oos = _oos_with_path([0.001] * 60, [1.0] * 60)
    result = staggered_strategy(oos, horizon=7, costs=CostModel())
    assert result.periods_per_year == pytest.approx(365.0)


@pytest.mark.unit
def test_entry_delay_zero_reproduces_the_same_close_convention() -> None:
    """Pins the meaning of ``entry_delay``, which a previous revision got wrong.

    At ``h = 1`` the staggered portfolio with no delay is exactly the single-phase backtest:
    one signal, one bar, entered at the close it was computed from. If this ever stops
    holding, the delay argument has silently changed meaning and every reported Sharpe ratio
    has moved with it.
    """
    idx = pd.date_range("2021-01-01", periods=400, freq="D")
    rng = np.random.default_rng(11)
    close = pd.Series(100.0 * np.exp(np.cumsum(0.02 * rng.standard_normal(400))), index=idx)
    oos = pd.DataFrame(
        {
            "y_pred": pd.Series(rng.standard_normal(400), index=idx),
            "y_true": close.pct_change().shift(-1).fillna(0.0),
            "close": close,
        }
    )
    costs = CostModel()
    same_close = backtest_strategy(oos, 1, costs, kind="sign")
    staggered = staggered_strategy(oos, 1, costs, entry_delay=0)
    delayed = staggered_strategy(oos, 1, costs, entry_delay=1)

    # The two label the same holding differently: ``backtest_strategy`` indexes a position by
    # the bar whose forward label it earns, ``staggered_strategy`` by the bar whose realised
    # one-bar return it earns, which is one later. Undo that and the holdings coincide.
    aligned = staggered.positions.shift(-1).dropna()
    overlap = aligned.index.intersection(same_close.positions.index)
    assert overlap.size > 300
    pd.testing.assert_series_equal(
        aligned[overlap], same_close.positions[overlap], check_names=False
    )
    assert sharpe_ratio(staggered.net, periods_per_year=365) == pytest.approx(
        sharpe_ratio(same_close.net, periods_per_year=365), abs=0.02
    )
    # ...and the feasible convention is a genuinely different strategy, not a relabelling.
    both = delayed.positions.index.intersection(staggered.positions.index)
    assert not np.allclose(delayed.positions[both].to_numpy(), staggered.positions[both].to_numpy())
    assert sharpe_ratio(delayed.net, periods_per_year=365) != pytest.approx(
        sharpe_ratio(staggered.net, periods_per_year=365), abs=1e-6
    )


@pytest.mark.unit
def test_entry_delay_one_uses_only_information_from_before_the_bar_opens() -> None:
    """The feasible convention: nothing about bar t enters the weight held during bar t."""
    idx = pd.date_range("2021-01-01", periods=60, freq="D")
    signal = pd.Series(0.0, index=idx)
    signal.iloc[30] = 1.0  # a single positive signal on one bar
    close = pd.Series(np.linspace(100.0, 130.0, 60), index=idx)
    oos = pd.DataFrame({"y_pred": signal, "y_true": close.pct_change().shift(-1), "close": close})
    delayed = staggered_strategy(oos, 1, CostModel(), entry_delay=1)
    # The signal is known at the close of bar 30; the first bar it can influence is 32,
    # whose return spans (31, 32].
    assert delayed.positions.loc[idx[31]] == pytest.approx(0.0)
    assert delayed.positions.loc[idx[32]] > 0.0


@pytest.mark.unit
def test_entry_delay_must_be_non_negative() -> None:
    idx = pd.date_range("2021-01-01", periods=20, freq="D")
    oos = pd.DataFrame(
        {
            "y_pred": 1.0,
            "y_true": 0.0,
            "close": pd.Series(np.arange(20, dtype=float) + 100, index=idx),
        },
        index=idx,
    )
    with pytest.raises(ValueError, match="entry_delay"):
        staggered_strategy(oos, 1, CostModel(), entry_delay=-1)
