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


def make_study(n: int = 300):
    """A small, deterministic StudyResults for one asset/horizon with two models.

    Built directly (bypassing the network and model fitting) so reporting, plotting,
    and CLI code can be exercised fast. The ``ridge`` run is a near-perfect sign
    predictor; the ``random_walk`` run predicts zero.
    """
    from cryptoforecast.backtest.strategy import (
        backtest_strategy,
        buy_and_hold,
        phase_sharpes,
        staggered_strategy,
    )
    from cryptoforecast.config import CostModel, StudyConfig
    from cryptoforecast.evaluate.metrics import regression_metrics
    from cryptoforecast.study import ModelRun, StudyResults

    rng = np.random.default_rng(11)
    index = pd.date_range("2021-01-01", periods=n, freq="D")
    y_true = pd.Series(rng.normal(0.0, 0.03, n), index=index)
    close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.03, n))), index=index)
    costs = CostModel()

    def _run(
        model: str,
        y_pred: pd.Series,
        dm_stat: float,
        dm_p: float,
        cw_stat: float,
        cw_p: float,
        pt_s: float,
        pt_p: float,
        cw_stat_mean: float | None = None,
        cw_p_mean: float | None = None,
    ):
        oos = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "close": close, "fold": 0})
        return ModelRun(
            asset="BTC",
            horizon=1,
            model=model,
            oos=oos,
            metrics=regression_metrics(y_true, y_pred.to_numpy()),
            strategy=staggered_strategy(oos, 1, costs),
            same_close_strategy=backtest_strategy(oos, 1, costs),
            dm_stat_vs_rw=dm_stat,
            dm_p_vs_rw=dm_p,
            cw_stat_vs_rw=cw_stat,
            cw_p_vs_rw=cw_p,
            cw_stat_vs_mean=cw_stat if cw_stat_mean is None else cw_stat_mean,
            cw_p_vs_mean=cw_p if cw_p_mean is None else cw_p_mean,
            pt_stat=pt_s,
            pt_p=pt_p,
            sign_excess=pt_s,
            sign_p=pt_p,
            phase_sharpes=phase_sharpes(oos, 1, costs),
        )

    nan = float("nan")
    runs = [
        _run("random_walk", pd.Series(0.0, index=index), nan, nan, nan, nan, nan, nan),
        _run("ridge", y_true * 0.5, -2.5, 0.01, 2.4, 0.008, 2.1, 0.02),
    ]
    config = StudyConfig(assets=("BTC",), horizons=(1,))
    reference = buy_and_hold(runs[0].oos, 1, costs)
    return StudyResults(config=config, runs=runs, buy_and_hold={("BTC", 1): reference})
