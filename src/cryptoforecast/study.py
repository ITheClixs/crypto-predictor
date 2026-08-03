"""Run every model through the walk-forward pipeline for each asset and horizon.

This is the orchestration layer: it loads data, builds the leak-free dataset, runs
each model out-of-sample, and packages the predictions together with forecast
metrics and the Diebold-Mariano / Pesaran-Timmermann tests against the random-walk
null. Rendering (tables, figures, markdown) lives in :mod:`evaluate.report` and
:mod:`plots`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .backtest.engine import walk_forward
from .backtest.strategy import StrategyResult, backtest_strategy, buy_and_hold, phase_sharpes
from .config import StudyConfig
from .data.cache import DEFAULT_CACHE_DIR
from .data.loaders import load_ohlcv
from .dataset import build_supervised
from .evaluate.metrics import regression_metrics
from .evaluate.stats import clark_west, diebold_mariano, pesaran_timmermann_non_overlapping
from .models.registry import PRIMARY_BENCHMARK, default_models

#: The ex-ante reference forecast for the out-of-sample R^2 (Campbell-Thompson), and the
#: forecaster the machine-learning models actually nest: setting their slopes to zero
#: returns the training-window mean, not zero, because every one fits an unpenalised
#: intercept. Clark-West is reported against both this and ``PRIMARY_BENCHMARK``; the
#: latter adds "and the mean return is zero" to the null being tested.
R2_BENCHMARK = "historical_mean"


@dataclass(frozen=True)
class ModelRun:
    asset: str
    horizon: int
    model: str
    oos: pd.DataFrame
    metrics: dict[str, float]
    strategy: StrategyResult
    dm_stat_vs_rw: float  # negative => lower squared error than the random walk
    dm_p_vs_rw: float  # two-sided
    cw_stat_vs_rw: float  # positive => beats the zero forecast; joint no-drift null
    cw_p_vs_rw: float  # one-sided
    cw_stat_vs_mean: float  # positive => beats the recursive mean it actually nests
    cw_p_vs_mean: float  # one-sided
    pt_stat: float  # positive => sign-timing skill; phase-averaged, non-overlapping
    pt_p: float  # two-sided, mean over phases
    phase_sharpes: list[float]  # net Sharpe on each of the h start offsets


@dataclass(frozen=True)
class StudyResults:
    config: StudyConfig
    runs: list[ModelRun]
    #: Always-long reference per (asset, horizon), on the same schedule and costs.
    buy_and_hold: dict[tuple[str, int], StrategyResult]


def run_asset_horizon(
    ohlcv: pd.DataFrame, asset: str, horizon: int, cfg: StudyConfig
) -> tuple[list[ModelRun], StrategyResult]:
    """Run every model for one (asset, horizon) and the buy-and-hold reference."""
    ds = build_supervised(ohlcv, horizon, target="logret")
    models = default_models(horizon)

    # All models share one dataset and one set of splits, so their OOS indices
    # coincide and the random-walk column lines up for the DM test.
    oos_by_model = {
        name: walk_forward(ds, factory, cfg.wf, horizon) for name, factory in models.items()
    }
    rw_pred = oos_by_model[PRIMARY_BENCHMARK]["y_pred"].to_numpy()
    r2_bench_pred = oos_by_model[R2_BENCHMARK]["y_pred"].to_numpy()

    runs: list[ModelRun] = []
    for name, oos in oos_by_model.items():
        y_true = oos["y_true"]
        y_pred = oos["y_pred"].to_numpy()
        dm = diebold_mariano(y_true, y_pred, rw_pred, horizon=horizon)
        cw = clark_west(y_true, y_pred, rw_pred, horizon=horizon)
        cw_mean = clark_west(y_true, y_pred, r2_bench_pred, horizon=horizon)
        pt = pesaran_timmermann_non_overlapping(y_true, y_pred, horizon=horizon)
        runs.append(
            ModelRun(
                asset=asset,
                horizon=horizon,
                model=name,
                oos=oos,
                metrics=regression_metrics(y_true, y_pred, r2_bench_pred),
                strategy=backtest_strategy(oos, horizon, cfg.costs, kind="sign"),
                dm_stat_vs_rw=dm.statistic,
                dm_p_vs_rw=dm.p_value,
                cw_stat_vs_rw=cw.statistic,
                cw_p_vs_rw=cw.p_value,
                cw_stat_vs_mean=cw_mean.statistic,
                cw_p_vs_mean=cw_mean.p_value,
                pt_stat=pt.statistic,
                pt_p=pt.p_value,
                phase_sharpes=phase_sharpes(oos, horizon, cfg.costs),
            )
        )
    reference = buy_and_hold(oos_by_model[PRIMARY_BENCHMARK], horizon, cfg.costs)
    return runs, reference


def run_study(cfg: StudyConfig, cache_dir: Path = DEFAULT_CACHE_DIR) -> StudyResults:
    runs: list[ModelRun] = []
    references: dict[tuple[str, int], StrategyResult] = {}
    for asset in cfg.assets:
        ohlcv = load_ohlcv(asset, cfg.start, cfg.end, cfg.interval, cache_dir=cache_dir)
        for horizon in cfg.horizons:
            asset_runs, reference = run_asset_horizon(ohlcv, asset, horizon, cfg)
            runs.extend(asset_runs)
            references[(asset, horizon)] = reference
    return StudyResults(config=cfg, runs=runs, buy_and_hold=references)
