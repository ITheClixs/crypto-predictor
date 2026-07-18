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
from .backtest.strategy import StrategyResult, backtest_strategy
from .config import StudyConfig
from .data.cache import DEFAULT_CACHE_DIR
from .data.loaders import load_ohlcv
from .dataset import build_supervised
from .evaluate.metrics import regression_metrics
from .evaluate.stats import diebold_mariano, pesaran_timmermann
from .models.registry import PRIMARY_BENCHMARK, default_models


@dataclass(frozen=True)
class ModelRun:
    asset: str
    horizon: int
    model: str
    oos: pd.DataFrame
    metrics: dict[str, float]
    strategy: StrategyResult
    dm_stat_vs_rw: float
    dm_p_vs_rw: float
    pt_stat: float
    pt_p: float


@dataclass(frozen=True)
class StudyResults:
    config: StudyConfig
    runs: list[ModelRun]


def run_asset_horizon(
    ohlcv: pd.DataFrame, asset: str, horizon: int, cfg: StudyConfig
) -> list[ModelRun]:
    ds = build_supervised(ohlcv, horizon, target="logret")
    models = default_models()

    # All models share one dataset and one set of splits, so their OOS indices
    # coincide and the random-walk column lines up for the DM test.
    oos_by_model = {
        name: walk_forward(ds, factory, cfg.wf, horizon) for name, factory in models.items()
    }
    rw_pred = oos_by_model[PRIMARY_BENCHMARK]["y_pred"].to_numpy()

    runs: list[ModelRun] = []
    for name, oos in oos_by_model.items():
        y_true = oos["y_true"]
        y_pred = oos["y_pred"].to_numpy()
        dm = diebold_mariano(y_true, y_pred, rw_pred, horizon=horizon)
        pt = pesaran_timmermann(y_true, y_pred)
        runs.append(
            ModelRun(
                asset=asset,
                horizon=horizon,
                model=name,
                oos=oos,
                metrics=regression_metrics(y_true, y_pred),
                strategy=backtest_strategy(oos, horizon, cfg.costs, kind="sign"),
                dm_stat_vs_rw=dm.statistic,
                dm_p_vs_rw=dm.p_value,
                pt_stat=pt.statistic,
                pt_p=pt.p_value,
            )
        )
    return runs


def run_study(cfg: StudyConfig, cache_dir: Path = DEFAULT_CACHE_DIR) -> StudyResults:
    runs: list[ModelRun] = []
    for asset in cfg.assets:
        ohlcv = load_ohlcv(asset, cfg.start, cfg.end, cfg.interval, cache_dir=cache_dir)
        for horizon in cfg.horizons:
            runs.extend(run_asset_horizon(ohlcv, asset, horizon, cfg))
    return StudyResults(config=cfg, runs=runs)
