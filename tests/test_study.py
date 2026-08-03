"""The study runner wires data -> dataset -> models -> OOS with no network."""

from __future__ import annotations

import math

import numpy as np
import pytest

import cryptoforecast.study as study_mod
from conftest import make_synthetic_ohlcv
from cryptoforecast.config import StudyConfig, WalkForwardConfig
from cryptoforecast.evaluate.stats import clark_west, pesaran_timmermann
from cryptoforecast.models.registry import default_models


def _small_cfg(horizon: int = 1) -> StudyConfig:
    return StudyConfig(
        assets=("BTC",),
        horizons=(horizon,),
        wf=WalkForwardConfig(train_size=250, test_size=50, embargo=3, min_train=150),
    )


@pytest.mark.unit
def test_run_study_produces_one_run_per_model(monkeypatch: pytest.MonkeyPatch) -> None:
    ohlcv = make_synthetic_ohlcv(n=900, seed=3)
    monkeypatch.setattr(study_mod, "load_ohlcv", lambda *a, **k: ohlcv)

    result = study_mod.run_study(_small_cfg())

    assert {r.model for r in result.runs} == set(default_models())
    for run in result.runs:
        assert list(run.oos.columns) == ["y_true", "y_pred", "close", "fold"]
        assert run.oos.index.is_monotonic_increasing


@pytest.mark.unit
def test_study_reports_clark_west_against_both_benchmarks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The zero forecast is not the slopes-zero restriction of a model with an intercept.

    Reporting only ``CW vs the random walk`` tests "no drift AND no conditional
    predictability" jointly. The recursive mean is the restriction these estimators
    actually nest, so both statistics have to be carried; on this study's real
    forecasts they disagree about 4 of 18 settings.
    """
    ohlcv = make_synthetic_ohlcv(n=900, seed=3)
    monkeypatch.setattr(study_mod, "load_ohlcv", lambda *a, **k: ohlcv)

    result = study_mod.run_study(_small_cfg())
    by_model = {r.model: r for r in result.runs}
    ridge = by_model["ridge"]
    mean_pred = by_model[study_mod.R2_BENCHMARK].oos["y_pred"].to_numpy()

    expected = clark_west(ridge.oos["y_true"], ridge.oos["y_pred"].to_numpy(), mean_pred, horizon=1)
    assert ridge.cw_stat_vs_mean == pytest.approx(expected.statistic)
    assert ridge.cw_p_vs_mean == pytest.approx(expected.p_value)

    # The recursive-mean benchmark scores exactly zero against itself, as the zero
    # forecast does under CW vs the random walk.
    assert math.isnan(by_model[study_mod.R2_BENCHMARK].cw_stat_vs_mean)


@pytest.mark.unit
def test_study_sign_test_uses_non_overlapping_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    """At h > 1 the reported PT statistic must be the phase-averaged one, not the raw one."""
    ohlcv = make_synthetic_ohlcv(n=900, seed=3)
    monkeypatch.setattr(study_mod, "load_ohlcv", lambda *a, **k: ohlcv)

    result = study_mod.run_study(_small_cfg(horizon=7))
    ridge = next(r for r in result.runs if r.model == "ridge")

    raw = pesaran_timmermann(ridge.oos["y_true"], ridge.oos["y_pred"].to_numpy())
    assert np.isfinite(ridge.pt_stat)
    assert ridge.pt_stat != pytest.approx(raw.statistic)
    assert abs(ridge.pt_stat) < abs(raw.statistic)
