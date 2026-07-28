"""The study runner wires data -> dataset -> models -> OOS with no network."""

from __future__ import annotations

import pytest

import cryptoforecast.study as study_mod
from conftest import make_synthetic_ohlcv
from cryptoforecast.config import StudyConfig, WalkForwardConfig
from cryptoforecast.models.registry import default_models


@pytest.mark.unit
def test_run_study_produces_one_run_per_model(monkeypatch: pytest.MonkeyPatch) -> None:
    ohlcv = make_synthetic_ohlcv(n=900, seed=3)
    monkeypatch.setattr(study_mod, "load_ohlcv", lambda *a, **k: ohlcv)

    cfg = StudyConfig(
        assets=("BTC",),
        horizons=(1,),
        wf=WalkForwardConfig(train_size=250, test_size=50, embargo=3, min_train=150),
    )
    result = study_mod.run_study(cfg)

    assert {r.model for r in result.runs} == set(default_models())
    for run in result.runs:
        assert list(run.oos.columns) == ["y_true", "y_pred", "close", "fold"]
        assert run.oos.index.is_monotonic_increasing
