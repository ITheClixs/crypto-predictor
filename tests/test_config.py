"""Configuration objects are immutable and validated."""

from __future__ import annotations

import dataclasses

import pytest

from cryptoforecast.config import CostModel, StudyConfig, WalkForwardConfig


@pytest.mark.unit
def test_frozen() -> None:
    cfg = StudyConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.seed = 99  # type: ignore[misc]


@pytest.mark.unit
def test_cost_per_side() -> None:
    costs = CostModel(fee_bps=10.0, slippage_bps=5.0, half_spread_bps=2.0)
    assert costs.cost_per_side == pytest.approx(17e-4)


@pytest.mark.unit
def test_walk_forward_validation() -> None:
    with pytest.raises(ValueError, match="mode must be"):
        WalkForwardConfig(mode="sideways")
    with pytest.raises(ValueError, match="must be positive"):
        WalkForwardConfig(train_size=0)
    with pytest.raises(ValueError, match="embargo must be non-negative"):
        WalkForwardConfig(embargo=-1)


@pytest.mark.unit
def test_defaults_are_sensible() -> None:
    cfg = StudyConfig()
    assert cfg.horizons == (1, 7)
    assert cfg.wf.train_size > cfg.wf.min_train
