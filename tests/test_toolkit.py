"""Tests for the conventional forecast-comparison toolkit.

These four statistics carry the paper's claim that the certificate's non-result is a property
of the data rather than of our instrument. If any of them is mis-implemented that argument
collapses, so each is pinned against a property it must satisfy: an algebraic identity, a
known null behaviour, or a case with an unambiguous right answer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "audit" / "scripts"))

from goyal_welch_toolkit import (  # noqa: E402
    clark_west,
    diebold_mariano,
    encompassing_t,
    model_confidence_set,
)


def _nested_forecasts(
    rng: np.random.Generator, n: int, signal_strength: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A benchmark, a nested model that adds noise plus ``signal_strength`` of real edge."""
    truth = signal_strength * rng.standard_normal(n)
    outcome = truth + rng.standard_normal(n)
    bench = np.zeros(n)
    model = truth + 0.3 * rng.standard_normal(n)
    return outcome, bench, model


@pytest.mark.unit
def test_encompassing_t_is_clark_west_for_a_one_step_nested_pair() -> None:
    """The identity the paper asserts: ENC-t and Clark-West are one statistic, not two.

    The Clark-West adjusted loss differential is exactly twice the encompassing product,
    so the scale cancels in the t-ratio. Reporting both as corroboration would double-count.
    """
    rng = np.random.default_rng(11)
    for strength in (0.0, 0.2, 0.8):
        outcome, bench, model = _nested_forecasts(rng, 600, strength)
        assert clark_west(outcome, bench, model)[0] == pytest.approx(
            encompassing_t(outcome, bench, model)[0], rel=1e-12
        )


@pytest.mark.unit
def test_clark_west_is_calibrated_when_the_extra_regressor_is_noise() -> None:
    """Under the null the one-sided rejection rate should sit near the nominal level."""
    rejections = 0
    reps = 300
    for rep in range(reps):
        rng = np.random.default_rng([13, rep])
        outcome, bench, model = _nested_forecasts(rng, 400, signal_strength=0.0)
        rejections += clark_west(outcome, bench, model)[1] < 0.05
    assert 0.01 < rejections / reps < 0.12


@pytest.mark.unit
def test_clark_west_detects_a_real_edge() -> None:
    rng = np.random.default_rng(17)
    outcome, bench, model = _nested_forecasts(rng, 2000, signal_strength=0.5)
    stat, p = clark_west(outcome, bench, model)
    assert stat > 0.0
    assert p < 0.01


@pytest.mark.unit
def test_diebold_mariano_sign_follows_the_better_forecast() -> None:
    """Positive when the model beats the benchmark; the statistic itself is well defined.

    Validity under nesting is a separate matter and is why the paper reports DM only for
    completeness; this test pins the arithmetic, not the sampling distribution.
    """
    rng = np.random.default_rng(19)
    outcome, bench, model = _nested_forecasts(rng, 1500, signal_strength=0.7)
    assert diebold_mariano((outcome - bench) ** 2, (outcome - model) ** 2)[0] > 0.0
    assert diebold_mariano((outcome - model) ** 2, (outcome - bench) ** 2)[0] < 0.0


@pytest.mark.unit
def test_diebold_mariano_reports_nan_on_a_degenerate_differential() -> None:
    zeros = np.zeros(50)
    assert np.isnan(diebold_mariano(zeros, zeros)[0])


@pytest.mark.unit
def test_model_confidence_set_keeps_everything_when_models_are_equivalent() -> None:
    """Exchangeable losses are indistinguishable, so nothing may be eliminated."""
    rng = np.random.default_rng(23)
    losses = rng.chisquare(3, size=(500, 5))
    survivors = model_confidence_set(losses, [f"m{i}" for i in range(5)], alpha=0.10)
    assert len(survivors) == 5


@pytest.mark.unit
def test_model_confidence_set_eliminates_a_clearly_worse_model() -> None:
    """One model with a large, persistent loss penalty must not survive."""
    rng = np.random.default_rng(29)
    base = rng.chisquare(3, size=(800, 4))
    losses = np.column_stack([base, base[:, 0] + 8.0])
    names = ["a", "b", "c", "d", "terrible"]
    survivors = model_confidence_set(losses, names, alpha=0.10)
    assert "terrible" not in survivors
    assert len(survivors) >= 1


@pytest.mark.unit
def test_model_confidence_set_never_returns_empty() -> None:
    """The elimination loop stops at one model, whatever the data look like."""
    rng = np.random.default_rng(31)
    losses = np.column_stack([rng.chisquare(1, 300) + 5.0 * k for k in range(4)])
    assert len(model_confidence_set(losses, list("abcd"), alpha=0.99)) >= 1
