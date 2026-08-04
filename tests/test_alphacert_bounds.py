"""Tests for the anytime-valid ceiling on incremental value."""

from __future__ import annotations

import numpy as np
import pytest

from alphacert import value_ceiling

SIGMA = 0.03


@pytest.mark.unit
def test_the_interval_brackets_zero_when_the_signal_is_noise() -> None:
    rng = np.random.default_rng(0)
    outcome = 0.001 + SIGMA * rng.standard_normal(1500)
    bound = value_ceiling(rng.standard_normal(1500), outcome, alpha=0.05)
    assert bound.lower <= 0.0 <= bound.upper
    assert bound.upper > 0.0


@pytest.mark.unit
def test_the_ceiling_tightens_as_evidence_accumulates() -> None:
    rng = np.random.default_rng(1)
    long_outcome = 0.001 + SIGMA * rng.standard_normal(4000)
    long_signal = rng.standard_normal(4000)
    wide = value_ceiling(long_signal[:600], long_outcome[:600])
    narrow = value_ceiling(long_signal, long_outcome)
    assert narrow.upper < wide.upper


@pytest.mark.unit
def test_the_ceiling_covers_a_real_signal_it_cannot_yet_exclude() -> None:
    rng = np.random.default_rng(2)
    signal = rng.standard_normal(3000)
    truth = 0.004
    outcome = 0.001 + truth * signal + SIGMA * rng.standard_normal(3000)
    bound = value_ceiling(signal, outcome)
    assert bound.lower <= truth <= bound.upper


@pytest.mark.unit
def test_ratio_ceiling_is_the_bound_read_as_an_information_ratio() -> None:
    rng = np.random.default_rng(3)
    outcome = 0.001 + SIGMA * rng.standard_normal(1200)
    bound = value_ceiling(rng.standard_normal(1200), outcome)
    ratio = bound.ratio_ceiling(SIGMA, periods_per_year=365.0)
    assert ratio == pytest.approx(bound.upper / SIGMA * np.sqrt(365.0))
    assert ratio > 0.0
    with pytest.raises(ValueError):
        bound.ratio_ceiling(0.0)


@pytest.mark.unit
def test_bounds_validate_their_input() -> None:
    rng = np.random.default_rng(4)
    with pytest.raises(ValueError, match="aligned"):
        value_ceiling(rng.standard_normal(5), rng.standard_normal(6))
    with pytest.raises(ValueError, match="alpha"):
        value_ceiling(rng.standard_normal(5), rng.standard_normal(5), alpha=0.0)
    with pytest.raises(ValueError, match="at least one"):
        value_ceiling(np.zeros(0), np.zeros(0))
