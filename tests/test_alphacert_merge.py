"""Tests for e-value merging, e-BH, and the overlapping-horizon phase split."""

from __future__ import annotations

import numpy as np
import pytest

from alphacert import (
    certify_overlapping,
    e_bh,
    merge_average,
    merge_product,
    phase_indices,
)

SIGMA = 0.03


@pytest.mark.unit
def test_average_of_e_values_is_bounded_by_the_extremes() -> None:
    assert merge_average([1.0, 3.0]) == pytest.approx(2.0)
    assert merge_average([1.0, 3.0], weights=[0.25, 0.75]) == pytest.approx(2.5)


@pytest.mark.unit
def test_averaging_null_e_values_keeps_expectation_below_one() -> None:
    """The property that makes a research grid correctable without a bootstrap.

    Eighteen certificates built on the *same* noise are maximally dependent. Their
    average is still an e-value, so its mean cannot exceed one.
    """
    means = []
    for rep in range(200):
        rng = np.random.default_rng([41, rep])
        shared = SIGMA * rng.standard_normal(600)
        evalues = [
            np.exp(np.sum(np.log1p(0.2 * np.sign(shared) * s)))
            for s in rng.choice((-1.0, 1.0), size=(18, 600))
        ]
        means.append(merge_average(evalues))
    assert np.mean(means) <= 1.2  # sampling slack around the theoretical bound of 1


@pytest.mark.unit
def test_product_requires_sequential_validity_and_is_reported_as_such() -> None:
    assert merge_product([2.0, 3.0]) == pytest.approx(6.0)
    with pytest.raises(ValueError, match="non-negative"):
        merge_product([2.0, -1.0])


@pytest.mark.unit
def test_merge_rejects_malformed_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        merge_average([])
    with pytest.raises(ValueError, match="non-negative"):
        merge_average([1.0, -2.0])
    with pytest.raises(ValueError, match="align"):
        merge_average([1.0, 2.0], weights=[1.0])
    with pytest.raises(ValueError, match="sum to 1"):
        merge_average([1.0, 2.0], weights=[0.5, 0.9])


@pytest.mark.unit
def test_e_bh_rejects_the_obvious_and_nothing_else() -> None:
    evalues = np.array([1000.0, 500.0, 1.0, 0.5, 0.1])
    rejected = e_bh(evalues, alpha=0.05)
    assert rejected.tolist() == [True, True, False, False, False]
    assert not e_bh(np.ones(10), alpha=0.05).any()


@pytest.mark.unit
def test_e_bh_threshold_matches_its_definition() -> None:
    """With m = 5 and alpha = 0.05 a single discovery needs an e-value of at least 100."""
    assert e_bh(np.array([99.0, 1.0, 1.0, 1.0, 1.0]), 0.05).sum() == 0
    assert e_bh(np.array([101.0, 1.0, 1.0, 1.0, 1.0]), 0.05).sum() == 1


@pytest.mark.unit
def test_e_bh_validates_input() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        e_bh(np.zeros(0))
    with pytest.raises(ValueError, match="alpha"):
        e_bh(np.ones(3), alpha=1.5)


@pytest.mark.unit
def test_phase_indices_partition_the_sample() -> None:
    idx = phase_indices(10, 3)
    assert len(idx) == 3
    assert sorted(np.concatenate(idx).tolist()) == list(range(10))
    assert idx[0].tolist() == [0, 3, 6, 9]
    with pytest.raises(ValueError):
        phase_indices(10, 0)


@pytest.mark.unit
def test_overlapping_certificate_splits_into_non_overlapping_phases() -> None:
    rng = np.random.default_rng(11)
    signal = rng.standard_normal(700)
    outcome = 0.001 + SIGMA * rng.standard_normal(700)
    evalue, certs = certify_overlapping(signal, outcome, horizon=7)
    assert len(certs) == 7
    assert sum(c.wealth.size for c in certs) == 700
    assert evalue == pytest.approx(np.mean([c.evalue for c in certs]))


@pytest.mark.unit
def test_overlapping_certificate_detects_a_real_multi_step_signal() -> None:
    rng = np.random.default_rng(12)
    base = SIGMA * rng.standard_normal(3500)
    signal = rng.standard_normal(3500)
    step = 0.006 * signal  # a strong seven-day signal, deliberately easy
    evalue, _ = certify_overlapping(signal, base + step, horizon=7)
    assert evalue > 20.0
