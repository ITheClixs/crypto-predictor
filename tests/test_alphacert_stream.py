"""Tests for the bare-return-stream certificate.

A published factor's long-short return is already a strategy return: the estimand is its
mean and there is no benchmark forecast to difference against. That makes the null simpler
than the one :func:`alphacert.certify` handles -- there is no drift nuisance, because the
drift is the thing being tested -- and a simpler null deserves its own primitive rather than
a mis-parameterised call to the general one.
"""

from __future__ import annotations

import numpy as np
import pytest

from alphacert import StreamCeiling, certify_mean, mean_ceiling

SIGMA = 0.04  # monthly long-short volatility, roughly what the factor zoo shows
BOUND = 0.6  # a priori envelope on a monthly long-short return


def _stream(rng: np.random.Generator, n: int, annual_sharpe: float) -> np.ndarray:
    return annual_sharpe / np.sqrt(12.0) * SIGMA + SIGMA * rng.standard_normal(n)


@pytest.mark.unit
def test_a_zero_mean_stream_is_not_certified_more_than_nominally() -> None:
    """The property the whole paper rests on: no edge, no certificate."""
    reps = 200
    rejections = 0
    for rep in range(reps):
        rng = np.random.default_rng([41, rep])
        rejections += certify_mean(_stream(rng, 600, 0.0), return_bound=BOUND).rejects(0.05)
    assert rejections / reps <= 0.10, f"leaked at {rejections / reps:.3f}"


@pytest.mark.unit
def test_a_real_edge_is_certified() -> None:
    rng = np.random.default_rng(43)
    cert = certify_mean(_stream(rng, 1200, annual_sharpe=0.8), return_bound=BOUND)
    assert cert.rejects(0.05)
    assert cert.stopping_time(0.05) is not None
    assert cert.growth_rate() > 0.0


@pytest.mark.unit
def test_a_negative_edge_is_never_certified() -> None:
    """One-sided by design: a factor that lost money is not evidence that it works."""
    for rep in range(20):
        rng = np.random.default_rng([47, rep])
        assert not certify_mean(_stream(rng, 1200, -0.8), return_bound=BOUND).rejects(0.05)


@pytest.mark.unit
def test_the_certificate_cannot_see_the_future() -> None:
    """Perturbing the last return cannot change wealth before that step."""
    rng = np.random.default_rng(53)
    x = _stream(rng, 400, 0.5)
    tampered = x.copy()
    tampered[-1] += 10.0 * SIGMA
    assert np.allclose(
        certify_mean(x, return_bound=BOUND).wealth[:-1],
        certify_mean(tampered, return_bound=BOUND).wealth[:-1],
    )


@pytest.mark.unit
def test_raw_wealth_is_the_process_and_wealth_is_its_running_maximum() -> None:
    """The distinction a diagnostic must respect; it was got wrong once already."""
    rng = np.random.default_rng(57)
    cert = certify_mean(_stream(rng, 1500, 0.6), return_bound=BOUND)
    assert np.allclose(cert.wealth, np.maximum.accumulate(cert.raw_wealth))
    assert np.all(np.diff(cert.wealth) >= -1e-12)
    assert np.any(np.diff(cert.raw_wealth) < 0.0)


@pytest.mark.unit
def test_an_envelope_violation_is_refused_rather_than_absorbed() -> None:
    """A fitted envelope would let the stake ceiling depend on the future."""
    with pytest.raises(ValueError, match="envelope"):
        certify_mean(np.array([0.1, 0.9]), return_bound=0.5)


@pytest.mark.unit
def test_rejects_malformed_input() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        certify_mean(np.zeros((3, 3)), return_bound=BOUND)
    with pytest.raises(ValueError, match="finite"):
        certify_mean(np.array([0.01, np.nan]), return_bound=BOUND)
    with pytest.raises(ValueError, match="cap"):
        certify_mean(np.zeros(5), return_bound=BOUND, cap=1.5)
    with pytest.raises(ValueError, match="return_bound"):
        certify_mean(np.zeros(5), return_bound=-1.0)


@pytest.mark.unit
def test_empty_input_yields_a_trivial_certificate() -> None:
    cert = certify_mean(np.zeros(0), return_bound=BOUND)
    assert cert.evalue == 1.0
    assert not cert.rejects(0.05)


@pytest.mark.unit
def test_the_interval_covers_the_truth_at_least_as_often_as_promised() -> None:
    """Time-uniform coverage: the true mean should escape at most alpha of the time."""
    reps = 200
    true_sharpe = 0.5
    truth = true_sharpe / np.sqrt(12.0) * SIGMA
    misses = 0
    for rep in range(reps):
        rng = np.random.default_rng([59, rep])
        interval = mean_ceiling(_stream(rng, 600, true_sharpe), return_bound=BOUND, alpha=0.05)
        misses += not (interval.lower <= truth <= interval.upper)
    assert misses / reps <= 0.10, f"coverage failed at {misses / reps:.3f}"


@pytest.mark.unit
def test_the_ceiling_tightens_as_data_arrives() -> None:
    """More evidence must narrow the bound; this is the whole point of reporting one."""
    rng = np.random.default_rng(61)
    x = _stream(rng, 4000, 0.0)
    assert (
        mean_ceiling(x, return_bound=BOUND).upper < mean_ceiling(x[:250], return_bound=BOUND).upper
    )


@pytest.mark.unit
def test_the_interval_brackets_the_sample_mean() -> None:
    rng = np.random.default_rng(67)
    x = _stream(rng, 800, 0.3)
    interval = mean_ceiling(x, return_bound=BOUND)
    assert interval.lower <= x.mean() <= interval.upper
    assert interval.lower < interval.upper


@pytest.mark.unit
def test_a_flat_stream_is_not_reported_as_an_edge() -> None:
    interval = mean_ceiling(np.zeros(500), return_bound=BOUND)
    assert not interval.excludes_zero()


@pytest.mark.unit
def test_sharpe_conversion_is_the_square_root_of_time_rule() -> None:
    interval = StreamCeiling(lower=-0.001, upper=0.004, alpha=0.05)
    assert interval.sharpe_ceiling(0.04, 12.0) == pytest.approx(0.004 / 0.04 * np.sqrt(12.0))
    with pytest.raises(ValueError, match="scale"):
        interval.sharpe_ceiling(0.0, 12.0)
    with pytest.raises(ValueError, match="periods_per_year"):
        interval.sharpe_ceiling(0.04, 0.0)


@pytest.mark.unit
def test_ceiling_rejects_malformed_input() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        mean_ceiling(np.zeros((2, 2)), return_bound=BOUND)
    with pytest.raises(ValueError, match="finite"):
        mean_ceiling(np.array([0.01, np.inf]), return_bound=BOUND)
    with pytest.raises(ValueError, match="alpha"):
        mean_ceiling(np.zeros(5), return_bound=BOUND, alpha=1.5)
