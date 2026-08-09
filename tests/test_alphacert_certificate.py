"""Tests for the certificate itself.

The properties worth testing are the ones the theory promises: the process is a
non-negative wealth path that starts at one, it cannot get rich on noise *at any drift*,
it does get rich on a real signal, and it never looks at the future.
"""

from __future__ import annotations

import numpy as np
import pytest

from alphacert import Certificate, certify, default_drift_grid, recommended_resolution
from alphacert.payoffs import available

SIGMA = 0.03
PERIODS = 365.0


def _series(rng: np.random.Generator, n: int, annual_sharpe: float = 0.0) -> np.ndarray:
    drift = annual_sharpe / np.sqrt(PERIODS) * SIGMA
    return drift + SIGMA * rng.standard_normal(n)


def _predictive_pair(
    rng: np.random.Generator, n: int, annual_ratio: float, annual_sharpe: float = 0.8
) -> tuple[np.ndarray, np.ndarray]:
    signal = annual_ratio / np.sqrt(PERIODS) * SIGMA * rng.standard_normal(n)
    return signal, _series(rng, n, annual_sharpe) + signal


@pytest.mark.unit
@pytest.mark.parametrize("payoff", available())
def test_wealth_is_a_non_decreasing_path_starting_at_one(payoff: str) -> None:
    rng = np.random.default_rng(0)
    cert = certify(rng.standard_normal(200), _series(rng, 200), payoff=payoff)
    assert isinstance(cert, Certificate)
    assert cert.wealth.shape == (200,)
    assert np.all(cert.wealth > 0.0)
    assert np.all(np.diff(cert.wealth) >= -1e-12)
    assert cert.wealth[0] <= 1.0 + 1e-9


@pytest.mark.unit
def test_p_value_is_the_reciprocal_of_peak_wealth() -> None:
    rng = np.random.default_rng(1)
    cert = certify(rng.standard_normal(300), _series(rng, 300))
    assert cert.p_value == pytest.approx(min(1.0, 1.0 / cert.wealth.max()))
    assert 0.0 < cert.p_value <= 1.0


@pytest.mark.unit
def test_noise_rejects_at_most_at_the_nominal_rate_whatever_the_drift() -> None:
    """The property Clark-West against a zero benchmark does not have.

    Clark-West against a zero forecast rejects more and more often as the asset's drift
    grows, because the statistic loads on that drift. The certificate's rejection rate is
    bounded by the nominal level uniformly in the drift, which is the whole point.
    """
    reps = 120
    for annual_sharpe in (0.0, 2.0, 5.0):
        rejections = 0
        for rep in range(reps):
            rng = np.random.default_rng([17, rep, int(annual_sharpe * 10)])
            cert = certify(rng.standard_normal(800), _series(rng, 800, annual_sharpe))
            rejections += cert.rejects(0.05)
        # 0.05 nominal plus three Monte Carlo standard errors at this replication count.
        assert rejections / reps <= 0.11, f"leaked at drift Sharpe {annual_sharpe}"


@pytest.mark.unit
def test_a_strong_signal_is_certified() -> None:
    rng = np.random.default_rng(3)
    signal, outcome = _predictive_pair(rng, 3000, annual_ratio=4.0)
    cert = certify(signal, outcome)
    assert cert.rejects(0.05)
    assert cert.stopping_time(0.05) is not None
    assert cert.growth_rate() > 0.0


@pytest.mark.unit
def test_reversing_the_sign_of_a_real_signal_destroys_the_certificate() -> None:
    """One-sided by design: a signal that predicts the wrong way is not evidence."""
    rng = np.random.default_rng(4)
    signal, outcome = _predictive_pair(rng, 3000, annual_ratio=4.0)
    assert not certify(-signal, outcome).rejects(0.05)


@pytest.mark.unit
def test_the_certificate_ignores_information_it_should_not_have() -> None:
    """Perturbing the last outcome cannot change wealth before that step."""
    rng = np.random.default_rng(5)
    signal, outcome = _predictive_pair(rng, 400, annual_ratio=3.0)
    base = certify(signal, outcome)
    tampered = outcome.copy()
    tampered[-1] += 10.0 * SIGMA
    assert np.allclose(base.wealth[:-1], certify(signal, tampered).wealth[:-1])


@pytest.mark.unit
def test_scaling_the_signal_leaves_the_certificate_unchanged() -> None:
    """The signal is standardised predictably, so only its shape can matter."""
    rng = np.random.default_rng(6)
    signal, outcome = _predictive_pair(rng, 800, annual_ratio=3.0)
    assert np.allclose(certify(signal, outcome).wealth, certify(37.0 * signal, outcome).wealth)


@pytest.mark.unit
def test_identity_payoff_is_valid_without_the_symmetry_assumption() -> None:
    """Skewed innovations break conditional symmetry; the identity payoff survives them."""
    rejections = 0
    for rep in range(40):
        rng = np.random.default_rng([23, rep])
        shocks = rng.chisquare(2, 1500) - 2.0  # mean zero, badly skewed
        outcome = 0.001 + SIGMA * shocks / np.sqrt(4.0)
        rejections += certify(
            rng.standard_normal(1500), outcome, payoff="identity", return_bound=1.0
        ).rejects(0.05)
    assert rejections == 0


@pytest.mark.unit
def test_rejects_mismatched_or_degenerate_input() -> None:
    rng = np.random.default_rng(7)
    with pytest.raises(ValueError, match="align"):
        certify(rng.standard_normal(10), rng.standard_normal(11))
    with pytest.raises(ValueError, match="one-dimensional"):
        certify(rng.standard_normal((4, 4)), rng.standard_normal((4, 4)))
    with pytest.raises(ValueError, match="finite"):
        certify(np.array([1.0, np.nan]), np.array([0.1, 0.2]))
    with pytest.raises(ValueError, match="cap"):
        certify(rng.standard_normal(10), rng.standard_normal(10), cap=1.5)
    with pytest.raises(ValueError, match="unknown payoff"):
        certify(rng.standard_normal(10), rng.standard_normal(10), payoff="quadratic")


@pytest.mark.unit
def test_empty_input_yields_a_trivial_certificate() -> None:
    cert = certify(np.zeros(0), np.zeros(0))
    assert cert.evalue == 1.0
    assert cert.p_value == 1.0
    assert not cert.rejects(0.05)


@pytest.mark.unit
def test_drift_grid_spans_zero_and_respects_its_resolution() -> None:
    grid = default_drift_grid(bound=0.01, resolution=1e-3)
    assert grid.min() == pytest.approx(-0.01)
    assert grid.max() == pytest.approx(0.01)
    assert 0.0 in grid
    assert np.allclose(np.diff(grid), 1e-3)
    with pytest.raises(ValueError):
        default_drift_grid(bound=-1.0)


@pytest.mark.unit
def test_recommended_resolution_shrinks_with_the_sample() -> None:
    assert recommended_resolution(0.03, 10_000) < recommended_resolution(0.03, 100)
    assert recommended_resolution(0.03, 2500) == pytest.approx(0.03 / (4 * 50))
    with pytest.raises(ValueError):
        recommended_resolution(0.0, 100)


@pytest.mark.unit
def test_a_grid_too_coarse_to_resolve_the_drift_is_the_documented_failure_mode() -> None:
    """Guards the one numerical trap: the infimum must actually bracket the true drift.

    With a grid whose spacing is far larger than the drift itself the numerical infimum
    sits above the true one and the process stops being conservative. This is why
    :func:`recommended_resolution` exists, and the test records the consequence of
    ignoring it rather than pretending it cannot happen.
    """
    coarse = np.array([-0.05, 0.0, 0.05])
    leaks = 0
    for rep in range(40):
        rng = np.random.default_rng([29, rep])
        leaks += certify(
            rng.standard_normal(2000), _series(rng, 2000, 6.0), drift_grid=coarse
        ).rejects(0.05)
    assert leaks > 0


@pytest.mark.unit
def test_raw_wealth_is_the_process_and_wealth_is_its_running_maximum() -> None:
    """Pins a distinction that is easy to misuse in a diagnostic.

    ``wealth`` is non-decreasing because Ville's inequality bounds the supremum, so asking
    "when was this rising?" of it answers "when did it set a new high", not "when was the
    signal contributing". A duty-cycle estimate built on the wrong one reports the wrong
    number, and did once.
    """
    rng = np.random.default_rng(31)
    signal, outcome = _predictive_pair(rng, 1500, annual_ratio=3.0)
    cert = certify(signal, outcome)
    assert cert.raw_wealth.shape == cert.wealth.shape
    assert np.allclose(cert.wealth, np.maximum.accumulate(cert.raw_wealth))
    assert np.all(np.diff(cert.wealth) >= -1e-12)
    # The underlying process genuinely goes down sometimes; the running maximum never does.
    assert np.any(np.diff(cert.raw_wealth) < 0)


@pytest.mark.unit
def test_raw_wealth_of_a_noise_signal_rises_about_half_the_time() -> None:
    """The sanity check that exposes the mistake: a driftless process is up half the time."""
    fractions = []
    for rep in range(20):
        rng = np.random.default_rng([53, rep])
        cert = certify(rng.standard_normal(1500), _series(rng, 1500, 0.8))
        steps = np.diff(np.log(cert.raw_wealth))
        moving = steps[np.abs(steps) > 1e-12]
        if moving.size > 100:
            fractions.append(float(np.mean(moving > 0)))
    assert 0.35 < float(np.mean(fractions)) < 0.65
