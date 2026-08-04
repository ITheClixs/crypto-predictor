"""Tests for the sample-size law and for the payoff registry."""

from __future__ import annotations

import numpy as np
import pytest

from alphacert import (
    IDENTITY,
    SIGN,
    TANH,
    available,
    certifiable_ratio,
    detection_horizon,
    fixed_sample_horizon,
    get_payoff,
    growth_to_ratio,
)


@pytest.mark.unit
def test_the_closed_form_is_two_log_one_over_alpha_over_ir_squared() -> None:
    """Without the drift hedge, the information-theoretic law."""
    for ratio in (0.5, 1.0, 2.0):
        expected = 2.0 * np.log(20.0) / ratio**2
        assert detection_horizon(ratio, kelly_known=True, hedge=False) == pytest.approx(expected)


@pytest.mark.unit
def test_the_drift_hedge_costs_exactly_log_two_nats() -> None:
    """Splitting capital with the centring bettor is 0.69 nats on a budget of 3.00."""
    for ratio in (0.5, 1.0, 2.0):
        plain = detection_horizon(ratio, kelly_known=True, hedge=False)
        hedged = detection_horizon(ratio, kelly_known=True, hedge=True)
        assert hedged / plain == pytest.approx((np.log(20.0) + np.log(2.0)) / np.log(20.0))
        assert hedged / plain == pytest.approx(1.23, abs=0.01)


@pytest.mark.unit
def test_six_over_ir_squared_years_at_the_five_percent_level() -> None:
    """The number a desk can carry in its head."""
    assert detection_horizon(1.0, kelly_known=True, hedge=False) == pytest.approx(5.99, abs=0.01)
    assert detection_horizon(0.5, kelly_known=True, hedge=False) == pytest.approx(23.97, abs=0.05)


@pytest.mark.unit
def test_learning_the_kelly_fraction_online_roughly_doubles_the_horizon() -> None:
    for ratio in (0.5, 1.0, 2.0, 3.0):
        known = detection_horizon(ratio, kelly_known=True)
        learned = detection_horizon(ratio, kelly_known=False)
        assert 1.7 < learned / known < 3.0


@pytest.mark.unit
def test_anytime_validity_costs_only_a_few_percent_against_a_fixed_sample_test() -> None:
    """5.99 versus 6.18 years at IR = 1: continuous monitoring is nearly free.

    The comparison is made without the drift hedge, because the fixed-sample benchmark does
    not hedge the drift either -- it assumes the benchmark is known.
    """
    sequential = detection_horizon(1.0, kelly_known=True, hedge=False)
    fixed = fixed_sample_horizon(1.0, alpha=0.05, power=0.80)
    assert fixed == pytest.approx(6.18, abs=0.01)
    assert 0.95 < sequential / fixed < 1.0


@pytest.mark.unit
def test_horizon_and_certifiable_ratio_are_inverses() -> None:
    for ratio in (0.4, 1.0, 2.5):
        for known in (True, False):
            for hedge in (True, False):
                years = detection_horizon(ratio, kelly_known=known, hedge=hedge)
                assert certifiable_ratio(years, kelly_known=known, hedge=hedge) == pytest.approx(
                    ratio, rel=1e-3
                )


@pytest.mark.unit
def test_a_seven_year_daily_sample_cannot_speak_to_a_weak_signal() -> None:
    """The design fact that governs every study of this length."""
    floor = certifiable_ratio(7.5, kelly_known=False)
    assert floor > 1.0
    assert certifiable_ratio(5.99, kelly_known=False) == pytest.approx(1.59, abs=0.01)


@pytest.mark.unit
def test_growth_rate_maps_back_to_an_information_ratio() -> None:
    ratio = 1.6
    growth = (ratio / np.sqrt(365.0)) ** 2 / 2.0
    assert growth_to_ratio(growth) == pytest.approx(ratio)


@pytest.mark.unit
def test_design_helpers_validate_their_arguments() -> None:
    with pytest.raises(ValueError):
        detection_horizon(0.0)
    with pytest.raises(ValueError):
        detection_horizon(1.0, alpha=0.0)
    with pytest.raises(ValueError):
        fixed_sample_horizon(1.0, power=1.0)
    with pytest.raises(ValueError):
        certifiable_ratio(0.0)
    with pytest.raises(ValueError):
        growth_to_ratio(-1.0)


@pytest.mark.unit
def test_payoff_registry_exposes_the_three_transforms() -> None:
    assert available() == ("identity", "sign", "tanh")
    assert get_payoff("tanh") is TANH
    assert get_payoff(SIGN) is SIGN
    assert TANH.requires_symmetry and SIGN.requires_symmetry
    assert not IDENTITY.requires_symmetry
    with pytest.raises(ValueError, match="unknown payoff"):
        get_payoff("gaussian")


@pytest.mark.unit
def test_bounded_payoffs_have_a_unit_envelope_and_identity_does_not() -> None:
    scale = np.full(3, 0.02)
    assert np.allclose(TANH.envelope(scale, 0.5, 0.05), 1.0)
    assert np.allclose(IDENTITY.envelope(scale, 0.5, 0.05), 0.55 / 0.02)
    assert np.all(np.abs(TANH(np.linspace(-10, 10, 21))) <= 1.0)
    assert np.all(np.abs(SIGN(np.linspace(-10, 10, 21))) <= 1.0)
