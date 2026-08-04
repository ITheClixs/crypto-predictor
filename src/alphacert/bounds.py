"""An anytime-valid ceiling on how much the features could be worth.

A certificate that fails to reject says the evidence is insufficient. That is the less
useful half of the answer. The half a desk can act on is the *upper* bound: given what
has been observed, how large could the incremental value still be? When the ceiling falls
below the cost of trading, the research line can be retired -- with a rigorous statement
rather than a shrug, and at any time, without pre-committing a sample size.

The estimand is

    theta  =  E[ z_t (y_t - mu) ],

the expected return of a position sized by the predictably standardised signal ``z_t``.
Because ``|z_t|`` is around one, ``theta / sigma_y`` is a per-period information ratio and
``sqrt(P) theta / sigma_y`` the annualised one, which is what
:meth:`ValueCeiling.ratio_ceiling` reports.

The construction reuses the certificate's device on a second axis. For each candidate
value ``theta`` and each candidate drift ``mu``, a two-sided betting martingale tests
``E[z_t (y_t - mu)] = theta``; it is averaged with the martingale that tests whether
``mu`` is the wrong centre, and ``theta`` survives when the infimum over ``mu`` of that
average never reaches ``1 / alpha``. The surviving set is an interval, so its endpoints
are found by bisection rather than by sweeping a grid.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .certificate import _EPS, _predictable_moments, _predictable_signal

#: Drift search for the ceiling. Narrower and coarser than the certificate's, because a
#: bound on the value only needs the drift resolved to its own sampling error.
DEFAULT_CEILING_DRIFT_BOUND = 0.02
DEFAULT_CEILING_DRIFT_RESOLUTION = 1e-4


@dataclass(frozen=True)
class ValueCeiling:
    """Time-uniform confidence interval for the incremental value ``theta``."""

    lower: float
    upper: float
    alpha: float
    point: float

    def ratio_ceiling(self, outcome_scale: float, periods_per_year: float = 365.0) -> float:
        """Upper bound expressed as an annualised information ratio."""
        if outcome_scale <= 0:
            raise ValueError("outcome_scale must be positive")
        if periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")
        return float(self.upper / outcome_scale * np.sqrt(periods_per_year))

    def excludes_zero(self) -> bool:
        """Whether the interval rules out "the features are worth nothing"."""
        return self.lower > 0.0 or self.upper < 0.0


def _ceiling_drift_grid() -> np.ndarray:
    half = np.arange(
        DEFAULT_CEILING_DRIFT_RESOLUTION,
        DEFAULT_CEILING_DRIFT_BOUND + 0.5 * DEFAULT_CEILING_DRIFT_RESOLUTION,
        DEFAULT_CEILING_DRIFT_RESOLUTION,
    )
    return np.concatenate([-half[::-1], [0.0], half])


def _predictable_stake(payoff: np.ndarray, envelope: np.ndarray, alpha: float) -> np.ndarray:
    """Waudby-Smith and Ramdas' predictable plug-in stake, capped for positivity."""
    n = payoff.size
    t = np.arange(1, n + 1)
    running_var = np.cumsum(payoff**2) / t
    predictable_var = np.empty(n)
    predictable_var[0] = float(np.mean(payoff**2)) if n else 1.0
    predictable_var[1:] = running_var[:-1]
    predictable_var = np.maximum(predictable_var, _EPS)
    ideal = np.sqrt(2.0 * np.log(2.0 / alpha) / (predictable_var * t * np.log(1.0 + t)))
    return np.minimum(ideal, 0.75 / np.maximum(envelope, _EPS))


def _rules_out(
    theta: float,
    centred: np.ndarray,
    stake: np.ndarray,
    log_centring: np.ndarray,
    log_threshold: float,
) -> bool:
    psi = centred - theta
    s = stake[:, None]
    plus = np.cumsum(np.log1p(s * psi), axis=0)
    minus = np.cumsum(np.log1p(-s * psi), axis=0)
    log_value = np.logaddexp(plus, minus) - np.log(2.0)
    log_average = np.logaddexp(log_value, log_centring) - np.log(2.0)
    return bool(np.max(np.min(log_average, axis=1)) >= log_threshold)


def _search_edge(
    inside: float,
    outside: float,
    tolerance: float,
    rules_out: Callable[[float], bool],
) -> float:
    """Bisect between a surviving value and an excluded one."""
    for _ in range(60):
        if abs(outside - inside) <= tolerance:
            break
        middle = 0.5 * (inside + outside)
        if rules_out(middle):
            outside = middle
        else:
            inside = middle
    return float(inside)


def value_ceiling(
    signal: np.ndarray,
    outcome: np.ndarray,
    *,
    alpha: float = 0.05,
    drift_grid: np.ndarray | None = None,
    return_bound: float = 0.5,
    centre_signal: bool = True,
    max_reach: float = 40.0,
) -> ValueCeiling:
    """Anytime-valid confidence interval for ``theta = E[z_t (y_t - mu)]``.

    The drift ``mu`` is eliminated exactly as in :func:`alphacert.certify`, so the bound
    holds whatever the asset's drift turns out to be. ``max_reach`` caps the search at
    that many standard errors from the sample value; a bound wider than that is reported
    as infinite rather than invented.
    """
    signal = np.asarray(signal, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    if signal.shape != outcome.shape or signal.ndim != 1:
        raise ValueError("signal and outcome must be aligned one-dimensional arrays")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    n = outcome.size
    if n == 0:
        raise ValueError("need at least one observation")
    reach = float(np.abs(outcome).max())
    if reach > return_bound:
        raise ValueError(
            f"return_bound={return_bound} is violated by an outcome of {reach:.4f}; the "
            "envelope must be set a priori and must actually hold -- at horizon h the "
            "outcome is an h-period return, so the envelope has to grow with h"
        )

    mean, scale = _predictable_moments(outcome)
    z = _predictable_signal(signal, centre_signal)
    drifts = _ceiling_drift_grid() if drift_grid is None else np.asarray(drift_grid, float)
    drift_bound = float(np.abs(drifts).max())

    deviation = (outcome[:, None] - drifts[None, :]) / scale[:, None]
    centring_envelope = (return_bound + drift_bound) / scale
    centring_stake = 0.9 / np.maximum(centring_envelope, _EPS)
    up = np.cumsum(np.log1p(centring_stake[:, None] * deviation), axis=0)
    down = np.cumsum(np.log1p(-centring_stake[:, None] * deviation), axis=0)
    log_centring = np.logaddexp(up, down) - np.log(2.0)

    centred = z[:, None] * (outcome[:, None] - drifts[None, :])
    sample = z * (outcome - mean)
    point = float(sample.mean())
    standard_error = float(sample.std(ddof=1) / np.sqrt(n)) if n > 1 else float("inf")
    envelope = np.abs(z) * (return_bound + drift_bound) + max_reach * abs(standard_error)
    stake = _predictable_stake(sample, envelope, alpha)
    log_threshold = np.log(1.0 / alpha)

    def rules_out(theta: float) -> bool:
        return _rules_out(theta, centred, stake, log_centring, log_threshold)

    if rules_out(point):  # the sample value itself is excluded: nothing to report
        return ValueCeiling(float("nan"), float("nan"), alpha, point)

    step = max(standard_error, _EPS)
    edges: list[float] = []
    for direction in (1.0, -1.0):
        far = point + direction * step
        reach = 1.0
        while not rules_out(far) and reach < max_reach:
            reach *= 2.0
            far = point + direction * reach * step
        if not rules_out(far):
            edges.append(direction * float("inf"))
            continue
        near = point + direction * (reach / 2.0) * step if reach > 1.0 else point
        edges.append(_search_edge(near, far, 1e-3 * step, rules_out))
    return ValueCeiling(min(edges[1], edges[0]), max(edges[0], edges[1]), alpha, point)
