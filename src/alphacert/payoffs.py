"""Bounded odd transforms of the forecast deviation.

The certificate bets on ``psi_t = g_t * phi((y_t - mu) / s_t)`` where ``g_t`` is a
predictable signal, ``mu`` the unknown drift and ``s_t`` a predictable scale. Validity
needs exactly two things from ``phi``:

1. ``E[phi((y_t - mu) / s_t) | F_{t-1}] = 0`` under the null;
2. ``|phi| <= bound``, so a predictable cap on the bet keeps the wealth positive.

Two families satisfy (1), under different assumptions, and the choice is the single
place where the user trades assumptions for power:

``IDENTITY``
    ``phi(u) = u``. Needs only a martingale-difference null, ``E[y_t | F_{t-1}] = mu``.
    Unbounded in principle, so the bet must be capped using an a-priori envelope ``R``
    on ``|y_t|``; when ``R`` is much larger than the outcome's standard deviation, that
    cap costs power. Use when conditional symmetry is not defensible.

``TANH`` and ``SIGN``
    ``phi(u) = tanh(u)`` and ``phi(u) = sign(u)``. Odd and bounded by 1, so ``E[phi] = 0``
    holds whenever ``y_t - mu`` is *conditionally symmetric* about zero. That is stronger
    than a martingale difference but far weaker than i.i.d.: it permits arbitrary
    volatility clustering, fat tails, and any dependence in ``|y_t - mu|``. Because the
    bound is 1 rather than ``R / s_t``, the bet is not throttled and the growth rate is
    close to Kelly-optimal.

``SIGN`` discards magnitude and tests directional predictability only; it is the
anytime-valid replacement for a Pesaran-Timmermann market-timing test, and it inherits
that test's interpretation without inheriting its dependence on an asymptotic variance.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Payoff:
    """An odd transform together with a *predictable* bound on its absolute value.

    :meth:`envelope` returns, for each step, a number ``B_t`` known at ``t-1`` with
    ``|phi_t| <= B_t`` for every admissible outcome and every drift in the search range.
    It is what lets the caller cap the betting fraction at ``c / (|g_t| B_t)`` and
    guarantee a strictly positive wealth factor without ever looking at the realised
    outcome. Bounded payoffs have ``B_t = 1``; the identity payoff has
    ``B_t = (R + mu_max) / s_t``, which is why a loose return envelope ``R`` costs power.
    """

    name: str
    fn: Callable[[np.ndarray], np.ndarray]
    requires_symmetry: bool
    bounded: bool
    kelly_constant: float

    def __call__(self, standardised_deviation: np.ndarray) -> np.ndarray:
        return self.fn(standardised_deviation)

    def envelope(self, scale: np.ndarray, return_bound: float, drift_bound: float) -> np.ndarray:
        """Per-step bound on ``|phi_t|``, measurable with respect to ``F_{t-1}``."""
        if self.bounded:
            return np.ones_like(scale)
        return (return_bound + drift_bound) / scale


# Kelly constants. For a unit-variance standardised signal ``z`` and a per-period information
# ratio ``r``, the log-optimal stake is ``lambda* = E[psi] / E[psi^2]`` with
# ``psi = z phi((y - mu) / s)``. Expanding to first order in ``r`` and taking the innovation
# standard normal, ``E[psi] = r E[phi'(u)]`` and ``E[psi^2] = E[phi(u)^2]``, so
# ``lambda* = c r`` with ``c = E[phi'(u)] / E[phi(u)^2]``. The three constants below are that
# ratio: 1 for the identity, ``E[sech^2 u] / E[tanh^2 u] = 0.644 / 0.418`` for tanh, and
# ``E[2 delta] / 1 = sqrt(2 / pi)`` for the sign. They set the stake when a design ratio is
# declared in advance; they affect power only, never validity.
IDENTITY = Payoff(
    "identity", lambda u: u, requires_symmetry=False, bounded=False, kelly_constant=1.0
)
TANH = Payoff("tanh", np.tanh, requires_symmetry=True, bounded=True, kelly_constant=1.54)
SIGN = Payoff(
    "sign",
    np.sign,
    requires_symmetry=True,
    bounded=True,
    kelly_constant=float(np.sqrt(2.0 / np.pi)),
)

_REGISTRY: dict[str, Payoff] = {p.name: p for p in (IDENTITY, TANH, SIGN)}


def get_payoff(name: str | Payoff) -> Payoff:
    """Look a payoff up by name, or pass a :class:`Payoff` through unchanged."""
    if isinstance(name, Payoff):
        return name
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown payoff {name!r}; expected one of {sorted(_REGISTRY)}") from None


def available() -> tuple[str, ...]:
    """Names of the registered payoffs."""
    return tuple(sorted(_REGISTRY))
