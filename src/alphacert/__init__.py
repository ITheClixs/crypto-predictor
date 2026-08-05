"""alphacert -- anytime-valid certificates of incremental predictive ability.

The problem this package solves is narrow and common. A forecasting pipeline produces
out-of-sample predictions; the analyst wants to know whether the *features* carry
information, as opposed to the model having rediscovered the asset's drift. Standard
nested tests answer a subtly different question, because they handle the drift by
plugging in an estimate of it, and on assets that went up the estimate is exactly what
they end up rewarding.

A certificate eliminates the drift instead. It is a wealth process: a bettor who is
allowed to stake on the signal leading the outcome, and who is simultaneously forced to
hedge every candidate drift. If the features are noise the bettor cannot get rich, at
any drift, in finite samples, under arbitrary heteroskedasticity and serial dependence
in volatility. So wealth *is* evidence, denominated in the same units as the alpha it
claims to have found.

    >>> import numpy as np
    >>> from alphacert import certify
    >>> rng = np.random.default_rng(0)
    >>> y = 0.001 + 0.03 * rng.standard_normal(2000)      # drifting, unpredictable
    >>> cert = certify(rng.standard_normal(2000), y)      # a signal that is pure noise
    >>> cert.rejects(0.05)
    False

Nothing is refit, so the whole thing runs in one pass over forecasts a pipeline has
already produced.
"""

from .bounds import value_ceiling
from .certificate import (
    Certificate,
    certify,
    default_drift_grid,
    recommended_resolution,
)
from .design import (
    anytime_validity_cost,
    certifiable_ratio,
    detection_horizon,
    fixed_sample_horizon,
    growth_to_ratio,
    power_matched_horizon,
)
from .merge import (
    certify_overlapping,
    e_bh,
    merge_average,
    merge_product,
    phase_indices,
)
from .payoffs import IDENTITY, SIGN, TANH, Payoff, available, get_payoff

__all__ = [
    "IDENTITY",
    "SIGN",
    "TANH",
    "Certificate",
    "Payoff",
    "anytime_validity_cost",
    "available",
    "certifiable_ratio",
    "certify",
    "certify_overlapping",
    "default_drift_grid",
    "detection_horizon",
    "e_bh",
    "fixed_sample_horizon",
    "get_payoff",
    "growth_to_ratio",
    "merge_average",
    "merge_product",
    "phase_indices",
    "power_matched_horizon",
    "recommended_resolution",
    "value_ceiling",
]
