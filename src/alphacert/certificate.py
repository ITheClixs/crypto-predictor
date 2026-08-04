"""The drift-robust predictability certificate.

A *certificate* is a non-negative process ``E_t`` with ``E_0 = 1`` whose expectation
never exceeds 1 under the null "the signal carries no predictive information about the
outcome, whatever the outcome's unconditional drift". Ville's inequality then gives, for
every level ``alpha`` and *simultaneously at every t*,

    P( there exists t with E_t >= 1 / alpha )  <=  alpha.

So ``1 / max_t E_t`` is an anytime-valid p-value: the process may be monitored
continuously, stopped whenever the evidence looks good, and restarted, without any
multiplicity correction and without pre-committing a sample size.

Why a certificate rather than Clark-West
----------------------------------------
A nested predictive-accuracy test compares a model against a benchmark. When the model
carries an unpenalised intercept, the natural benchmark is an *estimated* mean, and the
test statistic inherits that estimate's error; when the benchmark is instead the zero
forecast, the statistic loads on the asset's drift and rejects for assets that merely
went up. Both failures come from the same source: the drift is a nuisance parameter and
the usual tests handle it by plugging in an estimate.

This module removes the drift instead of estimating it. For a fixed candidate drift
``mu`` two non-negative martingales are built,

``W_sig(mu)``
    the wealth of a bettor who stakes a predictable fraction on ``g_t phi((y_t - mu)/s_t)``
    being positive -- that is, on the signal genuinely leading the outcome;
``W_ctr(mu)``
    the wealth of a bettor who stakes on ``mu`` being the wrong centre.

Their average is a non-negative martingale for every ``mu``, and the certificate is its
infimum over ``mu``. At the true drift the second bettor makes nothing, so the infimum is
driven by the first; at a wrong drift the second bettor gets rich, so the infimum is not
dragged down by candidates the data have ruled out. The nuisance is eliminated, not
estimated, and the elimination costs one line of proof rather than a bootstrap.

What it costs
-------------
Nothing is refit. The certificate consumes the walk-forward forecasts a pipeline has
already produced and runs in one pass, so calibrating it does not require re-running the
pipeline thousands of times under a simulated null. That is the difference between an
instrument a desk can run every night and one that occupies a cluster for a weekend.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .payoffs import Payoff, get_payoff

#: Default half-width of the drift search. Daily log returns with a mean outside
#: +/- 5% per period are not a thing; the certificate is insensitive to widening it.
DEFAULT_DRIFT_BOUND = 0.05

#: Default drift grid spacing. The infimum over ``mu`` is a continuous one, evaluated
#: numerically; the grid must resolve the drift to better than the sampling error of the
#: mean, ``sigma / sqrt(n)``, or the numerical infimum sits above the true one and the
#: test stops being conservative. 1e-4 is safe for daily returns with ``sigma`` around
#: 3% out to n = 10^4. Use :func:`recommended_resolution` when in doubt.
DEFAULT_DRIFT_RESOLUTION = 1e-4

_EPS = 1e-12


def recommended_resolution(scale_hint: float, n: int, safety: float = 4.0) -> float:
    """Drift-grid spacing that resolves the mean to well inside its sampling error.

    ``scale_hint`` is an a-priori standard deviation for the outcome -- a-priori because
    a grid built from the sample would make the search set random, and the validity
    argument needs the true drift to lie in a set fixed in advance.
    """
    if scale_hint <= 0 or n <= 0:
        raise ValueError("scale_hint and n must be positive")
    return float(scale_hint / (safety * np.sqrt(n)))


def default_drift_grid(
    bound: float = DEFAULT_DRIFT_BOUND, resolution: float = DEFAULT_DRIFT_RESOLUTION
) -> np.ndarray:
    """Uniform drift grid on ``[-bound, bound]``, always containing 0."""
    if bound <= 0 or resolution <= 0:
        raise ValueError("bound and resolution must be positive")
    half = np.arange(resolution, bound + 0.5 * resolution, resolution)
    return np.concatenate([-half[::-1], [0.0], half])


@dataclass(frozen=True)
class Certificate:
    """Result of :func:`certify`: a wealth path plus the quantities read off it."""

    wealth: np.ndarray
    payoff: str
    drift_grid: np.ndarray = field(repr=False)
    n_bets: int = 0

    @property
    def evalue(self) -> float:
        """The certificate's terminal value; an e-value for the no-predictability null."""
        return float(self.wealth[-1]) if self.wealth.size else 1.0

    @property
    def log_wealth(self) -> np.ndarray:
        """Cumulative log wealth, in nats. Also the strategy's cumulative log return."""
        return np.log(self.wealth)

    @property
    def p_value(self) -> float:
        """Anytime-valid p-value ``1 / sup_t E_t``, valid at any stopping time."""
        peak = float(self.wealth.max()) if self.wealth.size else 1.0
        return float(min(1.0, 1.0 / peak)) if peak > 0 else 1.0

    def rejects(self, alpha: float = 0.05) -> bool:
        """Whether the certificate ever reached the ``1 / alpha`` threshold."""
        return self.p_value <= alpha

    def stopping_time(self, alpha: float = 0.05) -> int | None:
        """First index at which the threshold was crossed, or ``None``."""
        hit = np.flatnonzero(self.wealth >= 1.0 / alpha)
        return int(hit[0]) + 1 if hit.size else None

    def growth_rate(self) -> float:
        """Mean log wealth per period -- the drift-hedged strategy's log growth."""
        return float(self.log_wealth[-1] / self.wealth.size) if self.wealth.size else 0.0


def _shift(a: np.ndarray, fill: float) -> np.ndarray:
    """Lag by one so that every quantity used at ``t`` is known at ``t-1``."""
    out = np.empty_like(a)
    out[0] = fill
    out[1:] = a[:-1]
    return out


def _predictable_moments(outcome: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Running mean and standard deviation of the outcome, both lagged one step."""
    n = outcome.size
    t = np.arange(1, n + 1)
    mean = _shift(np.cumsum(outcome) / t, 0.0)
    var = _shift(np.cumsum((outcome - mean) ** 2) / t, 0.0)
    scale = np.maximum(np.sqrt(var), _EPS)
    scale[0] = max(float(np.abs(outcome[0])), _EPS)  # nothing known yet; no bet is placed
    return mean, scale


def _predictable_signal(signal: np.ndarray, centre: bool) -> np.ndarray:
    """Centre and standardise the signal using only its own past.

    Centring matters for power, not validity: a signal with a non-zero mean spends part
    of the bet on the drift, which the certificate hedges away by construction, so an
    uncentred signal simply wastes stake.
    """
    n = signal.size
    t = np.arange(1, n + 1)
    g = signal - _shift(np.cumsum(signal) / t, 0.0) if centre else signal
    rms = _shift(np.sqrt(np.cumsum(g**2) / t), 0.0)
    rms[0] = max(float(np.abs(g[0])), _EPS)
    return np.clip(g / np.maximum(rms, _EPS), -6.0, 6.0)


def _betting_fractions(psi: np.ndarray, envelope: np.ndarray, cap: float) -> np.ndarray:
    """Predictable Kelly plug-in ``sum(psi) / sum(psi^2)``, capped to keep wealth positive.

    The ratio is the empirical maximiser of expected log wealth for a small bet, computed
    from strictly past payoffs. It is not the oracle fraction, and the gap is a real cost:
    see :func:`alphacert.design.detection_horizon`, which prices it.
    """
    n = psi.size
    t = np.arange(1, n + 1)
    first = _shift(np.cumsum(psi), 0.0)
    second = _shift(np.cumsum(psi**2), 0.0)
    prior = _shift(np.cumsum(psi**2) / t, 1.0)  # keeps the ratio finite at the start
    raw = first / (second + prior)
    return np.clip(raw, 0.0, cap / np.maximum(envelope, _EPS))


def certify(
    signal: np.ndarray,
    outcome: np.ndarray,
    *,
    payoff: str | Payoff = "tanh",
    drift_grid: np.ndarray | None = None,
    drift_resolution: float = DEFAULT_DRIFT_RESOLUTION,
    return_bound: float = 0.5,
    centre_signal: bool = True,
    cap: float = 0.9,
    centring_fraction: float = 0.9,
    design_ratio: float | None = None,
    periods_per_year: float = 365.0,
    chunk: int = 256,
) -> Certificate:
    """Build a drift-robust certificate that ``signal`` predicts ``outcome``.

    Parameters
    ----------
    signal
        Predictable signal ``g_t``: any quantity computable from information available
        strictly before the outcome is realised. For a nested forecast comparison the
        natural choice is the model's forecast minus the intercept-only forecast, which
        is what makes the null "the features add nothing beyond the intercept". Validity
        does not depend on the choice; power does.
    outcome
        Realised outcome ``y_t``, aligned with ``signal``.
    payoff
        ``"tanh"`` (default), ``"sign"`` or ``"identity"``; see :mod:`alphacert.payoffs`.
        The first two assume the outcome is conditionally symmetric about its drift; the
        third assumes only a martingale difference and pays for it in power.
    drift_grid, drift_resolution
        The drift search. ``drift_resolution`` sets the spacing of the default grid and
        must sit well below the outcome's sampling error ``sigma / sqrt(n)``; a coarser
        grid does not merely lose precision, it lifts the numerical infimum above the
        true one and the process stops being conservative.
    return_bound
        A-priori envelope ``R`` with ``|y_t| <= R``. Used only by the identity payoff.
    cap, centring_fraction
        Stake ceilings, in ``(0, 1)``. Both exist to keep every wealth factor positive.
    design_ratio
        Annualised information ratio the test is designed to detect. When given, the
        stake is fixed at the Kelly fraction for that ratio instead of being learned
        online, which removes the ``(1/2) log T`` learning regret and roughly halves the
        data requirement -- at the price of losing power against signals much weaker or
        much stronger than the design point. Declaring the smallest edge worth finding
        before looking at the data is therefore a power decision, not a formality.
    periods_per_year
        Calendar convention used to turn ``design_ratio`` into a per-period stake.
    chunk
        Drift-grid columns processed at a time. Trades memory for speed only.

    Returns
    -------
    Certificate
        Whose ``wealth`` is non-decreasing by construction: ``E_t`` records the *running
        supremum* of the underlying martingale average, which is what Ville's inequality
        bounds and what an analyst who may stop at any time is entitled to use.
    """
    signal = np.asarray(signal, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    if signal.shape != outcome.shape:
        raise ValueError(f"signal {signal.shape} and outcome {outcome.shape} must align")
    if signal.ndim != 1:
        raise ValueError("signal and outcome must be one-dimensional")
    if not np.all(np.isfinite(signal)) or not np.all(np.isfinite(outcome)):
        raise ValueError("signal and outcome must be finite")
    if not 0.0 < cap < 1.0 or not 0.0 < centring_fraction < 1.0:
        raise ValueError("cap and centring_fraction must lie in (0, 1)")
    n = outcome.size
    if n == 0:
        return Certificate(np.ones(0), get_payoff(payoff).name, np.zeros(0), 0)

    phi = get_payoff(payoff)
    if not phi.bounded:
        reach = float(np.abs(outcome).max())
        if reach > return_bound:
            raise ValueError(
                f"return_bound={return_bound} is violated by an outcome of {reach:.4f}; the "
                "envelope must be set a priori and must actually hold -- at horizon h the "
                "outcome is an h-period return, so the envelope has to grow with h"
            )
    if drift_grid is None:
        grid = default_drift_grid(resolution=drift_resolution)
    else:
        grid = np.asarray(drift_grid, float)
    drift_bound = float(np.abs(grid).max()) if grid.size else 0.0

    mean, scale = _predictable_moments(outcome)
    g = _predictable_signal(signal, centre_signal)
    envelope = phi.envelope(scale, return_bound, drift_bound)

    # Stake sizing uses the running mean as a stand-in for the drift. That choice affects
    # only how much is bet, never whether the bet is fair, so it cannot break validity.
    psi_ref = g * phi((outcome - mean) / scale)
    stake_ceiling = cap / np.maximum(np.abs(g) * envelope, _EPS)
    if design_ratio is None:
        lam = _betting_fractions(psi_ref, np.abs(g) * envelope, cap)
    else:
        if design_ratio <= 0 or periods_per_year <= 0:
            raise ValueError("design_ratio and periods_per_year must be positive")
        lam = np.minimum(
            phi.kelly_constant * design_ratio / np.sqrt(periods_per_year), stake_ceiling
        )
    lam_ctr = centring_fraction / np.maximum(envelope, _EPS)

    running_min = np.full(n, np.inf)
    for start in range(0, grid.size, chunk):
        mus = grid[start : start + chunk][None, :]
        transformed = phi((outcome[:, None] - mus) / scale[:, None])
        log_signal = np.cumsum(np.log1p(lam[:, None] * g[:, None] * transformed), axis=0)
        stake = lam_ctr[:, None] * transformed
        up = np.cumsum(np.log1p(stake), axis=0)
        down = np.cumsum(np.log1p(-stake), axis=0)
        log_centring = np.logaddexp(up, down) - np.log(2.0)
        log_average = np.logaddexp(log_signal, log_centring) - np.log(2.0)
        running_min = np.minimum(running_min, log_average.min(axis=1))

    wealth = np.exp(np.maximum.accumulate(running_min))
    return Certificate(wealth, phi.name, grid, int(np.count_nonzero(lam)))
