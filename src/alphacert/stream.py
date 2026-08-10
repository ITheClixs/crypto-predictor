"""A certificate for a bare return stream.

:func:`alphacert.certify` tests whether a *signal* predicts an outcome, and eliminates the
outcome's drift as a nuisance. For an already-formed strategy -- a published factor's
long-short portfolio return, say -- there is no signal and no benchmark: the drift *is* the
estimand. The null is one-sided,

    H_0:  E[X_t | F_{t-1}]  <=  0,

and the certificate is a single betting martingale on the stream itself. No infimum over a
nuisance is needed, which makes this both simpler and strictly more powerful than routing a
constant signal through the general construction. It is also the honest parameterisation:
``certify`` centres its signal using the signal's own past, so a constant signal centres to
zero and places no bet at all.

What is assumed
---------------
Only that the stream is a martingale difference under the null and that an envelope ``B``
with ``|X_t| <= B`` is fixed *a priori*. No distributional assumption, no stationarity, no
bound on serial dependence in the volatility. The envelope is asserted rather than fitted,
and a violation raises rather than being silently absorbed -- a fitted envelope would make
the betting fractions depend on the future.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .certificate import _EPS, Certificate, _betting_fractions, _shift


def certify_mean(returns: np.ndarray, *, return_bound: float, cap: float = 0.9) -> Certificate:
    """Anytime-valid certificate for ``E[X_t] > 0`` on a bare return stream.

    Parameters
    ----------
    returns
        The strategy's realised per-period returns, in decimal units.
    return_bound
        An a-priori envelope: ``|X_t| <= return_bound`` must hold for every period. It is
        checked, not assumed, because a violated envelope invalidates the stake ceiling.
    cap
        Largest fraction of wealth the bettor may put at risk in one period. Affects power
        only; any predictable stake leaves the process a martingale under the null.

    Returns
    -------
    Certificate
        Whose ``wealth`` is the running maximum -- the object Ville's inequality bounds and
        the only one a test may use -- and whose ``raw_wealth`` is the underlying process,
        which is what any diagnostic must use.
    """
    x = np.asarray(returns, dtype=float)
    if x.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    if not np.all(np.isfinite(x)):
        raise ValueError("returns must be finite")
    if not 0.0 < cap < 1.0:
        raise ValueError("cap must lie in (0, 1)")
    if return_bound <= 0.0:
        raise ValueError("return_bound must be positive")
    if x.size == 0:
        return Certificate(np.ones(0), np.ones(0), "stream", np.zeros(0), 0)
    reach = float(np.abs(x).max())
    if reach > return_bound:
        raise ValueError(
            f"return_bound={return_bound} is violated by a return of {reach:.4f}; the "
            "envelope must be set a priori and must actually hold"
        )

    lam = _betting_fractions(x, np.full(x.size, return_bound), cap)
    running = np.cumsum(np.log1p(lam * x))
    raw = np.exp(running)
    wealth = np.exp(np.maximum.accumulate(running))
    return Certificate(wealth, raw, "stream", np.zeros(0), int(np.count_nonzero(lam)))


@dataclass(frozen=True)
class StreamCeiling:
    """Time-uniform confidence interval for the mean of a return stream."""

    lower: float
    upper: float
    alpha: float

    def sharpe_ceiling(self, scale: float, periods_per_year: float = 12.0) -> float:
        """Upper endpoint expressed as an annualised Sharpe ratio."""
        if scale <= 0:
            raise ValueError("scale must be positive")
        if periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")
        return float(self.upper / scale * np.sqrt(periods_per_year))

    def excludes_zero(self) -> bool:
        """Whether the interval rules out "this strategy is worth nothing"."""
        return self.lower > 0.0 or self.upper < 0.0


def _variance_adaptive_stake(centred: np.ndarray, reach: float, alpha: float) -> np.ndarray:
    """Predictable stakes set by the running variance, truncated for positivity.

    This is the Waudby-Smith--Ramdas predictable-mixture rule. It matters far more than it
    looks. A stake fixed at ``c / B`` makes the interval's width proportional to the
    a-priori envelope ``B``, so the ceiling would be a statement about the envelope rather
    than about the data -- and since one global envelope has to cover the single most
    extreme observation in the whole panel, that penalises every well-behaved series for an
    outlier belonging to a different one.

    Setting the stake from the running variance removes that coupling: the envelope enters
    only through a truncation that binds when the variance is small, and the leading term is
    the series' own scale. The stake is a function of strictly past data, so validity is
    untouched -- any predictable stake leaves the process a martingale under the null.
    """
    n = centred.size
    t = np.arange(1, n + 1)
    running_mean = _shift(np.cumsum(centred) / t, 0.0)
    running_var = _shift(np.cumsum((centred - running_mean) ** 2) / t, 0.25 * reach**2)
    variance = np.maximum(running_var, _EPS)
    scale = np.sqrt(2.0 * np.log(1.0 / alpha) / (variance * t * np.log1p(t)))
    return np.minimum(scale, 0.9 / max(reach, _EPS))


def _rules_out(x: np.ndarray, candidate: float, envelope: float, alpha: float) -> bool:
    """Does a two-sided betting martingale against ``E[X] = candidate`` ever reach 1/alpha?

    Two bettors stake in opposite directions on the centred stream and their wealths are
    averaged, so the pair is a non-negative martingale when the candidate is the truth.
    Stakes are predictable and variance-adaptive; see :func:`_variance_adaptive_stake` for
    why that choice, rather than a fixed fraction of the envelope, is load-bearing.
    """
    centred = x - candidate
    reach = max(envelope + abs(candidate), _EPS)
    stake = _variance_adaptive_stake(centred, reach, alpha)
    up = np.cumsum(np.log1p(stake * centred))
    down = np.cumsum(np.log1p(-stake * centred))
    peak = float(np.max(np.logaddexp(up, down) - np.log(2.0)))
    return peak >= np.log(1.0 / alpha)


def _search_edge(
    x: np.ndarray, inside: float, outside: float, envelope: float, alpha: float
) -> float:
    """Bisect between a retained and a rejected candidate.

    The retained set is an interval, so bisection finds its endpoint exactly; sweeping a
    grid would cost a factor of the grid size for a worse answer.
    """
    for _ in range(60):
        middle = 0.5 * (inside + outside)
        if _rules_out(x, middle, envelope, alpha):
            outside = middle
        else:
            inside = middle
    return 0.5 * (inside + outside)


def mean_ceiling(returns: np.ndarray, *, return_bound: float, alpha: float = 0.05) -> StreamCeiling:
    """Time-uniform interval for ``E[X_t]``, from which the Sharpe ceiling follows.

    The upper endpoint is the number this project exists to report: the largest mean the
    data still permit, valid at every stopping time, whether or not anything is rejected.
    """
    x = np.asarray(returns, dtype=float)
    if x.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    if not np.all(np.isfinite(x)):
        raise ValueError("returns must be finite")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if return_bound <= 0.0:
        raise ValueError("return_bound must be positive")
    if x.size == 0:
        return StreamCeiling(-return_bound, return_bound, alpha)

    centre = float(x.mean())
    if _rules_out(x, centre, return_bound, alpha):
        # The sample mean is itself excluded, which happens only for pathological streams.
        # Report a degenerate point rather than silently widening to something defensible.
        return StreamCeiling(centre, centre, alpha)
    return StreamCeiling(
        _search_edge(x, centre, centre - return_bound, return_bound, alpha),
        _search_edge(x, centre, centre + return_bound, return_bound, alpha),
        alpha,
    )
