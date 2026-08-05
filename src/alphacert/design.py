"""How much data an alpha needs before it can be certified.

The certificate makes the sample-size question answerable in closed form, because the
evidence it accumulates is log wealth and log wealth grows at a rate set by the signal's
information ratio. A bettor staking the Kelly fraction on a signal whose per-period
information ratio is ``IR_p`` grows log wealth at ``IR_p^2 / 2`` nats per period, and
rejecting at level ``alpha`` needs ``ln(1 / alpha)`` nats. With ``P`` periods a year and
an annualised ratio ``IR = IR_p sqrt(P)``,

    T*  =  2 ln(1 / alpha) / IR^2   years.

At the 5% level that is ``6 / IR^2`` years: about six years to certify an information
ratio of 1, twenty-four for 0.5, sixty-seven for 0.3. The number is not an artifact of
this instrument. ``IR_p^2 / 2`` is the Kullback-Leibler rate separating the alternative
from the null, so by the standard sequential-testing bound no level-``alpha``
anytime-valid procedure can do materially better.

Three corrections matter in practice, and only one of them is favourable.

*Anytime-validity is not nearly free, and an earlier version of this module said it was.*
The comparison has to be made at matched power. A fixed-sample one-sided test needs
``(z_a + z_b)^2 / IR^2`` years. The e-process's log wealth at the oracle stake is
approximately ``N(x, 2x)`` with ``x = T * IR_p^2 / 2``, so attaining power ``beta`` needs
``x(beta) = ([z_b sqrt(2) + sqrt(2 z_b^2 + 4 ln(1/alpha))] / 2)^2``. The ratio of the two is
about **1.9 at 80% power** and 2.2 at median power -- anytime-validity costs roughly a factor
of two in data, not three per cent. The earlier claim compared the e-process's *median*
crossing time with the z-test's *80%-power* sample size, which are not the same quantity.
:func:`power_matched_horizon` computes the corrected number and
:func:`anytime_validity_cost` the ratio.

*The drift hedge costs less than ln 2 if you split capital unevenly.* The certificate runs a
signal bettor against a bettor who hedges the drift. An equal split discards ``ln 2 = 0.69``
nats of the signal bettor's wealth at the true drift, where the hedging bettor breaks even.
Weighting the signal by ``w`` costs only ``-ln w``; at ``w = 0.9`` that is 0.105 nats, and
measured power rises by a quarter to a half with no change in size.

*Not knowing your own Kelly fraction is not free.* If the stake must be learned online, log
wealth falls short of the oracle's by a regret term growing like ``(1/2) ln T``. At ``T``
around two thousand periods that is roughly 3.8 nats -- larger than the 3.0 nats needed to
reject at 5%. Pre-committing the stake to the smallest information ratio worth detecting
removes the term, and in simulation it lifts power at IR = 2 from 0.23 to 0.31.
"""

from __future__ import annotations

import math

from scipy import stats

#: Calendar convention for daily crypto, which trades every day.
CRYPTO_PERIODS_PER_YEAR = 365.0

#: Calendar convention for daily equities.
EQUITY_PERIODS_PER_YEAR = 252.0


def _validate(information_ratio: float, alpha: float, periods_per_year: float) -> None:
    if information_ratio <= 0:
        raise ValueError("information_ratio must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")


def detection_horizon(
    information_ratio: float,
    alpha: float = 0.05,
    periods_per_year: float = CRYPTO_PERIODS_PER_YEAR,
    *,
    kelly_known: bool = False,
    hedge: bool = True,
    max_periods: float = 1e9,
) -> float:
    """Years of data before a signal of this annualised information ratio can be certified.

    With ``kelly_known`` the stake is pre-committed and the answer is the closed form
    ``2 [ln(1/alpha) + ln 2] / IR^2``, the ``ln 2`` being the drift hedge. Otherwise the
    online learning regret ``(1/2) ln T`` is added and the implicit equation
    ``T IR^2 / (2 P) = ln(1/alpha) + ln 2 + (1/2) ln T`` is solved by fixed-point
    iteration. Set ``hedge=False`` for the information-theoretic law without this
    construction's overhead.
    """
    _validate(information_ratio, alpha, periods_per_year)
    budget = math.log(1.0 / alpha) + (math.log(2.0) if hedge else 0.0)
    rate = information_ratio**2 / (2.0 * periods_per_year)  # nats per period
    if kelly_known:
        return budget / rate / periods_per_year
    periods = budget / rate
    for _ in range(200):
        updated = (budget + 0.5 * math.log(max(periods, math.e))) / rate
        if abs(updated - periods) < 1e-6 * max(periods, 1.0):
            periods = updated
            break
        periods = min(updated, max_periods)
    return periods / periods_per_year


def fixed_sample_horizon(
    information_ratio: float,
    alpha: float = 0.05,
    power: float = 0.80,
    periods_per_year: float = CRYPTO_PERIODS_PER_YEAR,
) -> float:
    """Years needed by a one-sided fixed-sample test with a pre-committed size.

    The comparison that prices anytime-validity. This test may be looked at once, at a
    sample size fixed in advance; the certificate may be looked at every day forever.
    """
    _validate(information_ratio, alpha, periods_per_year)
    if not 0.0 < power < 1.0:
        raise ValueError("power must lie in (0, 1)")
    z = stats.norm.ppf(1.0 - alpha) + stats.norm.ppf(power)
    return float(z**2 / information_ratio**2)


def certifiable_ratio(
    years: float,
    alpha: float = 0.05,
    periods_per_year: float = CRYPTO_PERIODS_PER_YEAR,
    *,
    kelly_known: bool = False,
    hedge: bool = True,
) -> float:
    """Smallest annualised information ratio certifiable from this much data.

    The design question read backwards, and the one worth asking before a study starts:
    a sample of this length simply cannot speak to anything weaker.
    """
    if years <= 0:
        raise ValueError("years must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    periods = years * periods_per_year
    budget = math.log(1.0 / alpha) + (math.log(2.0) if hedge else 0.0)
    if not kelly_known:
        budget += 0.5 * math.log(max(periods, math.e))
    return float(math.sqrt(2.0 * budget * periods_per_year / periods))


def power_matched_horizon(
    information_ratio: float,
    alpha: float = 0.05,
    power: float = 0.80,
    periods_per_year: float = CRYPTO_PERIODS_PER_YEAR,
) -> float:
    """Years the certificate needs to attain ``power``, at the oracle stake.

    The honest comparison against :func:`fixed_sample_horizon`. Log wealth at the oracle Kelly
    stake is approximately ``N(x, 2x)`` with ``x = T IR_p^2 / 2``, so power ``beta`` requires
    ``x - z_beta sqrt(2x) = ln(1/alpha)``, a quadratic in ``sqrt(x)``.
    """
    _validate(information_ratio, alpha, periods_per_year)
    if not 0.0 < power < 1.0:
        raise ValueError("power must lie in (0, 1)")
    budget = math.log(1.0 / alpha)
    z = stats.norm.ppf(power)
    root = (z * math.sqrt(2.0) + math.sqrt(2.0 * z**2 + 4.0 * budget)) / 2.0
    return float(2.0 * root**2 / information_ratio**2)


def anytime_validity_cost(alpha: float = 0.05, power: float = 0.80) -> float:
    """Sample-size ratio, e-process to fixed-sample test, at matched level and power.

    Scale-free: the information ratio cancels. About 1.9 at the conventional settings, and
    the single number a practitioner needs in order to decide whether continuous monitoring
    is worth its price.
    """
    return float(power_matched_horizon(1.0, alpha, power) / fixed_sample_horizon(1.0, alpha, power))


def growth_to_ratio(
    growth_per_period: float, periods_per_year: float = CRYPTO_PERIODS_PER_YEAR
) -> float:
    """Annualised information ratio implied by an observed log-wealth growth rate.

    Inverts ``growth = IR_p^2 / 2``. Applied to a certificate's realised growth this
    reads the evidence back out in the units a desk budgets in.
    """
    if growth_per_period < 0:
        raise ValueError("growth_per_period must be non-negative")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    return float(math.sqrt(2.0 * growth_per_period * periods_per_year))
