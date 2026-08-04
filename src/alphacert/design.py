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

Two corrections matter in practice, and they point in opposite directions.

*The drift hedge costs ln 2.* The certificate splits capital between the bettor who
stakes on the signal and the bettor who hedges the drift, so half of any terminal wealth
is the signal bettor's. That is ``ln 2 = 0.69`` nats on a budget of ``ln(1/alpha) = 3.00``
at the 5% level: eliminating the nuisance rather than estimating it costs 23% more data,
and buys exact validity at every drift. ``hedge`` controls whether the term is included.

*Anytime-validity is nearly free.* The fixed-sample benchmark -- a one-sided test with a
pre-committed sample size, looked at exactly once -- needs ``(z_alpha + z_beta)^2 / IR^2``
years, which at 5% and 80% power is ``6.18 / IR^2``. Against ``5.99 / IR^2`` for the
certificate, continuous monitoring costs about 3% more data and buys the right to stop
whenever the evidence arrives.

*Not knowing your own Kelly fraction is not free.* If the stake must be learned online,
log wealth falls short of the oracle's by a regret term that grows like ``(1/2) ln T``.
At ``T`` around two thousand periods that is roughly 3.8 nats -- larger than the 3.0 nats
needed to reject at 5%. The cost of not knowing how strong your signal is can exceed the
entire evidentiary budget, and it multiplies the horizon by two to three.
Pre-committing the stake to the smallest information ratio worth detecting removes the
term entirely, which makes
"declare the alpha you care about before you look" a power decision rather than a
methodological nicety.
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
