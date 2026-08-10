"""One audited source for every detectable-effect number the paper quotes.

Two columns of an earlier table were generated from different quantities -- the threshold
from one span and the required-span column from a rounded information ratio -- so they did not
correspond. Everything is computed here, once, from the same expressions.

The quantity is a *minimum detectable effect* under a Gaussian constant-effect benchmark, not
a distribution-free impossibility bound. Under independent Gaussian strategy returns with mean
mu and known variance sigma^2, P observations per year and annualised ratio rho = sqrt(P)
mu/sigma, the one-sided z-test of H0: mu <= 0 at size alpha has power

    pi(rho, T) = Phi(rho sqrt(T) - z_{1-alpha}),

so the smallest ratio attaining target power pi after T years of *evaluation* is

    rho_MDE = [z_{1-alpha} + z_pi] / sqrt(T),

and the span needed against a given rho is [z_{1-alpha} + z_pi]^2 / rho^2.

The span that matters is the out-of-sample evaluation span, not the raw data span. The
Goyal-Welch data run 1926-2025, but the design reserves the first twenty years for
initialisation, so evaluation covers at most eighty years.

Usage: goyal_welch_mde.py
"""

from __future__ import annotations

import math

import pandas as pd
from scipy import stats

ALPHA = 0.05
POWER = 0.80
RAW_SPAN = 100.0
OOS_SPAN = 80.0
#: Campbell-Thompson's economically significant monthly out-of-sample R^2, and the annualised
#: ratio it implies under the small-effect mapping. Reported unrounded: rounding it to 0.25
#: moves the required span by four years.
CT_R2 = 0.005
CT_IR = math.sqrt(CT_R2 / (1 - CT_R2)) * math.sqrt(12)


def z_sum(alpha: float, power: float = POWER) -> float:
    return stats.norm.ppf(1.0 - alpha) + stats.norm.ppf(power)


def mde(span_years: float, alpha: float) -> float:
    """Smallest annualised information ratio detectable at ``POWER`` from this span."""
    return z_sum(alpha) / math.sqrt(span_years)


def span_needed(ir: float, alpha: float) -> float:
    """Evaluation years needed to detect ``ir`` at ``POWER``."""
    return z_sum(alpha) ** 2 / ir**2


def main() -> None:
    print(
        f"Campbell-Thompson threshold: monthly R^2_OS = {CT_R2:.3f} "
        f"-> annualised IR = {CT_IR:.4f} (not 0.25; rounding moves the span by ~4 years)\n"
        f"Raw data span {RAW_SPAN:.0f} y; out-of-sample evaluation span {OOS_SPAN:.0f} y "
        f"after the 20-year burn-in.\n"
    )
    rows = []
    for label, m in (
        ("one predictor", 1),
        ("Bonferroni over 17 (GW 2008)", 17),
        ("Bonferroni over 46 (GWZ 2024)", 46),
        ("Bonferroni over 100", 100),
    ):
        alpha = ALPHA / m
        rows.append(
            {
                "correction": label,
                "alpha": alpha,
                "mde_at_80y": mde(OOS_SPAN, alpha),
                "mde_at_100y": mde(RAW_SPAN, alpha),
                "years_needed_for_CT": span_needed(CT_IR, alpha),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv("audit/goyal_welch_mde.csv", index=False)
    print(table.to_string(index=False, float_format=lambda v: f"{v:10.4f}"))
    print(
        f"\nAgainst the {OOS_SPAN:.0f}-year evaluation span the design is short by "
        f"{span_needed(CT_IR, ALPHA) - OOS_SPAN:.0f} years for a single predictor and "
        f"{span_needed(CT_IR, ALPHA / 46) - OOS_SPAN:.0f} years after correcting for the "
        f"46-variable search."
    )
    print("\nwrote audit/goyal_welch_mde.csv")


if __name__ == "__main__":
    main()
