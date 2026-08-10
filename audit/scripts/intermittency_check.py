"""Does the intermittency proposition hold, and where does its approximation break?

Proposition 3 claims that an effect present only a fraction ``phi`` of the time presents with
per-period information ratio

    rho_measured = sqrt(phi) * rho_on / sqrt(1 + (1 - phi) * rho_on**2),

which reduces to ``sqrt(phi) * rho_on`` for small ``rho_on``. Two things need checking and the
paper asserts both, so neither may be left to algebra alone:

1. the exact expression matches a simulation at every duty cycle, not just at convenient ones;
2. the approximation's error is negligible at the magnitudes the paper actually uses, and the
   paper's stated error figures at larger ratios are right.

Usage: intermittency_check.py
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

DRAWS = 4_000_000
SEED = 7


def exact(phi: float, rho_on: float) -> float:
    return math.sqrt(phi) * rho_on / math.sqrt(1.0 + (1.0 - phi) * rho_on**2)


def simulate(phi: float, rho_on: float, rng: np.random.Generator) -> float:
    """Per-period ratio of the payoff X_t = g_t y_t with g_t = m I_t, y_t = m I_t + eps."""
    sigma, m = 1.0, rho_on
    indicator = (rng.random(DRAWS) < phi).astype(float)
    outcome = m * indicator + sigma * rng.standard_normal(DRAWS)
    payoff = (m * indicator) * outcome
    return float(payoff.mean() / payoff.std(ddof=1))


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = []
    for phi in (0.1, 0.25, 0.4, 0.5, 0.75, 0.9, 1.0):
        for rho_on in (0.14, 0.58, 1.7):
            theory = exact(phi, rho_on)
            approx = math.sqrt(phi) * rho_on
            observed = simulate(phi, rho_on, rng)
            rows.append(
                {
                    "phi": phi,
                    "rho_on": rho_on,
                    "exact": theory,
                    "simulated": observed,
                    "rel_error_vs_sim": abs(observed - theory) / theory,
                    "approx": approx,
                    "approx_overstates_pct": 100.0 * (approx / theory - 1.0),
                }
            )

    table = pd.DataFrame(rows)
    table.to_csv("audit/intermittency_check.csv", index=False)
    print(table.to_string(index=False, float_format=lambda v: f"{v:10.4f}"))

    worst = table["rel_error_vs_sim"].max()
    print(f"\nWorst simulation-vs-exact relative error over all cells: {worst:.4%}")

    print("\nApproximation error by per-period on-period ratio, worst case over phi:")
    for rho_on, block in table.groupby("rho_on"):
        print(
            f"  rho_on = {rho_on:.2f}: approximation overstates by up to "
            f"{block['approx_overstates_pct'].max():.1f}%"
        )
    annual = 0.5
    print(
        f"\nAn annualised on-period ratio of {annual} is a per-period "
        f"{annual / math.sqrt(12):.3f} at monthly frequency, where the approximation error is "
        f"under {max(abs(100.0 * (math.sqrt(p) * 0.144 / exact(p, 0.144) - 1.0)) for p in (0.1, 0.4, 0.9)):.2f}%."
    )
    print("\nwrote audit/intermittency_check.csv")


if __name__ == "__main__":
    main()
