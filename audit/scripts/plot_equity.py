"""The two figures the equity-premium paper turns on.

*Left.* The certification floor as a function of calendar span and of the size of the search.
A single predictor at the 5% level needs 99 years to reach 80% power against the effect
Campbell and Thompson call economically significant; correcting for the 46-predictor search
Goyal, Welch and Zafirov actually ran needs 244. The dataset is 100 years old. The literature
has been operating with no margin, and with the correction, below the line entirely.

*Right.* Every predictor-frequency pair in the study, plotted as its anytime-valid ceiling
against its out-of-sample R-squared. The ceiling is the number a reader can act on: it says how
large the incremental information ratio could still be, having seen a century. None of the
intervals excludes zero, and the median ceiling sits at roughly twice the effect size the
literature treats as economically meaningful -- so the data cannot separate "nothing" from
"twice what matters".

Usage: plot_equity.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from cryptoforecast.plots.style import NEGATIVE, PALETTE, POSITIVE, REFERENCE_BLACK, finish

OUT = Path("reports/figures/fig12_equity.png")
#: Campbell and Thompson's monthly out-of-sample R^2 of 0.5%, as an annualised ratio.
CAMPBELL_THOMPSON = 0.25
SPAN = 100.0


def floor_at(years: float, alpha: float, power: float = 0.80) -> float:
    """Smallest annualised information ratio a fixed-sample test can reach at this power."""
    z = stats.norm.ppf(1 - alpha) + stats.norm.ppf(power)
    return z / math.sqrt(years)


def main() -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))

    left = axes[0]
    years = np.linspace(10, 400, 300)
    searches = ((1, "one predictor"), (17, "17 (GW 2008)"), (46, "46 (GWZ 2024)"))
    for i, (m, label) in enumerate(searches):
        left.plot(
            years, [floor_at(y, 0.05 / m) for y in years],
            color=PALETTE[i], lw=1.7, label=f"Bonferroni over {label}",
        )
    left.axhline(CAMPBELL_THOMPSON, color=NEGATIVE, ls="--", lw=1.4)
    left.annotate(
        "Campbell-Thompson 'economically significant' (IR 0.25)",
        (400, CAMPBELL_THOMPSON * 1.06), ha="right", fontsize=7.5, color=NEGATIVE,
    )
    left.axvline(SPAN, color=REFERENCE_BLACK, ls=":", lw=1.2)
    left.annotate(
        f"the data: {SPAN:.0f} years", (SPAN * 1.05, 1.05), fontsize=7.5,
        color=REFERENCE_BLACK,
    )
    left.set_xscale("log")
    left.set_yscale("log")
    left.set_xlabel("Calendar span of the sample (years)")
    left.set_ylabel("Smallest certifiable annualised IR")
    left.set_title("What a sample of a given length can settle")
    left.legend(loc="upper right")

    right = axes[1]
    table = pd.read_csv("audit/goyal_welch_all.csv")
    table = table[~table["degenerate_fit"]]
    for i, (freq, group) in enumerate(table.groupby("frequency")):
        right.scatter(
            group["r2_oos_pct"].clip(lower=-8), group["ir_ceiling"],
            s=22, alpha=0.75, color=PALETTE[i], label=freq, edgecolor="none",
        )
    right.axhline(
        CAMPBELL_THOMPSON, color=NEGATIVE, ls="--", lw=1.4,
    )
    right.annotate(
        "IR 0.25", (-7.8, CAMPBELL_THOMPSON * 1.08), fontsize=7.5, color=NEGATIVE,
    )
    median = float(table["ir_ceiling"].median())
    right.axhline(median, color=POSITIVE, lw=1.3)
    right.annotate(
        f"median ceiling {median:.2f}", (-7.8, median * 1.05), fontsize=7.5, color=POSITIVE,
    )
    right.axvline(0.0, color=REFERENCE_BLACK, lw=0.8, alpha=0.5)
    right.set_xlabel("Out-of-sample $R^2$ (%), clipped at $-8$")
    right.set_ylabel("Anytime-valid ceiling on annualised IR")
    right.set_title(f"All {len(table)} predictors, and how much they could still be worth")
    right.legend(loc="upper left")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    finish(fig, OUT)
    print(f"wrote {OUT} and {OUT.with_suffix('.pdf')}")
    print(
        f"  ceilings: median {median:.2f}, "
        f"{int((table['ir_ceiling'] > CAMPBELL_THOMPSON).sum())} of {len(table)} above "
        f"Campbell-Thompson's threshold, so the data cannot rule out an economically "
        f"meaningful edge for those."
    )


if __name__ == "__main__":
    main()
