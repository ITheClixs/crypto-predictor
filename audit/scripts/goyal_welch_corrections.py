"""Is the audit's headline an artifact of choosing Bonferroni?

Bonferroni is the most conservative multiplicity adjustment in common use, so the claim that
no variable's own sample could have certified an economically significant effect deserves a
check against the alternatives. It survives them, for a structural reason: the floor asks
whether *any* variable could clear the bar, and at the first rejection Holm and
Benjamini-Hochberg both use the same alpha/m threshold Bonferroni uses throughout. The three
therefore agree exactly on the quantity that matters here.

Usage: goyal_welch_corrections.py
"""

from __future__ import annotations

import math

import pandas as pd
from goyal_welch_audit import CAMPBELL_THOMPSON, POWER, WORKBOOK, span_years
from scipy import stats

SEARCH = 46


def floor(years: float, alpha: float) -> float:
    return (stats.norm.ppf(1.0 - alpha) + stats.norm.ppf(POWER)) / math.sqrt(years)


def main() -> None:
    readme = pd.read_excel(WORKBOOK, sheet_name="ReadMe")
    spans = []
    for r in readme.itertuples():
        if not (isinstance(r.Name, str) and isinstance(r.Frequency, str)):
            continue
        years = span_years(r.SampleBeg, r.SampleEnd, r.Frequency)
        if years and years > 1:
            spans.append(years)
    span = pd.Series(spans)

    schemes = (
        ("none (single test)", 0.05),
        ("alpha = 0.01, still a single test", 0.01),
        ("Benjamini-Hochberg, first rejection", 0.05 / SEARCH),
        ("Holm, first rejection", 0.05 / SEARCH),
        ("Bonferroni over 46", 0.05 / SEARCH),
        ("Bonferroni over 100", 0.05 / 100),
    )
    rows = []
    for label, alpha in schemes:
        floors = span.map(lambda y, a=alpha: floor(y, a))
        rows.append(
            {
                "correction": label,
                "alpha": alpha,
                "median_floor": float(floors.median()),
                "below_threshold": int((floors <= CAMPBELL_THOMPSON).sum()),
                "n": len(floors),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv("audit/goyal_welch_corrections.csv", index=False)
    print(
        "Variables whose own sample could certify an annualised information ratio of "
        f"{CAMPBELL_THOMPSON}.\nAt the first rejection Holm and Benjamini-Hochberg use the "
        "same alpha/m as Bonferroni, so the\nthree agree exactly.\n"
    )
    print(table.to_string(index=False, float_format=lambda v: f"{v:9.5f}"))
    plain = int((span.map(lambda y: floor(y, 0.05)) > CAMPBELL_THOMPSON).sum())
    print(
        f"\nWith no multiplicity correction at all, {plain} of {len(span)} "
        f"({100 * plain / len(span):.0f}%) still have samples too short."
    )
    print("\nwrote audit/goyal_welch_corrections.csv")


if __name__ == "__main__":
    main()
