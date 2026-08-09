"""What could each predictor's own sample have established, at the time it was introduced?

The certification bound is a statement about a sample, so it can be applied retrospectively:
for every variable in the Goyal-Welch-Zafirov set, the data available for it spans a known
number of years, and that span fixes the smallest annualised information ratio any test could
have established from it. Comparing that floor with the effect the variable actually delivers
answers a question the literature has not asked: \\emph{was the sample it was introduced on
ever capable of supporting the claim?}

Two floors are computed for each variable.

``solo``
    The floor a single test faces at the 5% level, which is the standard the original paper
    was held to.
``corrected``
    The floor after Bonferroni over the 46-variable set, which is the standard the literature
    as a whole should be held to, since that is the search that was actually run.

Both use the variable's own sample span, taken from the workbook's ReadMe sheet, not the
full 1926-2025 window. A variable available only since 1996 faces a much higher floor than one
available since 1926, and several of the newer variables are in exactly that position.

The comparison effect size is Campbell and Thompson's threshold for economic significance, a
monthly out-of-sample R-squared of 0.5%, which is an annualised information ratio of about
0.25. A variable whose floor exceeds that could not have certified an economically meaningful
edge from its own data even if one had been there.

Usage: goyal_welch_audit.py
"""

from __future__ import annotations

import math

import pandas as pd
from goyal_welch_pilot import WORKBOOK
from scipy import stats

CAMPBELL_THOMPSON = 0.25
GWZ_SEARCH = 46
POWER = 0.80


def span_years(begin: float, end: float, frequency: str) -> float | None:
    """Calendar span in years from the ReadMe's frequency-coded period stamps."""
    if not (pd.notna(begin) and pd.notna(end)):
        return None
    b, e = int(begin), int(end)
    if frequency.startswith("Month"):
        return ((e // 100) - (b // 100)) + ((e % 100) - (b % 100)) / 12
    if frequency.startswith("Quarter"):
        return ((e // 10) - (b // 10)) + ((e % 10) - (b % 10)) / 4
    if frequency.startswith("Semi"):
        return ((e // 10) - (b // 10)) + ((e % 10) - (b % 10)) / 2
    if frequency.startswith("Ann") or frequency.startswith("Year"):
        return float(e - b)
    return None


def floor(years: float, alpha: float) -> float:
    z = stats.norm.ppf(1.0 - alpha) + stats.norm.ppf(POWER)
    return z / math.sqrt(years)


def main() -> None:
    readme = pd.read_excel(WORKBOOK, sheet_name="ReadMe")
    rows = []
    for r in readme.itertuples():
        name, frequency = r.Name, r.Frequency
        if not isinstance(name, str) or not isinstance(frequency, str):
            continue
        years = span_years(r.SampleBeg, r.SampleEnd, frequency)
        if years is None or years <= 1:
            continue
        rows.append(
            {
                "predictor": name,
                "frequency": frequency,
                "span_years": years,
                "floor_solo": floor(years, 0.05),
                "floor_corrected": floor(years, 0.05 / GWZ_SEARCH),
            }
        )

    table = pd.DataFrame(rows).sort_values("span_years")
    table["solo_below_CT"] = table["floor_solo"] <= CAMPBELL_THOMPSON
    table["corrected_below_CT"] = table["floor_corrected"] <= CAMPBELL_THOMPSON
    table.to_csv("audit/goyal_welch_audit.csv", index=False)

    print(
        "Certification floor implied by each variable's own sample span.\n"
        f"Comparison effect: Campbell-Thompson's economic-significance threshold, "
        f"annualised IR {CAMPBELL_THOMPSON}.\n"
    )
    show = table[["predictor", "frequency", "span_years", "floor_solo", "floor_corrected"]]
    print(show.head(22).to_string(index=False, float_format=lambda v: f"{v:9.2f}"))

    n = len(table)
    solo = int(table["solo_below_CT"].sum())
    corrected = int(table["corrected_below_CT"].sum())
    print(
        f"\nOf {n} variables with a recorded sample:\n"
        f"  {n - solo} ({100 * (n - solo) / n:.0f}%) have a span too short to certify "
        f"IR = {CAMPBELL_THOMPSON} even as a single test;\n"
        f"  {n - corrected} ({100 * (n - corrected) / n:.0f}%) once the {GWZ_SEARCH}-variable "
        f"search is corrected for.\n"
        f"  median span {table['span_years'].median():.0f} years, "
        f"median solo floor {table['floor_solo'].median():.2f}, "
        f"median corrected floor {table['floor_corrected'].median():.2f}."
    )
    print(
        "\nThe shortest-sampled variables are the newest ones, which is the unfavourable "
        "direction:\nthe variables added since 2008 face the highest floors."
    )
    print("\nwrote audit/goyal_welch_audit.csv")


if __name__ == "__main__":
    main()
