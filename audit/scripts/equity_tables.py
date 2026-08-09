"""Emit the equity paper's LaTeX tables from the committed result files.

Generated rather than transcribed, so the manuscript cannot quote a number the analysis does
not produce.

Each file is a *complete* ``tabular`` environment rather than a fragment of rows. That is not
a style preference: ``\\input`` of bare rows inside a ``tabular`` raises "Misplaced \\noalign"
at the following rule, whatever the rows contain, because the file boundary interrupts the
alignment. Emitting the whole environment sidesteps it and leaves the manuscript to place only
the float.

Usage: equity_tables.py
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from scipy import stats

OUT = Path("paper-equity/tables")
CAMPBELL_THOMPSON = 0.25
GWZ_SEARCH = 46


def emit(path: Path, spec: str, header: str, body: str) -> None:
    """Write a complete tabular environment."""
    path.write_text(
        "\\begin{tabular}{" + spec + "}\n"
        "    \\toprule\n"
        f"    {header} \\\\\n"
        "    \\midrule\n"
        f"{body.rstrip()}\n"
        "    \\bottomrule\n"
        "\\end{tabular}\n"
    )


def escape(name: str) -> str:
    return name.replace("_", r"\_")


def years_needed(ir_annual: float, alpha: float, power: float = 0.80) -> float:
    """Calendar span for a one-sided fixed-sample test to reach ``power`` against ``ir``."""
    z = stats.norm.ppf(1.0 - alpha) + stats.norm.ppf(power)
    return z**2 / ir_annual**2


def full_results() -> str:
    table = pd.read_csv("audit/goyal_welch_all.csv").sort_values(
        ["frequency", "evalue"], ascending=[True, False]
    )
    lines: list[str] = []
    for frequency, group in table.groupby("frequency"):
        lines.append(rf"    \multicolumn{{7}}{{l}}{{\emph{{{frequency}}}}} \\")
        for row in group.itertuples():
            flag = r"$^\dagger$" if row.degenerate_fit else ""
            lines.append(
                f"    {escape(row.predictor)}{flag} & {row.source} & {row.n} & "
                f"${row.r2_oos_pct:.2f}$ & ${row.clark_west:.2f}$ & "
                f"${row.evalue:.2f}$ & ${row.ir_ceiling:.2f}$ \\\\"
            )
        lines.append(r"    \midrule")
    return "\n".join(lines[:-1])


def economic_translation() -> str:
    """Monthly out-of-sample R^2, as the literature reports it, in every other unit."""
    lines = []
    for r2 in (0.001, 0.002, 0.005, 0.010, 0.020):
        per_period = math.sqrt(r2 / (1 - r2))
        annual = per_period * math.sqrt(12)
        lines.append(
            f"    ${100 * r2:.1f}\\%$ & ${per_period:.3f}$ & ${annual:.2f}$ & "
            f"${years_needed(annual, 0.05):.0f}$ & "
            f"${years_needed(annual, 0.05 / GWZ_SEARCH):.0f}$ \\\\"
        )
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    emit(
        OUT / "full_results.tex",
        "llrrrrr",
        r"Predictor & Source & $n$ & $\rsq$ (\%) & Clark--West & e-value & Ceiling",
        full_results(),
    )
    emit(
        OUT / "economic_translation.tex",
        "rrrrr",
        r"Monthly $\rsq$ & Per-period $\ir$ & Annualised $\ir$ & Years, one test "
        r"& Years, 46 tests",
        economic_translation(),
    )
    table = pd.read_csv("audit/goyal_welch_all.csv")
    clean = table[~table["degenerate_fit"]]
    print(
        f"wrote {OUT}/full_results.tex ({len(table)} rows) and economic_translation.tex\n"
        f"  median ceiling {clean['ir_ceiling'].median():.2f}; "
        f"{int((clean['ir_ceiling'] > CAMPBELL_THOMPSON).sum())}/{len(clean)} above "
        f"{CAMPBELL_THOMPSON}; max e-value {clean['evalue'].max():.2f}"
    )


if __name__ == "__main__":
    main()
