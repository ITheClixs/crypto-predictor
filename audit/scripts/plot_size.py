"""Figure 9: measured size of Clark-West under three nulls, by benchmark and horizon.

The calibration is the paper's main contribution and had no figure. This draws it from the
Monte Carlo cells written by ``mc_null_wcw.py`` -- so, like every other figure in the study,
it cannot drift away from the table it illustrates.

Bar height is the mean rejection rate across the estimators measured in that cell; the whisker
spans them. The dashed line is the nominal 5%.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
import matplotlib.pyplot as plt

from cryptoforecast.plots.style import PALETTE, REFERENCE_BLACK, finish

NULLS = (
    ("indep.", "audit/mc_null_wcw_h{h}_b1.csv"),
    ("sign-flip", "audit/mc_null_wcw_h{h}_b21_signflip.csv"),
    ("block", "audit/mc_null_wcw_h{h}_b21.csv"),
)
MODELS = ("ridge", "elastic_net", "gbm")
BENCHMARKS = (("zero forecast", "zero"), ("recursive mean", "mean"))


def _rates(path: str, bench: str) -> list[float]:
    p = pathlib.Path(path)
    if not p.exists():
        return []
    d = pd.read_csv(p)
    out = []
    for m in MODELS:
        col = f"{m}_cw_{bench}"
        if col in d:
            v = d[col].dropna().to_numpy()
            if v.size:
                out.append(100.0 * float(np.mean(v > 1.645)))
    return out


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), sharey=True)
    width = 0.36
    x = np.arange(len(NULLS))

    for ax, horizon in zip(axes, (1, 7), strict=True):
        for j, (label, bench) in enumerate(BENCHMARKS):
            heights, los, his = [], [], []
            for _, template in NULLS:
                r = _rates(template.format(h=horizon), bench)
                heights.append(np.mean(r) if r else np.nan)
                los.append(np.mean(r) - min(r) if r else 0.0)
                his.append(max(r) - np.mean(r) if r else 0.0)
            ax.bar(
                x + (j - 0.5) * width,
                heights,
                width,
                yerr=[los, his],
                capsize=3,
                color=PALETTE[j],
                label=f"CW vs. {label}",
                edgecolor="white",
                linewidth=0.6,
            )
        ax.axhline(5.0, color=REFERENCE_BLACK, linestyle="--", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([n for n, _ in NULLS])
        ax.set_title(f"$h = {horizon}$", fontsize=10)
        ax.set_xlabel("resampling null")

    axes[0].set_ylabel("rejection rate at a nominal 5%")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(
        "Measured size of Clark--West. Only the sign-flip null has both realistic "
        "dependence\nand a true no-predictability hypothesis; the block null is an upper bound.",
        fontsize=8.5,
        y=1.06,
    )
    finish(fig, pathlib.Path("reports/figures/fig9_size.png"))
    print("wrote reports/figures/fig9_size.{pdf,png}")


if __name__ == "__main__":
    main()
