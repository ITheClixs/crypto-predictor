"""Figure 9: what the drift does to each instrument, and what the sample can settle.

Three panels, because the paper's argument has three steps and each one has a picture.

*Left.* Rejection rate on data containing no predictability, as the asset's drift grows.
Clark-West against a zero benchmark climbs from nominal to almost certain; against the
recursive mean it is correctly sized in this design; the certificate is flat and below
nominal, uniformly in the drift. The closed form of Proposition 1 is overlaid on the
zero-benchmark curve, converted to a rejection probability.

*Middle.* Every certificate the study produces, plotted against the threshold it would have
to cross. The vertical distance is the whole result: the largest wealth in the grid is 1.8
where 20 is required, so the picture is not close.

*Right.* The detection-horizon law. Years of daily data needed to certify a signal of a given
annualised information ratio, with the study's six years marked. Everything below the
intersection is out of reach at this sample size, whatever the test.

Usage: plot_certificate.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from alphacert import certify, certify_overlapping, detection_horizon, fixed_sample_horizon
from cryptoforecast.plots.style import NEGATIVE, PALETTE, POSITIVE, REFERENCE_BLACK, finish

OUT = Path("reports/figures/fig10_certificate.png")
ALPHA = 0.05
THRESHOLD = 1.0 / ALPHA
PERIODS = 365.0
N_TEST = 2186
WINDOW = 504
N_FEATURES = 12


def _predicted_rejection(annual_sharpe: float) -> float:
    """Proposition 1, converted from an expected statistic to a rejection probability."""
    s = annual_sharpe / math.sqrt(PERIODS)
    centre = math.sqrt(N_TEST) * s**2 / math.sqrt(s**2 + (1.0 + N_FEATURES) / WINDOW)
    return float(1.0 - stats.norm.cdf(stats.norm.ppf(0.95) - centre))


def _wealth_paths() -> list[tuple[str, np.ndarray, np.ndarray]]:
    frame = pd.read_csv("audit/forecasts.csv", parse_dates=["date"])
    paths: list[tuple[str, np.ndarray, np.ndarray]] = []
    for asset in sorted(frame["asset"].unique()):
        for horizon in sorted(frame["horizon"].unique()):
            cell = frame[(frame["asset"] == asset) & (frame["horizon"] == horizon)]
            bench = cell[cell["model"] == "historical_mean"].sort_values("date")
            outcome = bench["y_true"].to_numpy()
            baseline = bench["y_pred"].to_numpy()
            for model in ("ridge", "elastic_net", "gbm"):
                fit = cell[cell["model"] == model].sort_values("date")
                if fit.empty:
                    continue
                signal = fit["y_pred"].to_numpy() - baseline
                envelope = float(np.sqrt(horizon))
                if horizon == 1:
                    wealth = certify(signal, outcome, return_bound=envelope).wealth
                else:
                    # Phase certificates share capital 1/h, so the portfolio's wealth is
                    # their mean -- the same object the merged e-value reports.
                    _, certs = certify_overlapping(
                        signal, outcome, horizon=int(horizon), return_bound=envelope
                    )
                    width = min(c.wealth.size for c in certs)
                    wealth = np.mean([c.wealth[:width] for c in certs], axis=0)
                # A phase advances h calendar bars per step, so the x-axis is calendar
                # years for every horizon rather than steps for some and years for others.
                years = np.arange(1, wealth.size + 1) * horizon / PERIODS
                paths.append((f"{asset} {model} h={horizon}", years, wealth))
    return paths


def main() -> None:
    import matplotlib.pyplot as plt

    sweep = pd.read_csv("audit/certificate_drift_sweep.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.5))

    left = axes[0]
    grid = np.linspace(0.0, 2.0, 60)
    left.plot(
        grid,
        [100 * _predicted_rejection(s) for s in grid],
        color=REFERENCE_BLACK,
        lw=1.0,
        ls=":",
        label="Proposition 1 (predicted)",
    )
    left.plot(
        sweep["annual_sharpe"],
        100 * sweep["reject_cw_zero"],
        "o-",
        color=NEGATIVE,
        lw=1.6,
        ms=4,
        label="Clark--West vs zero",
    )
    left.plot(
        sweep["annual_sharpe"],
        100 * sweep["reject_cw_mean"],
        "s-",
        color=PALETTE[0],
        lw=1.6,
        ms=4,
        label="Clark--West vs recursive mean",
    )
    left.plot(
        sweep["annual_sharpe"],
        100 * sweep["reject_certificate"],
        "^-",
        color=POSITIVE,
        lw=1.6,
        ms=4,
        label="Certificate",
    )
    left.axhline(5.0, color=REFERENCE_BLACK, lw=0.8, alpha=0.5)
    left.annotate("nominal 5%", (2.0, 6.5), ha="right", fontsize=7.5, color="#555555")
    left.set_xlabel("Annualised Sharpe ratio of the asset")
    left.set_ylabel("Rejection rate (%)")
    left.set_title("No predictability in the data")
    left.set_ylim(-3, 100)
    left.legend(loc="upper left")

    middle = axes[1]
    paths = _wealth_paths()
    best = max(paths, key=lambda item: item[2].max())
    for name, years, wealth in paths:
        is_best = name == best[0]
        middle.plot(
            years,
            wealth,
            color=NEGATIVE if is_best else PALETTE[0],
            lw=1.4 if is_best else 0.7,
            alpha=1.0 if is_best else 0.4,
            zorder=3 if is_best else 1,
        )
    middle.annotate(
        f"largest: {best[0]}, {best[2].max():.2f}",
        (best[1][-1], best[2].max()),
        xytext=(-4, 6),
        textcoords="offset points",
        ha="right",
        fontsize=7.5,
        color=NEGATIVE,
    )
    middle.axhline(
        THRESHOLD,
        color=REFERENCE_BLACK,
        lw=1.4,
        ls="--",
        label=f"reject at 5%: $\\mathcal{{E}} \\geq {THRESHOLD:.0f}$",
    )
    middle.axhline(1.0, color=REFERENCE_BLACK, lw=0.8, alpha=0.5)
    middle.set_yscale("log")
    middle.set_ylim(0.3, 40)
    middle.set_xlabel("Years of out-of-sample data")
    middle.set_ylabel("Certificate $\\mathcal{E}_t$ (log scale)")
    middle.set_title("All 18 settings, real data")
    middle.legend(loc="upper left")

    right = axes[2]
    ratios = np.linspace(0.4, 3.0, 80)
    right.plot(
        ratios,
        [detection_horizon(r, kelly_known=True) for r in ratios],
        color=POSITIVE,
        lw=1.6,
        label="certificate, stake pre-committed",
    )
    right.plot(
        ratios,
        [detection_horizon(r) for r in ratios],
        color=PALETTE[3],
        lw=1.6,
        ls="-.",
        label="certificate, stake learned online",
    )
    right.plot(
        ratios,
        [fixed_sample_horizon(r) for r in ratios],
        color=REFERENCE_BLACK,
        lw=1.0,
        ls=":",
        label="fixed sample, 80% power",
    )
    span = N_TEST / PERIODS
    right.axhline(span, color=NEGATIVE, lw=1.2, ls="--")
    right.annotate(
        f"this study: {span:.1f} years",
        (3.0, span * 1.25),
        ha="right",
        fontsize=7.5,
        color=NEGATIVE,
    )
    right.set_yscale("log")
    right.set_xlabel("Annualised information ratio")
    right.set_ylabel("Years of daily data needed")
    right.set_title("What a sample of this length can settle")
    right.legend(loc="upper right")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    finish(fig, OUT)
    print(f"wrote {OUT} and {OUT.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
