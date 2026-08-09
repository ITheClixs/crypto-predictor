"""Pilot: the certificate on the canonical equity-premium predictors.

The crypto study's binding constraint is calendar span, not method: six years of daily data
cannot certify an annualised information ratio below about 1.6, and nothing in that literature
claims one. The Goyal-Welch dataset is the same question with a century of data and a hundred
years of published candidate predictors, so it is where the instrument can actually
discriminate. This script is the go/no-go pilot for that project.

Design, following Goyal and Welch (2008) exactly so the reproduction check is meaningful:

* Outcome: the monthly log equity premium, ``log(1 + ret) - log(1 + Rfree)``.
* Predictor: lagged one month, so nothing contemporaneous enters the forecast.
* Forecast: expanding-window OLS with an intercept, re-estimated every month, first forecast
  after a 240-month burn-in. The benchmark is the same window's historical mean, which is the
  intercept-only restriction of that regression -- the alignment the manuscript argues for.
* Signal for the certificate: the regression forecast minus the historical-mean forecast.

The reproduction check matters more than anything else here. Goyal and Welch's headline is that
essentially nothing beats the historical mean out of sample, so out-of-sample R-squared should
be negative for almost every predictor. If this script reports otherwise, it is wrong, and no
conclusion drawn from it is worth anything.

Usage: goyal_welch_pilot.py [--start YYYYMM] [--burn 240]
"""

from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd

from alphacert import certifiable_ratio, certify, growth_to_ratio, merge_average, value_ceiling
from cryptoforecast.evaluate.stats import newey_west_lrv

WORKBOOK = "data/goyal_welch/gw_17mw_IpaiLFDrGnrPRQ2o1ugV5nJsZuD1.xlsx"
BURN = int(sys.argv[sys.argv.index("--burn") + 1]) if "--burn" in sys.argv else 240
START = int(sys.argv[sys.argv.index("--start") + 1]) if "--start" in sys.argv else 192601
PERIODS_PER_YEAR = 12.0
#: A monthly log equity premium outside +/-60% has never happened; the worst month in the
#: sample is around -35%. Fixed in advance, and never approached.
ENVELOPE = 0.6

#: The classic Goyal-Welch predictors, restricted to those with a long monthly history.
PREDICTORS = (
    "d/p",
    "d/y",
    "e/p",
    "d/e",
    "b/m",
    "ntis",
    "tbl",
    "lty",
    "ltr",
    "tms",
    "dfy",
    "dfr",
    "infl",
    "svar",
)


def load() -> pd.DataFrame:
    frame = pd.read_excel(WORKBOOK, sheet_name="Monthly")
    frame = frame[frame["yyyymm"] >= START].copy()
    for column in ("ret", "Rfree", *PREDICTORS):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    # Goyal-Welch's log equity premium.
    frame["premium"] = np.log1p(frame["ret"]) - np.log1p(frame["Rfree"])
    return frame.reset_index(drop=True)


def walk_forward(
    x: np.ndarray, y: np.ndarray, burn: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expanding-window OLS with an intercept, and the intercept-only benchmark beside it.

    Returns (outcome, model forecast, benchmark forecast) over the evaluation window. Every
    forecast at ``t`` uses observations strictly before ``t``, and the predictor is already
    lagged, so nothing contemporaneous is available to either forecaster.
    """
    n = y.size
    model = np.full(n, np.nan)
    bench = np.full(n, np.nan)
    for t in range(burn, n):
        xt, yt = x[:t], y[:t]
        keep = np.isfinite(xt) & np.isfinite(yt)
        if keep.sum() < 60 or not np.isfinite(x[t]):
            continue
        design = np.column_stack([np.ones(keep.sum()), xt[keep]])
        coef, *_ = np.linalg.lstsq(design, yt[keep], rcond=None)
        model[t] = coef[0] + coef[1] * x[t]
        bench[t] = yt[keep].mean()
    ok = np.isfinite(model) & np.isfinite(bench) & np.isfinite(y)
    return y[ok], model[ok], bench[ok]


def clark_west(outcome: np.ndarray, model: np.ndarray, bench: np.ndarray) -> float:
    adjusted = 2.0 * (outcome - bench) * (model - bench)
    lrv = newey_west_lrv(adjusted, 0)
    return float(adjusted.mean() / math.sqrt(lrv / adjusted.size)) if lrv > 0 else float("nan")


def main() -> None:
    frame = load()
    span_years = len(frame) / PERIODS_PER_YEAR
    print(
        f"Goyal-Welch monthly, {frame['yyyymm'].iloc[0]}-{frame['yyyymm'].iloc[-1]}: "
        f"{len(frame)} months = {span_years:.1f} years\n"
        f"burn-in {BURN} months; certifiable IR floor for this span: "
        f"{certifiable_ratio(span_years, periods_per_year=PERIODS_PER_YEAR):.2f} "
        f"({certifiable_ratio(span_years, periods_per_year=PERIODS_PER_YEAR, kelly_known=True):.2f} "
        f"pre-committed)\n"
    )

    premium = frame["premium"].to_numpy()
    rows = []
    for name in PREDICTORS:
        lagged = frame[name].shift(1).to_numpy()
        outcome, model, bench = walk_forward(lagged, premium, BURN)
        if outcome.size < 240:
            continue
        signal = model - bench
        r2 = 1.0 - np.sum((outcome - model) ** 2) / np.sum((outcome - bench) ** 2)
        cert = certify(signal, outcome, return_bound=ENVELOPE)
        ceiling = value_ceiling(signal, outcome, return_bound=ENVELOPE)
        rows.append(
            {
                "predictor": name,
                "n": int(outcome.size),
                "r2_oos_pct": 100 * r2,
                "clark_west": clark_west(outcome, model, bench),
                "evalue": cert.evalue,
                "certified": cert.rejects(0.05),
                "implied_ir": growth_to_ratio(max(cert.growth_rate(), 0.0), PERIODS_PER_YEAR),
                "ir_ceiling": ceiling.ratio_ceiling(float(outcome.std()), PERIODS_PER_YEAR),
            }
        )

    result = pd.DataFrame(rows).sort_values("evalue", ascending=False)
    result.to_csv("audit/goyal_welch_pilot.csv", index=False)
    print(result.to_string(index=False, float_format=lambda v: f"{v:9.3f}"))

    negative = int((result["r2_oos_pct"] < 0).sum())
    print(
        f"\nREPRODUCTION CHECK: out-of-sample R^2 is negative for {negative} of "
        f"{len(result)} predictors. Goyal and Welch (2008) report that essentially nothing "
        f"beats the historical mean, so a high count here is the pipeline behaving."
    )
    print(
        f"\nGrid-level e-value (average, valid under arbitrary dependence): "
        f"{merge_average(result['evalue'].to_numpy()):.3f}   (20 required)"
    )
    print(f"Certified at 5%: {int(result['certified'].sum())} of {len(result)}")
    print(
        f"IR ceilings span {result['ir_ceiling'].min():.2f} to "
        f"{result['ir_ceiling'].max():.2f}; e-values span "
        f"{result['evalue'].min():.2f} to {result['evalue'].max():.2f}"
    )
    print("\nwrote audit/goyal_welch_pilot.csv")


if __name__ == "__main__":
    main()
