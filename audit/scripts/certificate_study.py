"""Run the drift-robust certificate on every setting in the study.

The signal is the model's forecast minus the intercept-only forecast produced by the same
estimation window and refitting schedule, so the null the certificate tests is exactly the
one the manuscript argues for: *the features add nothing beyond the intercept*. Nothing is
refit -- the certificate consumes the walk-forward forecasts the pipeline already wrote.

At h = 7 the outcomes overlap, so the per-day payoffs are not a martingale difference
sequence. The sample is split into the seven non-overlapping phases, one certificate is
built on each, and the seven e-values are averaged. That average is valid however the
phases depend on one another, and read as a trading rule it is the equal-weighted
portfolio over all seven daily vintages -- the same object the economic section needs.

Usage: certificate_study.py [--payoff tanh|sign|identity]
"""

from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd

from alphacert import (
    certifiable_ratio,
    certify,
    certify_overlapping,
    e_bh,
    growth_to_ratio,
    merge_average,
    value_ceiling,
)
from cryptoforecast.evaluate.stats import newey_west_lrv

PAYOFF = sys.argv[sys.argv.index("--payoff") + 1] if "--payoff" in sys.argv else "identity"
BENCHMARK = "historical_mean"
MODELS = ("ridge", "elastic_net", "gbm")
PERIODS_PER_YEAR = 365.0

#: Smallest annualised information ratio the pre-committed variant is designed to catch.
#: Declared here, before the data are touched, which is the entire point of declaring it.
DESIGN_RATIO = 1.0


def _clark_west(outcome: np.ndarray, model: np.ndarray, bench: np.ndarray, lags: int) -> float:
    adjusted = 2.0 * (outcome - bench) * (model - bench)
    lrv = newey_west_lrv(adjusted, lags)
    if lrv <= 0:
        return float("nan")
    return float(adjusted.mean() / math.sqrt(lrv / adjusted.size))


def main() -> None:
    frame = pd.read_csv("audit/forecasts.csv", parse_dates=["date"])
    rows: list[dict[str, object]] = []
    for asset in sorted(frame["asset"].unique()):
        for horizon in sorted(frame["horizon"].unique()):
            cell = frame[(frame["asset"] == asset) & (frame["horizon"] == horizon)]
            bench = cell[cell["model"] == BENCHMARK].sort_values("date")
            outcome = bench["y_true"].to_numpy()
            baseline = bench["y_pred"].to_numpy()
            for model in MODELS:
                fit = cell[cell["model"] == model].sort_values("date")
                if fit.empty:
                    continue
                signal = fit["y_pred"].to_numpy() - baseline
                # An h-period log return needs an envelope that grows with h.
                # One per sqrt-period is about 30 daily standard deviations:
                # generous, fixed in advance, never approached by the sample.
                envelope = float(np.sqrt(horizon))
                if horizon == 1:
                    cert = certify(signal, outcome, payoff=PAYOFF, return_bound=envelope)
                    designed = certify(
                        signal,
                        outcome,
                        payoff=PAYOFF,
                        return_bound=envelope,
                        design_ratio=DESIGN_RATIO,
                    )
                    evalue, peak = cert.evalue, float(cert.wealth.max())
                    growth = cert.growth_rate()
                    designed_e = designed.evalue
                    phases = 1
                else:
                    evalue, certs = certify_overlapping(
                        signal,
                        outcome,
                        horizon=int(horizon),
                        payoff=PAYOFF,
                        return_bound=envelope,
                    )
                    designed_e, _ = certify_overlapping(
                        signal,
                        outcome,
                        horizon=int(horizon),
                        payoff=PAYOFF,
                        return_bound=envelope,
                        design_ratio=DESIGN_RATIO,
                    )
                    peak = merge_average([float(c.wealth.max()) for c in certs])
                    growth = float(np.mean([c.growth_rate() for c in certs])) / horizon
                    phases = len(certs)
                ceiling = value_ceiling(signal, outcome, alpha=0.05, return_bound=envelope)
                rows.append(
                    {
                        "asset": asset,
                        "h": int(horizon),
                        "model": model,
                        "n": int(outcome.size),
                        "phases": phases,
                        "evalue": evalue,
                        "evalue_designed": designed_e,
                        "p_anytime": min(1.0, 1.0 / peak) if peak > 0 else 1.0,
                        "growth_nats_per_period": growth,
                        "implied_ratio": growth_to_ratio(max(growth, 0.0), PERIODS_PER_YEAR),
                        "cw_vs_mean": _clark_west(
                            outcome, fit["y_pred"].to_numpy(), baseline, max(0, int(horizon) - 1)
                        ),
                        "cw_vs_zero": _clark_west(
                            outcome,
                            fit["y_pred"].to_numpy(),
                            np.zeros_like(outcome),
                            max(0, int(horizon) - 1),
                        ),
                        "value_lower": ceiling.lower,
                        "value_upper": ceiling.upper,
                        "ratio_ceiling": ceiling.ratio_ceiling(
                            float(outcome.std()), PERIODS_PER_YEAR
                        ),
                    }
                )

    result = pd.DataFrame(rows)
    result["e_bh_reject"] = e_bh(result["evalue"].to_numpy(), alpha=0.05)
    out = f"audit/certificate_study_{PAYOFF}.csv"
    result.to_csv(out, index=False)

    grid_evalue = merge_average(result["evalue"].to_numpy())
    years = float(result["n"].max()) / PERIODS_PER_YEAR
    print(f"payoff = {PAYOFF}, {len(result)} settings, {years:.1f} years of daily data\n")
    print(
        result[
            [
                "asset",
                "h",
                "model",
                "cw_vs_zero",
                "cw_vs_mean",
                "evalue",
                "evalue_designed",
                "p_anytime",
                "implied_ratio",
                "ratio_ceiling",
            ]
        ].to_string(index=False, float_format=lambda v: f"{v:8.3f}")
    )
    print(f"\nglobal-null e-value (average over the grid) = {grid_evalue:.3f}")
    print(f"  rejects the global null at 5%: {grid_evalue >= 20.0}")
    print(f"  settings surviving e-BH at 5%: {int(result['e_bh_reject'].sum())}")
    print(
        f"\nsmallest annualised information ratio this sample could certify: "
        f"{certifiable_ratio(years, kelly_known=False):.2f} "
        f"({certifiable_ratio(years, kelly_known=True):.2f} with a pre-committed stake)"
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
