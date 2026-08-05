"""How much of a reported Sharpe interval is Monte Carlo noise in the bootstrap itself.

An interval whose lower bound sits near zero is a decision, so the number of resamples behind
it is not a detail. This re-computes the primary specification's intervals at several
resample counts and across seeds, and reports how far the endpoints move.

Usage: bootstrap_stability.py
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd

from cryptoforecast.backtest.strategy import staggered_strategy
from cryptoforecast.config import DEFAULT_CONFIG
from cryptoforecast.evaluate.stats import block_bootstrap_ci, sharpe_ratio

COSTS = DEFAULT_CONFIG.costs
COUNTS = (500, 2_000, 10_000)
SEEDS = (7, 11, 23, 41, 97)


def main() -> None:
    frame = pd.read_csv("audit/forecasts.csv", parse_dates=["date"])
    rows = []
    for (asset, horizon), group in frame.groupby(["asset", "horizon"]):
        pivot = group.pivot_table(
            index="date", columns="model", values=["y_true", "y_pred", "close"]
        )
        base = pd.DataFrame(
            {"y_true": pivot[("y_true", "ridge")], "close": pivot[("close", "ridge")]}
        )
        for model in ("ridge", "elastic_net", "gbm"):
            oos = base.assign(y_pred=pivot[("y_pred", model)])
            result = staggered_strategy(oos, int(horizon), COSTS, entry_delay=1)
            statistic = partial(sharpe_ratio, periods_per_year=result.periods_per_year)
            point = float(statistic(result.net.to_numpy()))
            for count in COUNTS:
                los, his = [], []
                for seed in SEEDS:
                    lo, hi = block_bootstrap_ci(result.net, statistic, n_boot=count, seed=seed)
                    los.append(lo)
                    his.append(hi)
                rows.append(
                    {
                        "asset": asset,
                        "h": int(horizon),
                        "model": model,
                        "sharpe": point,
                        "n_boot": count,
                        "lo_mean": float(np.mean(los)),
                        "lo_range": float(np.ptp(los)),
                        "hi_range": float(np.ptp(his)),
                        "sign_flips": int(len({lo > 0 for lo in los}) > 1),
                    }
                )

    result = pd.DataFrame(rows)
    result.to_csv("audit/bootstrap_stability.csv", index=False)
    summary = result.groupby("n_boot").agg(
        max_lower_bound_range=("lo_range", "max"),
        median_lower_bound_range=("lo_range", "median"),
        settings_whose_verdict_flips=("sign_flips", "sum"),
    )
    print("Movement of the 95% Sharpe interval across five bootstrap seeds:\n")
    print(summary.to_string(float_format=lambda v: f"{v:8.4f}"))
    print(
        "\n'verdict flips' counts settings where the lower bound crosses zero between seeds, "
        "i.e. where the reject-or-not reading is a property of the random number generator."
    )
    print("wrote audit/bootstrap_stability.csv")


if __name__ == "__main__":
    main()
