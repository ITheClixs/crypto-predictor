"""What the execution convention and the trading schedule are worth, in Sharpe ratios.

Three specifications on identical forecasts:

``same-close, single phase``
    What earlier versions reported. Enters at the close its own features were computed from,
    which requires knowing the price before it prints, and samples every h-th bar from one
    arbitrary starting offset out of h.
``staggered, same close``
    Removes the phase arbitrariness only: 1/h of capital in each daily vintage, still entered
    at the contemporaneous close.
``staggered, one-bar delay`` (the paper's primary specification)
    Feasible: the weight in force during bar t depends only on signals through t-1.

The gap between the first and the last is the size of the two conventions an earlier version
of this study got wrong, and it is not small at h = 1.

Usage: execution_contrast.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptoforecast.backtest.strategy import backtest_strategy, staggered_strategy
from cryptoforecast.config import DEFAULT_CONFIG
from cryptoforecast.evaluate.stats import sharpe_ratio

COSTS = DEFAULT_CONFIG.costs
MODELS = ("ridge", "elastic_net", "gbm")


def _sharpe(result: object) -> float:
    return float(
        sharpe_ratio(result.net, periods_per_year=result.periods_per_year)  # type: ignore[attr-defined]
    )


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
        for model in MODELS:
            oos = base.assign(y_pred=pivot[("y_pred", model)])
            rows.append(
                {
                    "asset": asset,
                    "h": int(horizon),
                    "model": model,
                    "same_close_single_phase": _sharpe(
                        backtest_strategy(oos, int(horizon), COSTS, kind="sign")
                    ),
                    "staggered_same_close": _sharpe(
                        staggered_strategy(oos, int(horizon), COSTS, entry_delay=0)
                    ),
                    "staggered_delayed": _sharpe(
                        staggered_strategy(oos, int(horizon), COSTS, entry_delay=1)
                    ),
                }
            )

    result = pd.DataFrame(rows)
    result["delay_cost"] = result["staggered_delayed"] - result["staggered_same_close"]
    result["stagger_cost"] = result["staggered_same_close"] - result["same_close_single_phase"]
    result.to_csv("audit/execution_contrast.csv", index=False)
    print(result.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    for horizon in sorted(result["h"].unique()):
        cell = result[result["h"] == horizon]
        print(
            f"\nh = {horizon}: mean Sharpe cost of the one-bar delay "
            f"{cell['delay_cost'].mean():+.3f} "
            f"(worst {cell['delay_cost'].min():+.3f}); "
            f"of staggering {cell['stagger_cost'].mean():+.3f}"
        )
    print(f"\nlargest single change, any setting: {np.abs(result['delay_cost']).max():.3f}")
    print("wrote audit/execution_contrast.csv")


if __name__ == "__main__":
    main()
