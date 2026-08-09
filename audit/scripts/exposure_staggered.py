"""Exposure decomposition under the *primary* economic specification.

``exposure.py`` regressed the single-phase, same-close backtest on buy-and-hold. That
specification is no longer the paper's primary one: it enters at the close its own features are
computed from, and it picks one of ``h`` schedules arbitrarily. This script repeats the
decomposition on the staggered, one-bar-delayed portfolio that replaced it, so the beta and
alpha quoted in the manuscript describe the strategy the manuscript actually reports.

Regression: daily net strategy return on daily net buy-and-hold return, same schedule, same
costs, Newey-West standard error on the intercept.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptoforecast.backtest.strategy import buy_and_hold, staggered_strategy
from cryptoforecast.config import DEFAULT_CONFIG
from cryptoforecast.evaluate.stats import sharpe_ratio

COSTS = DEFAULT_CONFIG.costs
MODELS = ("historical_mean", "ridge", "elastic_net", "gbm")
NW_LAGS = 5


def _nw_intercept_t(x_market: np.ndarray, y_strategy: np.ndarray) -> tuple[float, float, float]:
    """OLS of strategy on market; returns (alpha per period, beta, Newey-West t on alpha)."""
    design = np.column_stack([np.ones_like(x_market), x_market])
    coef, *_ = np.linalg.lstsq(design, y_strategy, rcond=None)
    resid = y_strategy - design @ coef
    xtx_inv = np.linalg.inv(design.T @ design)
    meat = np.zeros((2, 2))
    for lag in range(NW_LAGS + 1):
        weight = 1.0 - lag / (NW_LAGS + 1)
        for t in range(lag, resid.size):
            outer = np.outer(design[t] * resid[t], design[t - lag] * resid[t - lag])
            meat += weight * (outer + outer.T) / 2 if lag else outer
    var = xtx_inv @ meat @ xtx_inv
    return float(coef[0]), float(coef[1]), float(coef[0] / np.sqrt(var[0, 0]))


def main() -> None:
    df = pd.read_csv("audit/forecasts.csv", parse_dates=["date"])
    print(
        f"{'asset':5}{'h':>3} {'model':16}{'Sharpe':>8}{'B&H':>7}{'%long':>7}"
        f"{'beta':>7}{'alpha_ann':>11}{'t(alpha)':>10}"
    )
    rows = []
    for (asset, horizon), group in df.groupby(["asset", "horizon"]):
        piv = group.pivot_table(index="date", columns="model", values=["y_true", "y_pred", "close"])
        base = pd.DataFrame(
            {
                "y_true": piv[("y_true", "ridge")],
                "close": piv[("close", "ridge")],
            }
        )
        ref = buy_and_hold(base.assign(y_pred=0.0), horizon, COSTS)
        for model in MODELS:
            oos = base.assign(y_pred=piv[("y_pred", model)])
            strat = staggered_strategy(oos, horizon, COSTS)
            joined = pd.concat([strat.net.rename("s"), ref.net.rename("m")], axis=1).dropna()
            alpha, beta, t_alpha = _nw_intercept_t(joined["m"].to_numpy(), joined["s"].to_numpy())
            pct_long = 100.0 * float((strat.positions > 0).mean())
            row = {
                "asset": asset,
                "h": horizon,
                "model": model,
                "sharpe": sharpe_ratio(strat.net, strat.periods_per_year),
                "bh_sharpe": sharpe_ratio(ref.net, ref.periods_per_year),
                "pct_long": pct_long,
                "beta": beta,
                "alpha_ann": alpha * 365.0,
                "t_alpha": t_alpha,
            }
            rows.append(row)
            print(
                f"{asset:5}{horizon:>3} {model:16}{row['sharpe']:>8.2f}{row['bh_sharpe']:>7.2f}"
                f"{pct_long:>7.0f}{beta:>7.2f}{row['alpha_ann']:>11.3f}{t_alpha:>10.2f}"
            )
    pd.DataFrame(rows).to_csv("audit/exposure_staggered.csv", index=False)
    print("\nwrote audit/exposure_staggered.csv")


if __name__ == "__main__":
    main()
