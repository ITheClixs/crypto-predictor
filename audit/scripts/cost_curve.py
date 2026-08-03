"""Net Sharpe as a function of the per-side cost, and the break-even cost of each setting.

The study charges one cost scenario, 17 bp per side, and reports the resulting Sharpe ratios as
"net of realistic costs". One scenario is not a cost model. This sweeps the per-side cost over a
grid and reports, for every setting, the cost at which its net Sharpe crosses zero and the cost
at which it stops beating buy-and-hold on the same schedule.

Reuses the saved forecasts, so no model is refit and nothing here can change a forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptoforecast.backtest.costs import turnover
from cryptoforecast.evaluate.stats import sharpe_ratio

ML = ("ridge", "elastic_net", "gbm")
GRID_BP = (0, 5, 10, 17, 25, 40, 60, 100)
BASELINE_BP = 17


def _net_sharpe(y_true: np.ndarray, y_pred: np.ndarray, horizon: int, cost: float) -> float:
    pos = np.sign(y_pred)
    net = pos * np.expm1(y_true) - turnover(pd.Series(pos)).to_numpy() * cost
    return float(sharpe_ratio(net, 365.0 / horizon))


def _break_even(y_true: np.ndarray, y_pred: np.ndarray, horizon: int) -> float:
    """Per-side cost in bp at which net Sharpe first hits zero, by bisection on a fine grid."""
    fine = np.linspace(0.0, 0.02, 401)  # 0 to 200 bp
    sharpes = [_net_sharpe(y_true, y_pred, horizon, c) for c in fine]
    if sharpes[0] <= 0:
        return 0.0
    for c, s in zip(fine, sharpes, strict=True):
        if s <= 0:
            return float(c * 1e4)
    return float("inf")


def main() -> None:
    df = pd.read_csv("audit/forecasts.csv", parse_dates=["date"])
    rows = []
    for (asset, horizon), group in df.groupby(["asset", "horizon"]):
        piv = group.pivot_table(index="date", columns="model", values=["y_true", "y_pred"])
        y = piv[("y_true", "ridge")].to_numpy()
        step = slice(None, None, horizon)
        y_s = y[step]
        long_only = np.ones_like(y_s)
        for model in ML:
            p_s = piv[("y_pred", model)].to_numpy()[step]
            row = {"asset": asset, "h": horizon, "model": model}
            for bp in GRID_BP:
                row[f"sharpe_{bp}bp"] = _net_sharpe(y_s, p_s, horizon, bp / 1e4)
            row["breakeven_bp"] = _break_even(y_s, p_s, horizon)
            row["bh_17bp"] = _net_sharpe(y_s, long_only, horizon, BASELINE_BP / 1e4)
            row["beats_bh_at_17bp"] = row["sharpe_17bp"] > row["bh_17bp"]
            rows.append(row)

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 220, "display.max_columns", 40)
    cols = ["asset", "h", "model"] + [f"sharpe_{bp}bp" for bp in GRID_BP] + ["breakeven_bp"]
    print("=" * 130)
    print("Net Sharpe vs per-side cost (bp). The study quotes the 17 bp column only.")
    print("=" * 130)
    print(res[cols].round(3).to_string(index=False))
    print()
    finite = res[np.isfinite(res.breakeven_bp)]
    print(f"settings with a positive net Sharpe at 0 bp: {int((res.sharpe_0bp > 0).sum())}/18")
    print(f"settings still positive at 17 bp:            {int((res.sharpe_17bp > 0).sum())}/18")
    print(f"settings still positive at 40 bp:            {int((res.sharpe_40bp > 0).sum())}/18")
    if len(finite):
        print(f"break-even cost, median {finite.breakeven_bp.median():.0f} bp, "
              f"max {finite.breakeven_bp.max():.0f} bp")
    print(f"settings beating buy-and-hold at 17 bp:       {int(res.beats_bh_at_17bp.sum())}/18")
    res.to_csv("audit/cost_curve.csv", index=False)
    print("wrote audit/cost_curve.csv")


if __name__ == "__main__":
    main()
