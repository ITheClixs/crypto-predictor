"""Net Sharpe as a function of the per-side cost, and the break-even cost of each setting.

The study charges one cost scenario, 17 bp per side, and reports the resulting Sharpe ratios as
"net of realistic costs". One scenario is not a cost model. This sweeps the per-side cost over a
grid and reports, for every setting, the cost at which its net Sharpe crosses zero and the cost
at which it stops beating buy-and-hold on the same schedule.

Reuses the saved forecasts, so no model is refit and nothing here can change a forecast. The
strategy is the paper's primary one -- staggered over all h daily vintages, entered one bar
after the signal -- so the break-even costs describe the specification the paper reports
rather than the same-close single-phase one it retracted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptoforecast.backtest.strategy import staggered_strategy
from cryptoforecast.config import CostModel
from cryptoforecast.evaluate.stats import sharpe_ratio

ML = ("ridge", "elastic_net", "gbm")
GRID_BP = (0, 5, 10, 17, 25, 40, 60, 100)
BASELINE_BP = 17


def _costs(bp: float) -> CostModel:
    """A cost model whose total per-side charge is ``bp`` basis points."""
    return CostModel(fee_bps=bp, slippage_bps=0.0, half_spread_bps=0.0)


def _net_sharpe(oos: pd.DataFrame, horizon: int, bp: float) -> float:
    result = staggered_strategy(oos, horizon, _costs(bp), kind="sign", entry_delay=1)
    return float(sharpe_ratio(result.net, periods_per_year=result.periods_per_year))


def _break_even(oos: pd.DataFrame, horizon: int) -> float:
    """Per-side cost in bp at which net Sharpe first hits zero, on a fine grid."""
    fine = np.linspace(0.0, 200.0, 401)
    if _net_sharpe(oos, horizon, 0.0) <= 0:
        return 0.0
    for bp in fine:
        if _net_sharpe(oos, horizon, float(bp)) <= 0:
            return float(bp)
    return float("inf")


def main() -> None:
    df = pd.read_csv("audit/forecasts.csv", parse_dates=["date"])
    rows = []
    for (asset, horizon), group in df.groupby(["asset", "horizon"]):
        piv = group.pivot_table(index="date", columns="model", values=["y_true", "y_pred", "close"])
        base = pd.DataFrame({"y_true": piv[("y_true", "ridge")], "close": piv[("close", "ridge")]})
        long_only = base.assign(y_pred=1.0)
        for model in ML:
            oos = base.assign(y_pred=piv[("y_pred", model)])
            row = {"asset": asset, "h": int(horizon), "model": model}
            for bp in GRID_BP:
                row[f"sharpe_{bp}bp"] = _net_sharpe(oos, int(horizon), float(bp))
            row["breakeven_bp"] = _break_even(oos, int(horizon))
            row["bh_17bp"] = _net_sharpe(long_only, int(horizon), float(BASELINE_BP))
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
        print(
            f"break-even cost, median {finite.breakeven_bp.median():.0f} bp, "
            f"max {finite.breakeven_bp.max():.0f} bp"
        )
    print(f"settings beating buy-and-hold at 17 bp:       {int(res.beats_bh_at_17bp.sum())}/18")
    res.to_csv("audit/cost_curve.csv", index=False)
    print("wrote audit/cost_curve.csv")


if __name__ == "__main__":
    main()
