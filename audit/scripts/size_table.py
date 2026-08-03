"""Assemble the size table from the Monte Carlo cells written by ``mc_null_wcw.py``.

Prints the LaTeX body of Table 2 in the paper plus Monte Carlo standard errors, so the
manuscript never carries a hand-transcribed rejection rate.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pandas as pd

CELLS = {
    "h1_iid": "audit/mc_null_wcw_h1_b1.csv",
    "h1_block": "audit/mc_null_wcw_h1_b21.csv",
    "h1_signflip": "audit/mc_null_wcw_h1_b21_signflip.csv",
    "h7_iid": "audit/mc_null_wcw_h7_b1.csv",
    "h7_block": "audit/mc_null_wcw_h7_b21.csv",
    "h7_signflip": "audit/mc_null_wcw_h7_b21_signflip.csv",
}
GBM_CELL = "audit/mc_null_wcw_h1_b1_gbm.csv"
ORDER = ["h1_iid", "h1_signflip", "h1_block", "h7_iid", "h7_signflip", "h7_block"]


def rate(series: pd.Series, two_sided: bool = False) -> tuple[float, float, int]:
    v = series.dropna().to_numpy()
    r = float(np.mean(np.abs(v) > 1.96) if two_sided else np.mean(v > 1.645))
    se = math.sqrt(r * (1 - r) / max(v.size, 1))
    return 100 * r, 100 * se, v.size


def main() -> None:
    frames = {k: pd.read_csv(v) for k, v in CELLS.items() if pathlib.Path(v).exists()}
    gbm = pd.read_csv(GBM_CELL) if pathlib.Path(GBM_CELL).exists() else None

    rows = [
        ("CW vs.\\ zero forecast", "ridge", "ridge_cw_zero", False),
        ("CW vs.\\ zero forecast", "elastic net", "elastic_net_cw_zero", False),
        ("WCW vs.\\ zero forecast", "ridge", "ridge_wcw_zero", False),
        ("WCW vs.\\ zero forecast", "elastic net", "elastic_net_wcw_zero", False),
        ("CW vs.\\ recursive mean", "ridge", "ridge_cw_mean", False),
        ("CW vs.\\ recursive mean", "elastic net", "elastic_net_cw_mean", False),
        ("WCW vs.\\ recursive mean", "ridge", "ridge_wcw_mean", False),
        ("WCW vs.\\ recursive mean", "elastic net", "elastic_net_wcw_mean", False),
        ("CW vs.\\ zero, drift-only model", "---", "mean_cw_zero", False),
        ("DM vs.\\ zero, two-sided", "ridge", "ridge_dm_zero", True),
    ]

    print(f"{'statistic':32}{'model':13}" + "".join(f"{c:>16}" for c in ORDER))
    for label, model, col, two in rows:
        cells = []
        for c in ORDER:
            if c not in frames or col not in frames[c]:
                cells.append("       ---      ")
                continue
            r, se, n = rate(frames[c][col], two)
            cells.append(f"{r:>9.1f}%±{se:.1f} ")
        print(f"{label:32}{model:13}" + "".join(cells))

    if gbm is not None:
        print()
        print(f"GBM cell (h=1 iid only), n={len(gbm)}")
        for col in ("gbm_cw_zero", "gbm_wcw_zero", "gbm_cw_mean", "gbm_wcw_mean"):
            r, se, n = rate(gbm[col])
            print(f"  {col:20}{r:>6.1f}% ± {se:.1f}pp  (n={n})")

    print()
    print("max |CW - WCW| per cell:")
    for c in ORDER:
        if c not in frames:
            continue
        d = frames[c]
        worst = 0.0
        for nm in ("ridge", "elastic_net", "gbm"):
            for bench in ("zero", "mean"):
                a, b = f"{nm}_cw_{bench}", f"{nm}_wcw_{bench}"
                if a in d and b in d:
                    ok = d[a].notna() & d[b].notna()
                    worst = max(worst, float((d.loc[ok, a] - d.loc[ok, b]).abs().max()))
        print(f"  {c:10}{worst:.4f}")


if __name__ == "__main__":
    main()
