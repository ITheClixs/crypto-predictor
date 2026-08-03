"""Power of Clark-West against the recursive mean, and the minimum detectable effect.

The paper says it is "not powered to exclude an economically relevant effect" without saying
what effect. This reads the ``--rho`` cells written by ``mc_null_wcw.py`` -- alternatives in
which the optimal one-step predictor has population ``R^2 = rho^2`` -- and reports the power
curve plus the smallest effect the design detects 80% of the time.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pandas as pd

RHOS = (0.02, 0.04, 0.06, 0.08, 0.10, 0.11, 0.12)
TARGET = 0.80


def _power(rho: float, col: str) -> tuple[float, float, int]:
    tag = f"{rho:g}".replace("0.", "")
    p = pathlib.Path(f"audit/mc_null_wcw_h1_b21_signflip_rho{tag}.csv")
    if not p.exists():
        return float("nan"), float("nan"), 0
    v = pd.read_csv(p)[col].dropna().to_numpy()
    r = float(np.mean(v > 1.645))
    return r, math.sqrt(r * (1 - r) / v.size), v.size


def main() -> None:
    print(f"{'rho':>6}{'pop R^2':>10}{'ridge':>16}{'elastic net':>18}")
    rows = []
    for rho in RHOS:
        pr, sr, n = _power(rho, "ridge_cw_mean")
        pe, se, _ = _power(rho, "elastic_net_cw_mean")
        if not n:
            continue
        rows.append((rho, rho**2, pr, pe))
        print(
            f"{rho:>6.2f}{rho**2:>10.4f}"
            f"{100 * pr:>12.1f}%±{100 * sr:.1f}{100 * pe:>13.1f}%±{100 * se:.1f}"
        )

    # Linear interpolation on the ridge curve for the 80%-power effect size.
    xs = [r[0] for r in rows]
    ys = [r[2] for r in rows]
    mde_rho = float("nan")
    for (x0, y0), (x1, y1) in zip(zip(xs, ys, strict=True), zip(xs[1:], ys[1:], strict=True), strict=False):
        if y0 < TARGET <= y1:
            mde_rho = x0 + (TARGET - y0) * (x1 - x0) / (y1 - y0)
            break
    print()
    if math.isfinite(mde_rho):
        print(f"minimum detectable effect at {TARGET:.0%} power, h=1, CW vs recursive mean:")
        print(f"  rho = {mde_rho:.3f}  =>  population R^2 = {mde_rho**2:.4f} ({100 * mde_rho**2:.2f}%)")
    else:
        print("80% power not bracketed by the grid; extend RHOS.")
    print()
    print("For scale, the study's own realised R^2_OS against the recursive mean spans")
    print("-0.0908 to +0.0031, so an effect this design would reliably detect is larger")
    print("than anything it observed.")


if __name__ == "__main__":
    main()
