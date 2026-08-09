"""Does the null survive the specification choices, or is it one setting deep?

A negative result that holds only at the authors' preferred window is not a result. Three
choices are swept, one at a time, on the monthly set:

*Burn-in.* Goyal, Welch and Zafirov begin the out-of-sample period twenty years after the
in-sample period starts. Ten years leaves a noisier benchmark and more evaluation points;
forty leaves a better-estimated benchmark and fewer. The trade is not obviously one-directional
for the certificate, whose evidence accumulates with evaluation points but whose signal is
cleaner when the benchmark is well estimated.

*Window.* An expanding window uses everything, which is efficient if the relationship is
stable and wrong if it is not. A rolling window forgets, which is the point. Goyal and Welch's
whole complaint is instability, so if any specification is going to find predictability it is
the one that only ever looks at recent history.

*Payoff.* The identity payoff assumes only a martingale difference; tanh assumes conditional
symmetry and is not throttled by the return envelope. The equity premium is left-skewed, so
tanh is the assumption-heavier of the two here and is reported as a sensitivity, not a
headline.

Usage: goyal_welch_sensitivity.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphacert import certify, merge_average, value_ceiling
from goyal_welch_pilot import ENVELOPE, PERIODS_PER_YEAR, PREDICTORS, load

BASELINE_BURN = 240


def rolling_forward(
    x: np.ndarray, y: np.ndarray, burn: int, window: int | None, min_train: int = 60
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Walk forward with an expanding window (``window is None``) or a rolling one."""
    n = y.size
    model = np.full(n, np.nan)
    bench = np.full(n, np.nan)
    for t in range(burn, n):
        start = 0 if window is None else max(0, t - window)
        xt, yt = x[start:t], y[start:t]
        keep = np.isfinite(xt) & np.isfinite(yt)
        if keep.sum() < min_train or not np.isfinite(x[t]):
            continue
        design = np.column_stack([np.ones(keep.sum()), xt[keep]])
        coef, *_ = np.linalg.lstsq(design, yt[keep], rcond=None)
        model[t] = coef[0] + coef[1] * x[t]
        bench[t] = yt[keep].mean()
    ok = np.isfinite(model) & np.isfinite(bench) & np.isfinite(y)
    return y[ok], model[ok], bench[ok]


def sweep(
    frame: pd.DataFrame, burn: int, window: int | None, payoff: str
) -> dict[str, object] | None:
    premium = frame["premium"].to_numpy()
    evalues, ceilings, r2s = [], [], []
    for name in PREDICTORS:
        lagged = frame[name].shift(1).to_numpy()
        outcome, model, bench = rolling_forward(lagged, premium, burn, window)
        if outcome.size < 120:
            continue
        signal = model - bench
        evalues.append(certify(signal, outcome, payoff=payoff, return_bound=ENVELOPE).evalue)
        ceilings.append(
            value_ceiling(signal, outcome, return_bound=ENVELOPE).ratio_ceiling(
                float(outcome.std()), PERIODS_PER_YEAR
            )
        )
        r2s.append(1 - np.sum((outcome - model) ** 2) / np.sum((outcome - bench) ** 2))
    if not evalues:
        return None
    evalues = np.asarray(evalues)
    return {
        "burn_years": burn / PERIODS_PER_YEAR,
        "window": "expanding" if window is None else f"rolling {window // 12}y",
        "payoff": payoff,
        "predictors": len(evalues),
        "median_r2_pct": 100 * float(np.median(r2s)),
        "max_evalue": float(evalues.max()),
        "grid_evalue": merge_average(evalues),
        "n_certified": int((evalues >= 20).sum()),
        "median_ceiling": float(np.median(ceilings)),
    }


def main() -> None:
    frame = load()
    rows = []
    for burn in (120, 240, 360, 480):
        row = sweep(frame, burn, None, "identity")
        if row:
            rows.append(row)
    for window in (240, 360, 600):
        row = sweep(frame, BASELINE_BURN, window, "identity")
        if row:
            rows.append(row)
    for payoff in ("tanh", "sign"):
        row = sweep(frame, BASELINE_BURN, None, payoff)
        if row:
            rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv("audit/goyal_welch_sensitivity.csv", index=False)
    print(
        "Monthly set, 14 classic predictors. Baseline is the first row of each block: "
        "20-year burn-in,\nexpanding window, identity payoff. Rejecting needs an e-value "
        "of 20.\n"
    )
    print(result.to_string(index=False, float_format=lambda v: f"{v:9.3f}"))
    print(
        f"\nAcross all {len(result)} specifications: certified anywhere "
        f"{int(result['n_certified'].sum())}; largest e-value seen "
        f"{result['max_evalue'].max():.2f}; grid e-value ranges "
        f"{result['grid_evalue'].min():.2f} to {result['grid_evalue'].max():.2f}; "
        f"median ceiling ranges {result['median_ceiling'].min():.2f} to "
        f"{result['median_ceiling'].max():.2f}."
    )
    print("\nwrote audit/goyal_welch_sensitivity.csv")


if __name__ == "__main__":
    main()
