"""The full Goyal-Welch-Zafirov predictor set, in real time, under the certificate.

Extends the pilot from the fourteen classic predictors to the complete monthly set, and adds
the two things the pilot could not do: false-discovery control across the whole search, and a
real-time-versus-revised comparison.

**The vintage structure matters and is easy to get wrong.** Goyal, Welch and Zafirov ship each
new predictor as a square matrix whose row ``t`` is the whole series *as it could have been
computed at time t*, because many of these variables need re-estimation every period (partial
least squares factors, output gaps, sentiment indices, technical composites). Two series can be
read out of that matrix:

``real time``
    the diagonal, ``M[t, t]`` -- what a forecaster actually had at ``t``;
``revised``
    the last *column*, ``M[:, T]`` -- the series as it looks today, with every parameter
    estimated on the full sample. (Rows index the period and columns the vintage, so the
    matrix is lower-triangular in the sense that a period is only defined for vintages at or
    after it. The orientation is verified empirically rather than assumed: across the 37
    monthly files the last column carries 20,685 finite values and the last row 1,220.)

Only the first is admissible for an out-of-sample claim. The second is what a study uses if it
downloads "the data" without noticing, and the gap between them is a measurable quantity rather
than a worry, so we report it.

Usage: goyal_welch_full.py [--burn 240]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from goyal_welch_pilot import (
    BURN as PILOT_BURN,
)
from goyal_welch_pilot import (
    ENVELOPE,
    PERIODS_PER_YEAR,
    clark_west,
    load,
    walk_forward,
)
from goyal_welch_pilot import (
    PREDICTORS as CLASSIC,
)

from alphacert import certify, e_bh, growth_to_ratio, merge_average, value_ceiling

BURN = int(sys.argv[sys.argv.index("--burn") + 1]) if "--burn" in sys.argv else PILOT_BURN
GWZ_DIR = Path("data/goyal_welch/gwz2025")
SHOW_COLUMNS = [
    "predictor",
    "source",
    "n",
    "r2_oos_pct",
    "clark_west",
    "evalue",
    "ir_ceiling",
]


def vintage_series(path: Path) -> tuple[pd.Series, pd.Series]:
    """Real-time (diagonal) and revised (last column) series from a GWZ vintage matrix."""
    raw = pd.read_csv(path, header=None)
    periods = raw.iloc[0, 1:].to_numpy()
    body = raw.iloc[1:, 1:].to_numpy(dtype=float)
    vintages = raw.iloc[1:, 0].to_numpy()
    width = min(body.shape[0], body.shape[1])
    index = pd.Index([int(p) for p in periods[:width]], name="yyyymm")
    real_time = pd.Series(np.diagonal(body)[:width], index=index)
    revised = pd.Series(body[:width, -1], index=index)
    assert len(vintages) >= width
    return real_time, revised


def evaluate(name: str, predictor: pd.Series, frame: pd.DataFrame) -> dict[str, object] | None:
    aligned = predictor.reindex(frame["yyyymm"].to_numpy())
    lagged = aligned.shift(1).to_numpy()
    outcome, model, bench = walk_forward(lagged, frame["premium"].to_numpy(), BURN)
    if outcome.size < 240:
        return None
    signal = model - bench
    cert = certify(signal, outcome, return_bound=ENVELOPE)
    ceiling = value_ceiling(signal, outcome, return_bound=ENVELOPE)
    return {
        "predictor": name,
        "n": int(outcome.size),
        "r2_oos_pct": 100 * (1 - np.sum((outcome - model) ** 2) / np.sum((outcome - bench) ** 2)),
        "clark_west": clark_west(outcome, model, bench),
        "evalue": cert.evalue,
        "implied_ir": growth_to_ratio(max(cert.growth_rate(), 0.0), PERIODS_PER_YEAR),
        "ir_ceiling": ceiling.ratio_ceiling(float(outcome.std()), PERIODS_PER_YEAR),
    }


def main() -> None:
    frame = load()
    rows: list[dict[str, object]] = []

    for name in CLASSIC:
        record = evaluate(name, frame.set_index("yyyymm")[name], frame)
        if record:
            record["source"] = "classic"
            record["vintage"] = "as published"
            rows.append(record)

    revised_rows: list[dict[str, object]] = []
    for path in sorted(GWZ_DIR.glob("*_M.csv")):
        name = path.stem[:-2]
        try:
            real_time, revised = vintage_series(path)
        except Exception as exc:  # a malformed matrix should be visible, not silent
            print(f"  skipped {name}: {exc}")
            continue
        record = evaluate(name, real_time, frame)
        if record:
            record["source"] = "GWZ"
            record["vintage"] = "real time"
            rows.append(record)
        other = evaluate(name, revised, frame)
        if other:
            other["source"] = "GWZ"
            other["vintage"] = "revised"
            revised_rows.append(other)

    result = pd.DataFrame(rows).sort_values("evalue", ascending=False)
    result["e_bh_reject"] = e_bh(result["evalue"].to_numpy(), alpha=0.05)
    result.to_csv("audit/goyal_welch_full.csv", index=False)

    span = len(frame) / PERIODS_PER_YEAR
    print(f"Goyal-Welch monthly, {span:.0f} years; {len(result)} predictors evaluated\n")
    print(
        result.head(20).to_string(
            index=False,
            columns=SHOW_COLUMNS,
            float_format=lambda v: f"{v:9.3f}",
        )
    )
    negative = int((result["r2_oos_pct"] < 0).sum())
    print(
        f"\nout-of-sample R^2 negative for {negative} of {len(result)} "
        f"({100 * negative / len(result):.0f}%)"
    )
    grid = merge_average(result["evalue"].to_numpy())
    print(f"grid-level e-value (average): {grid:.3f}  (20 required)")
    print(f"certified at 5%:            {int((result['evalue'] >= 20).sum())} of {len(result)}")
    print(f"surviving e-BH at 5%:       {int(result['e_bh_reject'].sum())} of {len(result)}")
    print(
        f"IR ceilings span {result['ir_ceiling'].min():.2f} to {result['ir_ceiling'].max():.2f}; "
        f"median {result['ir_ceiling'].median():.2f}"
    )

    if revised_rows:
        revised = pd.DataFrame(revised_rows).set_index("predictor")
        real = result[result["vintage"] == "real time"].set_index("predictor")
        both = real.join(revised, rsuffix="_rev", how="inner")
        both["evalue_gap"] = both["evalue_rev"] - both["evalue"]
        both["r2_gap"] = both["r2_oos_pct_rev"] - both["r2_oos_pct"]
        both.to_csv("audit/goyal_welch_vintage.csv")
        print(
            f"\nREAL TIME vs REVISED, {len(both)} GWZ predictors:\n"
            f"  mean out-of-sample R^2, real time {both['r2_oos_pct'].mean():+.3f}%, "
            f"revised {both['r2_oos_pct_rev'].mean():+.3f}%\n"
            f"  revised looks better for {int((both['r2_gap'] > 0).sum())} of {len(both)}; "
            f"mean e-value {both['evalue'].mean():.2f} -> {both['evalue_rev'].mean():.2f}"
        )
    print("\nwrote audit/goyal_welch_full.csv, audit/goyal_welch_vintage.csv")


if __name__ == "__main__":
    main()
