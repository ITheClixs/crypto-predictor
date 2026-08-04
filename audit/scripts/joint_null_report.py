"""Read the joint-null replications and report what the marginal experiment cannot.

The manuscript's Clark-West rejection count is compared here against the distribution of the
count under the joint null, not against its expectation. Also reported: the max-T global-null
p-value, Monte Carlo p-values per setting, and Romano-Wolf step-down adjusted p-values, which
are valid under arbitrary dependence.

Usage: joint_null_report.py [null-csv]
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import stats

NULL = sys.argv[1] if len(sys.argv) > 1 else "audit/mc_joint_null_gbm.csv"
CRITICAL = float(stats.norm.ppf(0.95))


def romano_wolf(observed: np.ndarray, draws: np.ndarray) -> np.ndarray:
    """Step-down max-T adjusted p-values; one-sided, larger statistic is more significant."""
    order = np.argsort(-observed)
    adjusted = np.empty(observed.size)
    remaining = list(order)
    running = 0.0
    while remaining:
        j = remaining[0]
        block_max = draws[:, remaining].max(axis=1)
        p = float((1.0 + np.sum(block_max >= observed[j])) / (1.0 + draws.shape[0]))
        running = max(running, p)
        adjusted[j] = running
        remaining.pop(0)
    return adjusted


def main() -> None:
    null = pd.read_csv(NULL).drop(columns=["rep"])
    observed = pd.read_csv("audit/retest_cw.csv")
    columns = [f"{r.asset}_h{r.h}_{r.model}_mean" for r in observed.itertuples()]
    missing = set(columns) - set(null.columns)
    if missing:
        raise SystemExit(f"null file is missing {sorted(missing)}")
    draws = null[columns].to_numpy()
    statistic = observed["cw_mean"].to_numpy()

    counts = (draws > CRITICAL).sum(axis=1)
    observed_count = int((statistic > CRITICAL).sum())
    reps = draws.shape[0]
    print(f"replications = {reps};  observed rejections at nominal 5% = {observed_count}")
    print(
        f"E[N] under the joint null = {counts.mean():.3f}, sd = {counts.std():.3f} "
        f"(binomial sd if independent = {np.sqrt(18 * 0.05 * 0.95):.3f})"
    )
    for k in range(4, 10):
        print(f"  P(N >= {k}) = {(counts >= k).mean():.4f}")
    peak = draws.max(axis=1)
    print(
        f"max-T: observed max = {statistic.max():.3f}, null 95th percentile = "
        f"{np.quantile(peak, 0.95):.3f}, global-null p = "
        f"{(1 + (peak >= statistic.max()).sum()) / (1 + reps):.4f}"
    )
    print(
        f"marginal cell size: min {(draws > CRITICAL).mean(axis=0).min():.3f}, "
        f"median {np.median((draws > CRITICAL).mean(axis=0)):.3f}, "
        f"max {(draws > CRITICAL).mean(axis=0).max():.3f}"
    )

    table = observed[["asset", "h", "model", "cw_mean", "p_mean", "holm_mean", "bh_mean"]].copy()
    table["p_monte_carlo"] = [
        (1 + (draws[:, i] >= statistic[i]).sum()) / (1 + reps) for i in range(statistic.size)
    ]
    table["p_romano_wolf"] = romano_wolf(statistic, draws)
    table = table.sort_values("cw_mean", ascending=False)
    print("\n" + table.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
    table.to_csv("audit/joint_null_results.csv", index=False)
    print("\nwrote audit/joint_null_results.csv")


if __name__ == "__main__":
    main()
