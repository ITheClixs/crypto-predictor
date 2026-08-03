"""Wild Clark-West (Pincheira, Hardy & Munoz 2021) applied to the study's saved forecasts.

The paper's own diagnosis -- that Clark-West against an analytic long-run variance is not
trustworthy at h=7 and that a dependence-robust alternative is needed -- was already answered in
the literature.  WCW replaces the degenerate CW core statistic

    f_t = e_b,t (e_b,t - e_m,t)            (= CW / 2)

with

    f_t = e_b,t (e_b,t - theta_t e_m,t),   theta_t ~ iid N(1, phi^2), independent of everything,

which keeps the statistic centred at zero under the null (E[e^2 (1 - theta)] = 0) while giving it
strictly positive variance, so West's (1996) asymptotics apply and the statistic is asymptotically
normal.  phi is set to a small fraction of sd(e_m) as the authors prescribe.

Because the statistic depends on one draw of theta, we report both the authors' K=2 smoothed
statistic and the distribution of that statistic over many independent draws.

One reading decision: the smoothed statistic divides the sum of K realisations by
sqrt(sum_i sum_j rho_ij), where the authors describe rho_ij as "the sample correlation between
the i-th and j-th realization of the WCW-t statistics".  A single sample yields one t-statistic
per realisation, not a sample of them, so rho_ij is computed here as the correlation between the
i-th and j-th core sequences, which is the only in-sample quantity that identifies it.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from cryptoforecast.evaluate.stats import (
    benjamini_hochberg_adjusted,
    holm_adjusted,
    newey_west_lrv,
)

ML = ("ridge", "elastic_net", "gbm")
C_GRID = (0.01, 0.02, 0.04)  # phi as a fraction of sd(e_model), per Pincheira et al. (2021)
C_MAIN = 0.02
K = 2  # number of theta draws averaged, per the authors' smoothed statistic
DRAWS = 500  # independent replications of the K-draw statistic, to expose theta randomness
SEED = 20260803


def _t_stat(f: np.ndarray, lags: int) -> float:
    lrv = newey_west_lrv(f, lags)
    if lrv <= 0:
        return float("nan")
    return float(f.mean() / math.sqrt(lrv / f.size))


def clark_west(y: np.ndarray, m: np.ndarray, b: np.ndarray, lags: int) -> float:
    """Standard CW-t.  2 (y - b)(m - b) is algebraically e_b (e_b - e_m) up to the factor 2."""
    return _t_stat(2.0 * (y - b) * (m - b), lags)


def wild_clark_west(
    y: np.ndarray,
    m: np.ndarray,
    b: np.ndarray,
    lags: int,
    c: float,
    rng: np.random.Generator,
    k: int = K,
) -> float:
    """Smoothed WCW-t over ``k`` independent theta sequences."""
    e_b = y - b
    e_m = y - m
    phi = c * float(e_m.std(ddof=1))
    cores = np.empty((k, y.size))
    for j in range(k):
        theta = rng.normal(1.0, phi, size=y.size)
        cores[j] = e_b * (e_b - theta * e_m)
    stats_k = np.array([_t_stat(cores[j], lags) for j in range(k)])
    if not np.all(np.isfinite(stats_k)):
        return float("nan")
    rho = np.corrcoef(cores) if k > 1 else np.ones((1, 1))
    return float(stats_k.sum() / math.sqrt(rho.sum()))


def _p_upper(z: float) -> float:
    return float(1.0 - stats.norm.cdf(z))


def main() -> None:
    df = pd.read_csv("audit/forecasts.csv", parse_dates=["date"])
    rng = np.random.default_rng(SEED)
    rows = []

    for (asset, h), group in df.groupby(["asset", "horizon"]):
        piv = group.pivot_table(index="date", columns="model", values=["y_true", "y_pred"])
        y = piv[("y_true", "ridge")].to_numpy()
        benches = {
            "zero": np.zeros_like(y),
            "recursive_mean": piv[("y_pred", "historical_mean")].to_numpy(),
        }
        lags = max(0, h - 1)
        for model in ML:
            m = piv[("y_pred", model)].to_numpy()
            for bench_name, b in benches.items():
                row = {
                    "asset": asset,
                    "h": h,
                    "model": model,
                    "benchmark": bench_name,
                    "n": y.size,
                    "cw": clark_west(y, m, b, lags),
                }
                row["p_cw"] = _p_upper(row["cw"])
                for c in C_GRID:
                    draws = np.array(
                        [wild_clark_west(y, m, b, lags, c, rng) for _ in range(DRAWS)]
                    )
                    tag = f"{c:g}".replace("0.", "")
                    row[f"wcw_{tag}"] = float(np.median(draws))
                    row[f"p_wcw_{tag}"] = _p_upper(float(np.median(draws)))
                    row[f"rejfrac_{tag}"] = float(np.mean(draws > 1.645))
                rows.append(row)

    res = pd.DataFrame(rows)
    main_tag = f"{C_MAIN:g}".replace("0.", "")
    for bench in ("zero", "recursive_mean"):
        sub = res.benchmark == bench
        res.loc[sub, "holm_cw"] = holm_adjusted(res.loc[sub, "p_cw"].to_numpy())
        res.loc[sub, "bh_cw"] = benjamini_hochberg_adjusted(res.loc[sub, "p_cw"].to_numpy())
        res.loc[sub, "holm_wcw"] = holm_adjusted(res.loc[sub, f"p_wcw_{main_tag}"].to_numpy())
        res.loc[sub, "bh_wcw"] = benjamini_hochberg_adjusted(
            res.loc[sub, f"p_wcw_{main_tag}"].to_numpy()
        )

    pd.set_option("display.width", 220, "display.max_columns", 40)
    cols = ["asset", "h", "model", "cw", "p_cw", f"wcw_{main_tag}", f"p_wcw_{main_tag}",
            f"rejfrac_{main_tag}", "wcw_01", "wcw_04"]
    for bench in ("zero", "recursive_mean"):
        sub = res[res.benchmark == bench]
        print("=" * 120)
        print(f"benchmark = {bench};  WCW at phi = {C_MAIN} sd(e_model), K={K}, "
              f"median over {DRAWS} theta draws")
        print("=" * 120)
        print(sub[cols].round(4).to_string(index=False))
        print(
            f"  reject at raw 5%:  CW {int((sub.p_cw < 0.05).sum())}/18   "
            f"WCW {int((sub[f'p_wcw_{main_tag}'] < 0.05).sum())}/18   "
            f"(phi=0.01 {int((sub['p_wcw_01'] < 0.05).sum())}/18, "
            f"phi=0.04 {int((sub['p_wcw_04'] < 0.05).sum())}/18)"
        )
        print(
            f"  survive BH(5%):    CW {int((sub.bh_cw < 0.05).sum())}   "
            f"WCW {int((sub.bh_wcw < 0.05).sum())}    |    "
            f"Holm(5%): CW {int((sub.holm_cw < 0.05).sum())}   WCW {int((sub.holm_wcw < 0.05).sum())}"
        )
        print()

    res.to_csv("audit/wcw.csv", index=False)
    print("wrote audit/wcw.csv")


if __name__ == "__main__":
    main()
