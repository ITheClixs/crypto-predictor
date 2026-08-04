"""What the drift does to each instrument, and what the sample size law predicts.

Three experiments, all on data containing no conditional predictability whatsoever.

**Drift sweep.** The "model" is the expanding-window mean -- an estimator with no features
at all, so the correct answer is always "no incremental predictability". Clark-West against
a zero forecast is run alongside Clark-West against that same recursive mean and alongside
the certificate. The first rejects more and more often as the asset's drift grows, at a
rate the closed form predicts; the second is roughly correct; the certificate is bounded by
its nominal level uniformly in the drift.

**Closed form.** With per-period Sharpe ``S`` and estimation window ``k``, the expected
zero-benchmark Clark-West statistic over ``n`` test points is

    E[CW_0]  ~=  sqrt(n) S^2 / sqrt(S^2 + 1/k),

which tends to ``sqrt(n) S`` once the window is long enough to pin the drift down --- that
is, the statistic converges to the *t-statistic of the asset's mean return*. The script
prints predicted against measured.

**Detection time.** Under a real signal of known information ratio, the median time to
certify is compared with ``2[ln(1/alpha) + regret] / IR^2``.

Usage: certificate_calibration.py [reps]
"""

from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd
from scipy import stats

from alphacert import certify, detection_horizon
from cryptoforecast.evaluate.stats import newey_west_lrv

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
SIGMA = 0.03
PERIODS_PER_YEAR = 365.0
N_TEST = 2186  # the study's out-of-sample count at h = 1
WINDOW = 504  # the study's initial training window
CRITICAL = float(stats.norm.ppf(0.95))
#: Drift-grid spacing: three and a half times finer than the outcome's sampling error
#: sigma/sqrt(n) here, and a refinement to 1e-4 moves no result.
GRID = 5e-4


def _clark_west(outcome: np.ndarray, model: np.ndarray, bench: np.ndarray) -> float:
    adjusted = 2.0 * (outcome - bench) * (model - bench)
    lrv = newey_west_lrv(adjusted, 0)
    return float(adjusted.mean() / math.sqrt(lrv / adjusted.size)) if lrv > 0 else float("nan")


#: Number of pure-noise features the larger model estimates. The study uses twelve.
N_FEATURES = 12


def _expanding_mean(pre: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    """The intercept-only forecast: the training mean, updated as the window expands."""
    running = np.concatenate([[0.0], np.cumsum(outcome)[:-1]])
    return (pre.sum() + running) / (pre.size + np.arange(outcome.size))


def _nested_model(benchmark: np.ndarray, rng: np.random.Generator, scale: float) -> np.ndarray:
    """Benchmark plus the out-of-sample contribution of features that carry no signal.

    The larger model estimates ``N_FEATURES`` coefficients that are truly zero on a window of
    ``WINDOW`` observations, so its forecast is the benchmark plus a term of standard
    deviation ``sigma sqrt(p / k)``. Making the features exogenous keeps that term exactly
    uncorrelated with the outcome, which is the case Clark-West is designed for and the case
    in which the recursive-mean benchmark should be correctly sized. The harder case, in
    which the features are lagged returns and therefore share observations with the window,
    is what the full-refit experiment of Table~\ref{tab:size} measures.
    """
    noise = scale * math.sqrt(N_FEATURES / WINDOW) * rng.standard_normal(benchmark.size)
    return benchmark + noise


def drift_sweep(reps: int) -> pd.DataFrame:
    rows = []
    for annual_sharpe in (0.0, 0.4, 0.8, 1.2, 1.6, 2.0):
        per_period = annual_sharpe / math.sqrt(PERIODS_PER_YEAR)
        drift = per_period * SIGMA
        zero_hits = mean_hits = cert_hits = 0
        statistics = np.empty(reps)
        for rep in range(reps):
            rng = np.random.default_rng([101, rep, int(annual_sharpe * 10)])
            outcome = drift + SIGMA * rng.standard_normal(N_TEST)
            pre = drift + SIGMA * rng.standard_normal(WINDOW)
            benchmark = _expanding_mean(pre, outcome)
            model = _nested_model(benchmark, rng, SIGMA)
            statistics[rep] = _clark_west(outcome, model, np.zeros_like(outcome))
            zero_hits += statistics[rep] > CRITICAL
            mean_hits += _clark_west(outcome, model, benchmark) > CRITICAL
            cert_hits += certify(model - benchmark, outcome, drift_resolution=GRID).rejects(0.05)
        predicted = math.sqrt(N_TEST) * per_period**2 / math.sqrt(per_period**2 + 1.0 / WINDOW)
        rows.append(
            {
                "annual_sharpe": annual_sharpe,
                "predicted_cw_zero": predicted,
                "measured_cw_zero": float(statistics.mean()),
                "reject_cw_zero": zero_hits / reps,
                "reject_cw_mean": mean_hits / reps,
                "reject_certificate": cert_hits / reps,
            }
        )
    return pd.DataFrame(rows)


def detection_check(reps: int) -> pd.DataFrame:
    rows = []
    n = 3000
    for ratio in (1.0, 2.0, 3.0, 4.0):
        for designed in (False, True):
            hits = []
            for rep in range(reps // 4):
                rng = np.random.default_rng([202, rep, int(ratio * 10)])
                signal = ratio / math.sqrt(PERIODS_PER_YEAR) * SIGMA * rng.standard_normal(n)
                outcome = 0.8 / math.sqrt(PERIODS_PER_YEAR) * SIGMA + signal
                outcome = outcome + SIGMA * rng.standard_normal(n)
                cert = certify(
                    signal,
                    outcome,
                    design_ratio=ratio if designed else None,
                    drift_resolution=GRID,
                )
                stop = cert.stopping_time(0.05)
                hits.append(stop if stop is not None else np.nan)
            observed = np.array(hits, dtype=float)
            rows.append(
                {
                    "information_ratio": ratio,
                    "stake": "pre-committed" if designed else "learned online",
                    "law_years": detection_horizon(ratio, kelly_known=designed),
                    "median_years": float(np.nanmedian(observed)) / PERIODS_PER_YEAR,
                    "power_at_8_2_years": float(np.mean(~np.isnan(observed))),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    sweep = drift_sweep(REPS)
    sweep.to_csv("audit/certificate_drift_sweep.csv", index=False)
    print("Rejection rate on data with no predictability, as the asset's drift grows.")
    print("The model has no features: it is the recursive mean. Nominal level 5%.\n")
    print(sweep.to_string(index=False, float_format=lambda v: f"{v:9.3f}"))

    detect = detection_check(REPS)
    detect.to_csv("audit/certificate_detection.csv", index=False)
    print("\n\nTime to certify a real signal, against the closed-form law.\n")
    print(detect.to_string(index=False, float_format=lambda v: f"{v:9.2f}"))
    print("\nwrote audit/certificate_drift_sweep.csv, audit/certificate_detection.csv")


if __name__ == "__main__":
    main()
