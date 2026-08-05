"""A fourth null generator, and an audit of what each one preserves.

The manuscript's calibration rests on resampling nulls, and a resampling null is only as
good as the properties it keeps. Block resampling keeps whatever within-block structure the
real series has, including any genuine predictability, so its rejection rate bounds size from
above rather than measuring it. Sign flipping removes conditional mean predictability exactly
while keeping the volatility path bar for bar -- but independent sign randomisation also
destroys return-sign autocorrelation, leverage effects, skewness dynamics, and any
relationship between the direction of a return and its volume or range. Attributing the gap
between the two to "genuine predictability" therefore over-claims: many properties differ.

This script adds a parametric null that is not a resampling at all -- a GARCH(1,1) with a
zero conditional mean, fitted to each asset's returns and simulated forward -- and then
audits all four generators on the properties that plausibly matter for a nested test:
marginal moments, volatility clustering, the leverage effect, and return autocorrelation.
Agreement of the measured rejection rate across constructions that preserve *different*
things is much stronger evidence than any one of them alone.

Usage: garch_null.py [reps]
"""

from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd
from scipy import stats

from alphacert import certify
from cryptoforecast.config import DEFAULT_CONFIG
from cryptoforecast.data.loaders import load_ohlcv
from cryptoforecast.evaluate.stats import newey_west_lrv

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
ASSET = "BTC"
BLOCK = 21
WINDOW = 504
N_FEATURES = 12
CRITICAL = float(stats.norm.ppf(0.95))
CFG = DEFAULT_CONFIG


def fit_garch(returns: np.ndarray) -> tuple[float, float, float, float]:
    """Gaussian quasi-MLE for a zero-mean GARCH(1,1), by grid-refined direct search.

    Fitting the conditional mean is deliberately omitted: the drift is carried separately so
    that the simulated series has a realistic drift and no conditional predictability, which
    is the null the paper needs.
    """
    mu = float(returns.mean())
    dev = returns - mu
    var = float(dev.var())

    def negative_log_likelihood(theta: np.ndarray) -> float:
        omega, alpha, beta = theta
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
            return 1e12
        h = np.empty(dev.size)
        h[0] = var
        for t in range(1, dev.size):
            h[t] = omega + alpha * dev[t - 1] ** 2 + beta * h[t - 1]
        return float(0.5 * np.sum(np.log(h) + dev**2 / h))

    best = (var * 0.05, 0.05, 0.90)
    best_value = negative_log_likelihood(np.array(best))
    for alpha in np.linspace(0.02, 0.25, 12):
        for beta in np.linspace(0.60, 0.97, 12):
            if alpha + beta >= 0.999:
                continue
            omega = var * (1.0 - alpha - beta)
            value = negative_log_likelihood(np.array([omega, alpha, beta]))
            if value < best_value:
                best_value, best = value, (omega, alpha, beta)
    return (mu, *best)


def simulate_garch(
    params: tuple[float, float, float, float], n: int, rng: np.random.Generator
) -> np.ndarray:
    mu, omega, alpha, beta = params
    h = omega / max(1.0 - alpha - beta, 1e-6)
    out = np.empty(n)
    for t in range(n):
        shock = rng.standard_normal()
        out[t] = mu + math.sqrt(h) * shock
        h = omega + alpha * (out[t] - mu) ** 2 + beta * h
    return out


def block_resample(returns: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = returns.size
    starts = rng.integers(0, n, math.ceil(n / BLOCK))
    idx = ((starts[:, None] + np.arange(BLOCK)[None, :]) % n).ravel()[:n]
    return returns[idx]


def sign_flip(returns: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    drawn = block_resample(returns, rng)
    mu = float(drawn.mean())
    return mu + rng.choice((-1.0, 1.0), drawn.size) * (drawn - mu)


def properties(returns: np.ndarray) -> dict[str, float]:
    """The properties a nested test could plausibly be sensitive to."""
    dev = returns - returns.mean()
    abs_dev = np.abs(dev)
    return {
        "sd": float(returns.std()),
        "excess_kurtosis": float(stats.kurtosis(returns)),
        "skew": float(stats.skew(returns)),
        "acf1_return": float(np.corrcoef(dev[:-1], dev[1:])[0, 1]),
        "acf1_abs": float(np.corrcoef(abs_dev[:-1], abs_dev[1:])[0, 1]),
        "leverage": float(np.corrcoef(dev[:-1], abs_dev[1:])[0, 1]),
        "sign_acf1": float(np.corrcoef(np.sign(dev[:-1]), np.sign(dev[1:]))[0, 1]),
    }


def _expanding_mean(pre: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    running = np.concatenate([[0.0], np.cumsum(outcome)[:-1]])
    return (pre.sum() + running) / (pre.size + np.arange(outcome.size))


def _clark_west(outcome: np.ndarray, model: np.ndarray, bench: np.ndarray) -> float:
    adjusted = 2.0 * (outcome - bench) * (model - bench)
    lrv = newey_west_lrv(adjusted, 0)
    return float(adjusted.mean() / math.sqrt(lrv / adjusted.size)) if lrv > 0 else float("nan")


def main() -> None:
    frame = load_ohlcv(ASSET, CFG.start, CFG.end, CFG.interval)
    returns = np.diff(np.log(frame["Close"].to_numpy()))
    params = fit_garch(returns)
    print(
        f"GARCH(1,1) fitted to {ASSET}: mu={params[0]:.5f} omega={params[1]:.3e} "
        f"alpha={params[2]:.3f} beta={params[3]:.3f} (persistence {params[2] + params[3]:.3f})"
    )

    generators = {
        "real series": lambda rng: returns,
        "iid resample": lambda rng: rng.choice(returns, returns.size, replace=True),
        "block resample": lambda rng: block_resample(returns, rng),
        "sign-flipped": lambda rng: sign_flip(returns, rng),
        "GARCH(1,1), zero mean": lambda rng: simulate_garch(params, returns.size, rng),
    }

    audit_rows = []
    for name, generator in generators.items():
        draws = [properties(generator(np.random.default_rng([3, r]))) for r in range(60)]
        row = {"generator": name}
        row.update({k: float(np.mean([d[k] for d in draws])) for k in draws[0]})
        audit_rows.append(row)
    audit = pd.DataFrame(audit_rows)
    audit.to_csv("audit/null_properties.csv", index=False)
    print("\nWhat each generator preserves (means over 60 draws):\n")
    print(audit.to_string(index=False, float_format=lambda v: f"{v:9.4f}"))

    print(f"\nRejection rate at a nominal 5%, {REPS} replications, drift-carrying nulls:\n")
    rates = []
    for name in ("block resample", "sign-flipped", "GARCH(1,1), zero mean"):
        generator = generators[name]
        zero_hits = mean_hits = cert_hits = 0
        for rep in range(REPS):
            rng = np.random.default_rng([909, rep])
            path = generator(rng)
            outcome = path[WINDOW:]
            benchmark = _expanding_mean(path[:WINDOW], outcome)
            noise = (
                float(outcome.std())
                * math.sqrt(N_FEATURES / WINDOW)
                * rng.standard_normal(outcome.size)
            )
            model = benchmark + noise
            zero_hits += _clark_west(outcome, model, np.zeros_like(outcome)) > CRITICAL
            mean_hits += _clark_west(outcome, model, benchmark) > CRITICAL
            cert_hits += certify(noise, outcome, drift_resolution=5e-4).rejects(0.05)
        error = 100 * math.sqrt(0.05 * 0.95 / REPS)
        rates.append(
            {
                "null": name,
                "cw_vs_zero_pct": 100 * zero_hits / REPS,
                "cw_vs_mean_pct": 100 * mean_hits / REPS,
                "certificate_pct": 100 * cert_hits / REPS,
                "mc_se_pct": error,
            }
        )
    result = pd.DataFrame(rates)
    result.to_csv("audit/null_agreement.csv", index=False)
    print(result.to_string(index=False, float_format=lambda v: f"{v:8.2f}"))
    print("\nwrote audit/null_properties.csv, audit/null_agreement.csv")


if __name__ == "__main__":
    main()
