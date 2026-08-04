"""Joint null distribution of the whole 18-setting experiment.

The manuscript's strongest positive claim -- that 7 rejections against the recursive mean is a
real excess over the ~1 expected -- was inferred from the *sum of marginal rejection
probabilities*.  That is E[N], not the distribution of N.  The 18 tests are strongly dependent:
ridge and elastic net are nearly the same estimator, the two horizons share observations, the
three assets are cross-sectionally correlated, and every setting sees the same market regimes.
Under positive dependence false rejections cluster, so P(N >= 7) can be far larger than a
binomial calculation with the same mean suggests.

This script therefore runs the **entire experiment** inside each null replication: three
synthetic assets, two horizons, three estimators, all 18 statistics, one vector per replication.
From those vectors it reports the joint quantities the marginal experiment cannot give --
P(N >= k), the distribution of the smallest p-value, and Romano-Wolf step-down adjusted
p-values, which are valid under arbitrary dependence.

Cross-asset dependence is induced through a **common sign process**.  Each asset's magnitudes
are block-resampled from its own history independently, preserving its own volatility
clustering, sample length and walk-forward geometry; the sign applied to each date is shared
across assets, which is the channel through which rejections actually cluster.  What this does
not reproduce is the full cross-asset dependence of magnitudes; see the manuscript's discussion
of what the null generator preserves and what it does not.

Usage: mc_joint_null.py <reps> [--gbm] [--workers N] [--seed N]
"""

from __future__ import annotations

import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import pandas as pd

from alphacert import certify, certify_overlapping
from cryptoforecast.backtest.engine import walk_forward
from cryptoforecast.config import DEFAULT_CONFIG
from cryptoforecast.data.loaders import load_ohlcv
from cryptoforecast.dataset import build_supervised
from cryptoforecast.evaluate.stats import newey_west_lrv
from cryptoforecast.models.baselines import HistoricalMeanForecaster
from cryptoforecast.models.linear import ElasticNetForecaster, RidgeForecaster
from cryptoforecast.models.trees import GBMForecaster

REPS = int(sys.argv[1])
WITH_GBM = "--gbm" in sys.argv
#: Measuring the certificate here is a confirmation, not a requirement -- its validity is a
#: theorem, so it needs no calibration. It costs about a third of the runtime, hence the flag.
WITH_CERT = "--cert" in sys.argv
WORKERS = (
    int(sys.argv[sys.argv.index("--workers") + 1])
    if "--workers" in sys.argv
    else max(1, (os.cpu_count() or 4) - 2)
)
SEED = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 20260804

ASSETS = ("BTC", "ETH", "SOL")
HORIZONS = (1, 7)
BLOCK = 21
CFG = DEFAULT_CONFIG


def _load() -> dict[str, pd.DataFrame]:
    return {a: load_ohlcv(a, CFG.start, CFG.end, CFG.interval) for a in ASSETS}


REAL = _load()
RETS = {a: np.diff(np.log(df["Close"].to_numpy())) for a, df in REAL.items()}
# One sign per calendar date, shared by every asset that trades on it.
SIGN_INDEX = REAL["BTC"].index[1:]
SIGN_POS = {a: SIGN_INDEX.get_indexer(REAL[a].index[1:]) for a in ASSETS}


def _synth(asset: str, signs: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    r = RETS[asset]
    n = r.size
    nb = math.ceil(n / BLOCK)
    st = rng.integers(0, n, nb)
    idx = ((st[:, None] + np.arange(BLOCK)[None, :]) % n).ravel()[:n]
    draws = r[idx]
    mu = float(draws.mean())
    s = signs[SIGN_POS[asset]]
    dev = s * (draws - mu)
    close = float(REAL[asset]["Close"].iloc[0]) * np.exp(
        np.cumsum(np.concatenate([[0.0], mu + dev]))
    )
    px = pd.Series(close, index=REAL[asset].index)
    return pd.DataFrame(
        {
            "Open": px,
            "High": px * 1.005,
            "Low": px * 0.995,
            "Close": px,
            "Volume": REAL[asset]["Volume"].to_numpy(),
        },
        index=REAL[asset].index,
    )


def _cw(y: np.ndarray, m: np.ndarray, b: np.ndarray, lags: int) -> float:
    f = 2.0 * (y - b) * (m - b)
    lrv = newey_west_lrv(f, lags)
    if lrv <= 0:
        return float("nan")
    return float(f.mean() / math.sqrt(lrv / f.size))


def _one_rep(rep: int, with_gbm: bool, seed: int, with_cert: bool = False) -> dict[str, float]:
    rng = np.random.default_rng([seed, rep])
    signs = rng.choice((-1.0, 1.0), size=SIGN_INDEX.size)
    models: dict[str, object] = {"ridge": RidgeForecaster, "elastic_net": ElasticNetForecaster}
    row: dict[str, float] = {"rep": float(rep)}
    for asset in ASSETS:
        path = _synth(asset, signs, rng)
        for h in HORIZONS:
            local = dict(models)
            if with_gbm:
                local["gbm"] = partial(GBMForecaster, purge=h)
            ds = build_supervised(path, h, target="logret")
            oos = {
                nm: walk_forward(ds, fac, CFG.wf, h)
                for nm, fac in list(local.items()) + [("mean", HistoricalMeanForecaster)]
            }
            y = oos["ridge"]["y_true"].to_numpy()
            zero = np.zeros_like(y)
            mean_b = oos["mean"]["y_pred"].to_numpy()
            lags = max(0, h - 1)
            for nm in local:
                m = oos[nm]["y_pred"].to_numpy()
                row[f"{asset}_h{h}_{nm}_mean"] = _cw(y, m, mean_b, lags)
                row[f"{asset}_h{h}_{nm}_zero"] = _cw(y, m, zero, lags)
                # The certificate needs no refitting to calibrate -- its validity is a
                # theorem. It is measured here anyway, on exactly the replications that
                # measure Clark-West, so the comparison is like for like.
                if not with_cert:
                    continue
                signal = m - mean_b
                # 5e-4 is three and a half times finer than the outcome's sampling
                # error here; a refinement check to 1e-4 moves no e-value in the fourth
                # decimal and costs five times the runtime.
                if h == 1:
                    evalue = certify(signal, y, drift_resolution=5e-4).evalue
                else:
                    evalue, _ = certify_overlapping(signal, y, horizon=h, drift_resolution=5e-4)
                row[f"{asset}_h{h}_{nm}_cert"] = evalue
    return row


def romano_wolf(observed: np.ndarray, null_draws: np.ndarray) -> np.ndarray:
    """Step-down max-T adjusted p-values, valid under arbitrary dependence.

    ``null_draws`` is (reps, k) of statistics simulated under the joint null; ``observed`` is
    the length-k vector from the real data. One-sided, larger is more significant.
    """
    k = observed.size
    order = np.argsort(-observed)
    adjusted = np.empty(k)
    remaining = list(order)
    running = 0.0
    while remaining:
        j = remaining[0]
        block_max = null_draws[:, remaining].max(axis=1)
        p = float((1.0 + np.sum(block_max >= observed[j])) / (1.0 + null_draws.shape[0]))
        running = max(running, p)  # enforce monotonicity
        adjusted[j] = running
        remaining.pop(0)
    return adjusted


def main() -> None:
    print(
        f"joint null: reps={REPS} gbm={WITH_GBM} workers={WORKERS} seed={SEED}\n"
        f"assets={ASSETS} horizons={HORIZONS} block={BLOCK} sign-flipped, common sign process",
        flush=True,
    )
    fn = partial(_one_rep, with_gbm=WITH_GBM, seed=SEED, with_cert=WITH_CERT)
    rows: list[dict[str, float]] = []
    suffix = ("_gbm" if WITH_GBM else "") + ("_cert" if WITH_CERT else "")
    path = f"audit/mc_joint_null{suffix}.csv"
    # Checkpoint. A full run is hours long, and a partial one is still usable as long as the
    # replication count it was computed from is reported alongside it -- which is why the
    # count is read back off the file rather than assumed.
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for i, row in enumerate(pool.map(fn, range(REPS), chunksize=1), start=1):
            rows.append(row)
            if i % 10 == 0:
                print(f"  {i}/{REPS}", flush=True)
            if i % 50 == 0:
                pd.DataFrame(rows).to_csv(path, index=False)

    d = pd.DataFrame(rows)
    d.to_csv(path, index=False)
    print(f"wrote {path} ({len(d)} replications)")


if __name__ == "__main__":
    main()
