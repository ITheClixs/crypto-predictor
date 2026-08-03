"""Size of Clark-West AND Wild Clark-West under a genuine nested-estimation null.

Same DGP as ``mc_null.py``: resample BTC daily log returns (iid, or in blocks to retain volatility
clustering) into a synthetic price path, so future returns are independent of every
past-information feature by construction while drift, tails, feature persistence, sample length,
the purged walk-forward geometry and the actual estimators are all retained.

This script adds the Wild Clark-West statistic of Pincheira, Hardy & Munoz (2021), the published
remedy for CW's long-horizon size distortion, so that its size can be measured in this setting
rather than assumed from the authors' DGPs.  It also re-runs the CW cells from the repository root,
which pins the return pool to the committed cache instead of silently re-downloading.

Three nulls are available.

* ``block=1``      iid resampling.  Destroys all dependence, including the volatility
                   clustering that daily return series actually have.  Measures size against
                   a null that is cleaner than reality.
* ``block=21``     block resampling.  Retains volatility clustering, but also retains whatever
                   genuine within-block predictability the real series has, so its rejection
                   rate is an upper bound on size rather than a measurement of it.
* ``--signflip``   block resampling followed by randomising the sign of each deviation from the
                   sample mean.  ``|r_t - mu|`` is untouched, so the volatility dynamics survive
                   intact, while ``E[r_t | past] = mu`` by construction, so no conditional mean
                   predictability remains.  This is the null the other two cannot isolate: it
                   has realistic dependence AND a true no-predictability hypothesis.

``--rho R`` switches the experiment from size to **power**. The sign-flipped deviations are
passed through an AR(1) filter, ``r_t - mu = R (r_{t-1} - mu) + sqrt(1 - R^2) d_t``, which leaves
the unconditional variance unchanged and gives the optimal one-step predictor a population
``R^2`` of exactly ``R^2``. The features include the one-bar return, so the estimators can in
principle find it. Sweeping ``R`` gives the minimum detectable effect.

Usage: mc_null_wcw.py <horizon> <block> <reps> [--gbm] [--signflip] [--seed N] [--asset SYM]
                      [--rho R]

``--asset`` chooses the return pool that the synthetic paths are drawn from.  Size measured on
one asset's return distribution does not automatically transfer to another, and the study's
rejections are not concentrated in the asset the default pool comes from.
"""

from __future__ import annotations

import math
import sys
from functools import partial

import numpy as np
import pandas as pd

from cryptoforecast.backtest.engine import walk_forward
from cryptoforecast.config import DEFAULT_CONFIG
from cryptoforecast.data.loaders import load_ohlcv
from cryptoforecast.dataset import build_supervised
from cryptoforecast.evaluate.stats import diebold_mariano, newey_west_lrv
from cryptoforecast.models.baselines import HistoricalMeanForecaster
from cryptoforecast.models.linear import ElasticNetForecaster, RidgeForecaster
from cryptoforecast.models.trees import GBMForecaster

H = int(sys.argv[1])
BLOCK = int(sys.argv[2])
REPS = int(sys.argv[3])
WITH_GBM = "--gbm" in sys.argv
SIGNFLIP = "--signflip" in sys.argv
PATH_SEED = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 20260730
ASSET = sys.argv[sys.argv.index("--asset") + 1] if "--asset" in sys.argv else "BTC"
RHO = float(sys.argv[sys.argv.index("--rho") + 1]) if "--rho" in sys.argv else 0.0

C_PHI = 0.02  # phi = C_PHI * sd(e_model), the midpoint of the grid in Pincheira et al. (2021)
K = 2

cfg = DEFAULT_CONFIG
real = load_ohlcv(ASSET, cfg.start, cfg.end, cfg.interval)
r = np.diff(np.log(real["Close"].to_numpy()))
n = r.size
print(
    f"asset={ASSET} h={H} block={BLOCK} reps={REPS} gbm={WITH_GBM} signflip={SIGNFLIP} | "
    f"daily log-ret: n={n} "
    f"mean={r.mean():.5f} sd={r.std():.4f} (annualised drift {365 * r.mean():.2f})",
    flush=True,
)


def synth_path(rng: np.random.Generator) -> pd.DataFrame:
    if BLOCK == 1:
        draws = r[rng.integers(0, n, n)]
    else:
        nb = math.ceil(n / BLOCK)
        st = rng.integers(0, n, nb)
        idx = ((st[:, None] + np.arange(BLOCK)[None, :]) % n).ravel()[:n]
        draws = r[idx]
    if SIGNFLIP:
        # |r - mu| is preserved bar for bar, so volatility clustering survives; the sign is
        # independent noise, so E[r_t | past] = mu and nothing conditional is predictable.
        mu = float(draws.mean())
        dev = rng.choice((-1.0, 1.0), size=draws.size) * (draws - mu)
        if RHO:
            # AR(1) in the deviations. The sqrt(1 - rho^2) scaling holds the unconditional
            # variance fixed, so the optimal predictor's population R^2 is exactly rho^2 and
            # the alternative is indexed by an effect size rather than by a nuisance scale.
            scaled = math.sqrt(1.0 - RHO**2) * dev
            out = np.empty_like(scaled)
            prev = 0.0
            for i, e in enumerate(scaled):
                prev = RHO * prev + e
                out[i] = prev
            dev = out
        draws = mu + dev
    close = float(real["Close"].iloc[0]) * np.exp(np.cumsum(np.concatenate([[0.0], draws])))
    px = pd.Series(close, index=real.index)
    return pd.DataFrame(
        {"Open": px, "High": px * 1.005, "Low": px * 0.995, "Close": px,
         "Volume": real["Volume"].to_numpy()},
        index=real.index,
    )


def _t(f: np.ndarray, lags: int) -> float:
    lrv = newey_west_lrv(f, lags)
    if lrv <= 0:
        return float("nan")
    return float(f.mean() / math.sqrt(lrv / f.size))


def cw(y: np.ndarray, m: np.ndarray, b: np.ndarray, lags: int) -> float:
    return _t(2.0 * (y - b) * (m - b), lags)


def wcw(y: np.ndarray, m: np.ndarray, b: np.ndarray, lags: int,
        rng: np.random.Generator) -> float:
    e_b, e_m = y - b, y - m
    phi = C_PHI * float(e_m.std(ddof=1))
    cores = np.empty((K, y.size))
    for j in range(K):
        cores[j] = e_b * (e_b - rng.normal(1.0, phi, size=y.size) * e_m)
    ts = np.array([_t(cores[j], lags) for j in range(K)])
    if not np.all(np.isfinite(ts)):
        return float("nan")
    return float(ts.sum() / math.sqrt(np.corrcoef(cores).sum()))


lag_paper = max(0, H - 1)
models: dict[str, object] = {"ridge": RidgeForecaster, "elastic_net": ElasticNetForecaster}
if WITH_GBM:
    models["gbm"] = partial(GBMForecaster, purge=H)  # same purged early-stopping holdout as the study

path_rng = np.random.default_rng(PATH_SEED)  # default matches mc_null.py: paths are comparable
theta_rng = np.random.default_rng(20260803)  # separate stream so paths are unaffected by theta
rec = []
for rep in range(REPS):
    ds = build_supervised(synth_path(path_rng), H, target="logret")
    oos = {nm: walk_forward(ds, fac, cfg.wf, H) for nm, fac in
           list(models.items()) + [("mean", HistoricalMeanForecaster)]}
    y = oos["ridge"]["y_true"].to_numpy()
    zb = np.zeros_like(y)
    mb = oos["mean"]["y_pred"].to_numpy()
    lag_nw = int(np.floor(4 * (y.size / 100) ** (2 / 9)))
    row: dict[str, float] = {"rep": rep}
    for nm in models:
        m = oos[nm]["y_pred"].to_numpy()
        row[f"{nm}_cw_zero"] = cw(y, m, zb, lag_paper)
        row[f"{nm}_cw_mean"] = cw(y, m, mb, lag_paper)
        row[f"{nm}_cw_mean_nw"] = cw(y, m, mb, lag_nw)
        row[f"{nm}_wcw_zero"] = wcw(y, m, zb, lag_paper, theta_rng)
        row[f"{nm}_wcw_mean"] = wcw(y, m, mb, lag_paper, theta_rng)
        row[f"{nm}_wcw_mean_nw"] = wcw(y, m, mb, lag_nw, theta_rng)
        row[f"{nm}_dm_zero"] = diebold_mariano(oos[nm]["y_true"], m, zb, horizon=H).statistic
    row["mean_cw_zero"] = cw(y, mb, zb, lag_paper)  # drift-only model, zero feature information
    rec.append(row)
    if (rep + 1) % 25 == 0:
        print(f"  {rep + 1}/{REPS}", flush=True)

d = pd.DataFrame(rec)
print()
print(f"{'statistic':24}{'mean':>8}{'sd':>7}{'reject@5%':>11}   (nominal 5%)")
for col in d.columns:
    if col == "rep":
        continue
    v = d[col].dropna().to_numpy()
    two_sided = "_dm_" in col
    rate = float(np.mean(np.abs(v) > 1.96) if two_sided else np.mean(v > 1.645))
    se = 100 * math.sqrt(rate * (1 - rate) / max(v.size, 1))
    print(f"{col:24}{v.mean():>8.2f}{v.std():>7.2f}{100 * rate:>10.1f}%  (n={v.size}, mc se {se:.1f}pp)")

# How far apart are CW and WCW on the same replication?
for nm in models:
    for bench in ("zero", "mean"):
        a, b = d[f"{nm}_cw_{bench}"].to_numpy(), d[f"{nm}_wcw_{bench}"].to_numpy()
        ok = np.isfinite(a) & np.isfinite(b)
        # CW core is 2x the WCW core at theta = 1, and the t-statistic is scale free.
        print(f"|CW - WCW| {nm:12}{bench:6} max={np.max(np.abs(a[ok] - b[ok])):.4f} "
              f"corr={np.corrcoef(a[ok], b[ok])[0, 1]:.6f}")

suffix = (("_gbm" if WITH_GBM else "") + ("_signflip" if SIGNFLIP else "")
          + (f"_rho{RHO:g}".replace("0.", "") if RHO else "")
          + ("" if ASSET == "BTC" else f"_{ASSET}")
          + ("" if PATH_SEED == 20260730 else f"_s{PATH_SEED}"))
d.to_csv(f"audit/mc_null_wcw_h{H}_b{BLOCK}{suffix}.csv", index=False)
print(f"wrote audit/mc_null_wcw_h{H}_b{BLOCK}{suffix}.csv")
