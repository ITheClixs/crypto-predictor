"""Size of DM / Clark-West under a GENUINE nested-estimation null, using the study's own pipeline.

DGP: resample BTC daily log returns (iid, or in blocks to retain volatility clustering) into a
synthetic price path.  Future returns are then independent of every past-information feature by
construction, so the null "features add no predictive value" is TRUE -- while the real drift,
the real fat tails, the real feature collinearity/persistence, the real sample length, the real
purged walk-forward geometry, and the real estimators (fitted intercepts included) are retained.
Unlike the paper's simulation, forecasts here are ESTIMATED, so the estimation-noise term that
Clark-West exists to remove is actually present.
"""
import sys, math, numpy as np, pandas as pd
from scipy import stats
from cryptoforecast.config import DEFAULT_CONFIG
from cryptoforecast.data.loaders import load_ohlcv
from cryptoforecast.dataset import build_supervised
from cryptoforecast.backtest.engine import walk_forward
from cryptoforecast.models.linear import RidgeForecaster, ElasticNetForecaster
from cryptoforecast.models.baselines import HistoricalMeanForecaster
from cryptoforecast.evaluate.stats import newey_west_lrv, diebold_mariano

H     = int(sys.argv[1])
BLOCK = int(sys.argv[2])      # 1 = iid bootstrap; 21 = block bootstrap (keeps vol clustering)
REPS  = int(sys.argv[3])
cfg   = DEFAULT_CONFIG
real  = load_ohlcv("BTC", cfg.start, cfg.end, cfg.interval)
r     = np.diff(np.log(real["Close"].to_numpy()))
n     = r.size
print(f"h={H} block={BLOCK} reps={REPS} | BTC daily log-ret: n={n} mean={r.mean():.5f} sd={r.std():.4f} "
      f"(annualised drift {365*r.mean():.2f})", flush=True)

def synth_path(rng):
    if BLOCK == 1:
        draws = r[rng.integers(0, n, n)]
    else:
        nb = math.ceil(n / BLOCK)
        st = rng.integers(0, n, nb)
        idx = ((st[:, None] + np.arange(BLOCK)[None, :]) % n).ravel()[:n]
        draws = r[idx]
    close = float(real["Close"].iloc[0]) * np.exp(np.cumsum(np.concatenate([[0.0], draws])))
    px = pd.Series(close, index=real.index)
    return pd.DataFrame({"Open": px, "High": px * 1.005, "Low": px * 0.995, "Close": px,
                         "Volume": real["Volume"].to_numpy()}, index=real.index)

def cw(y, m, b, lags):
    f = 2.0 * (y - b) * (m - b)
    lrv = newey_west_lrv(f, lags)
    if lrv <= 0: return np.nan
    return f.mean() / math.sqrt(lrv / f.size)

lag_paper = max(0, H - 1)
rec = []
rng = np.random.default_rng(20260730)
for rep in range(REPS):
    ds = build_supervised(synth_path(rng), H, target="logret")
    oos = {nm: walk_forward(ds, fac, cfg.wf, H) for nm, fac in
           (("ridge", RidgeForecaster), ("elastic_net", ElasticNetForecaster),
            ("mean", HistoricalMeanForecaster))}
    y  = oos["ridge"]["y_true"].to_numpy()
    zb = np.zeros_like(y)
    mb = oos["mean"]["y_pred"].to_numpy()
    nn = y.size
    lag_nw = int(np.floor(4 * (nn / 100) ** (2 / 9)))
    row = {"rep": rep}
    for nm in ("ridge", "elastic_net"):
        m = oos[nm]["y_pred"].to_numpy()
        row[f"{nm}_cw_zero"]    = cw(y, m, zb, lag_paper)
        row[f"{nm}_cw_mean"]    = cw(y, m, mb, lag_paper)
        row[f"{nm}_cw_mean_nw"] = cw(y, m, mb, lag_nw)
        row[f"{nm}_dm_zero"]    = diebold_mariano(oos[nm]["y_true"], m, zb, horizon=H).statistic
    row["mean_cw_zero"] = cw(y, mb, zb, lag_paper)      # drift-only model, zero features
    rec.append(row)
    if (rep + 1) % 50 == 0: print(f"  {rep+1}/{REPS}", flush=True)

d = pd.DataFrame(rec)
print()
print(f"{'statistic':22}{'mean':>8}{'sd':>7}{'reject@5%':>11}   (nominal 5%)")
for col, side in [("ridge_cw_zero","up"),("ridge_cw_mean","up"),("ridge_cw_mean_nw","up"),
                  ("elastic_net_cw_zero","up"),("elastic_net_cw_mean","up"),("elastic_net_cw_mean_nw","up"),
                  ("mean_cw_zero","up"),("ridge_dm_zero","two"),("elastic_net_dm_zero","two")]:
    v = d[col].dropna().to_numpy()
    rate = np.mean(v > 1.645) if side == "up" else np.mean(np.abs(v) > 1.96)
    print(f"{col:22}{v.mean():>8.2f}{v.std():>7.2f}{100*rate:>10.1f}%  (n={v.size}, mc se {100*math.sqrt(rate*(1-rate)/max(v.size,1)):.1f}pp)")
d.to_csv(f"audit/mc_null_h{H}_b{BLOCK}.csv", index=False)
