"""Regenerate and persist every OOS forecast the study produces (the repo does not save them)."""
from pathlib import Path

import pandas as pd

from cryptoforecast.backtest.engine import walk_forward
from cryptoforecast.config import DEFAULT_CONFIG
from cryptoforecast.data.loaders import load_ohlcv
from cryptoforecast.dataset import build_supervised
from cryptoforecast.models.registry import default_models

OUT = Path("audit/forecasts.csv")  # run from the repo root so data/cache resolves
cfg = DEFAULT_CONFIG
frames = []
for asset in cfg.assets:
    ohlcv = load_ohlcv(asset, cfg.start, cfg.end, cfg.interval)
    for h in cfg.horizons:
        ds = build_supervised(ohlcv, h, target="logret")
        for name, factory in default_models(h).items():
            oos = walk_forward(ds, factory, cfg.wf, h)
            oos = oos.assign(asset=asset, horizon=h, model=name)
            oos.index.name = "date"
            frames.append(oos.reset_index())
df = pd.concat(frames, ignore_index=True)
df.to_csv(OUT, index=False)
print("wrote", OUT, df.shape)
print(df.groupby(["asset","horizon","model"]).size().head(8))
