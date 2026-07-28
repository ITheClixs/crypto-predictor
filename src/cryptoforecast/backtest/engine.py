"""The walk-forward backtest loop.

Fits a fresh model on each training block and predicts the immediately following
test block, concatenating the out-of-sample predictions into one series that
covers the whole evaluation period. No test row ever informs a fit, and the
splits are purged/embargoed, so the resulting predictions are genuinely OOS.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import WalkForwardConfig
from ..dataset import Dataset
from ..models.base import ModelFactory
from ..splits import walk_forward_splits

OOS_COLUMNS: tuple[str, ...] = ("y_true", "y_pred", "close", "fold")


def walk_forward(
    ds: Dataset,
    make_model: ModelFactory,
    wf: WalkForwardConfig,
    horizon: int,
) -> pd.DataFrame:
    """Run one model through walk-forward CV.

    Returns a DataFrame indexed by date with columns :data:`OOS_COLUMNS`
    (realized target, predicted target, price at decision time, fold id).
    """
    data = ds.labeled  # only rows with an observable target participate
    X, y, close = data.X, data.y, data.close
    splits = walk_forward_splits(
        len(X),
        train_size=wf.train_size,
        test_size=wf.test_size,
        horizon=horizon,
        embargo=wf.embargo,
        mode=wf.mode,
        min_train=wf.min_train,
    )
    if not splits:
        raise ValueError(
            f"not enough data for a walk-forward fold: {len(X)} labeled rows, "
            f"need > {wf.min_train + horizon + wf.embargo}"
        )

    frames: list[pd.DataFrame] = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        model = make_model()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred = np.asarray(model.predict(X.iloc[test_idx]), dtype=float)
        frames.append(
            pd.DataFrame(
                {
                    "y_true": y.iloc[test_idx].to_numpy(dtype=float),
                    "y_pred": y_pred,
                    "close": close.iloc[test_idx].to_numpy(dtype=float),
                    "fold": fold,
                },
                index=X.index[test_idx],
            )
        )

    out = pd.concat(frames).sort_index()
    return out[~out.index.duplicated(keep="first")]
