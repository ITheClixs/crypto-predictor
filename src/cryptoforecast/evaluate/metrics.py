"""Point-forecast accuracy metrics for return targets.

MAPE is deliberately absent: the target is a return centered near zero, so
percentage error explodes and is meaningless. The honest metrics for this problem
are error magnitude (RMSE/MAE), out-of-sample R^2 (which can and often should go
negative — worse than predicting the mean), and — because a trader only needs the
sign — directional accuracy and rank information coefficient.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _clean(y_true: pd.Series, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    return yt[mask], yp[mask]


def mae(y_true: pd.Series, y_pred: np.ndarray) -> float:
    yt, yp = _clean(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    yt, yp = _clean(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def r2_oos(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """Out-of-sample R^2 against the in-sample mean; negative = worse than the mean."""
    yt, yp = _clean(y_true, y_pred)
    sse = float(np.sum((yt - yp) ** 2))
    sst = float(np.sum((yt - yt.mean()) ** 2))
    return 1.0 - sse / sst if sst > 0 else float("nan")


def directional_accuracy(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """Fraction of non-zero forecasts whose sign matches the realized sign."""
    yt, yp = _clean(y_true, y_pred)
    betting = yp != 0
    if not betting.any():
        return float("nan")  # a model that never takes a side (e.g. random walk)
    return float(np.mean(np.sign(yt[betting]) == np.sign(yp[betting])))


def rank_ic(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """Spearman rank correlation (information coefficient) between forecast and outcome."""
    yt, yp = _clean(y_true, y_pred)
    if np.unique(yp).size < 2 or np.unique(yt).size < 2:
        return float("nan")
    return float(spearmanr(yp, yt).statistic)


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2_oos": r2_oos(y_true, y_pred),
        "dir_acc": directional_accuracy(y_true, y_pred),
        "rank_ic": rank_ic(y_true, y_pred),
    }
