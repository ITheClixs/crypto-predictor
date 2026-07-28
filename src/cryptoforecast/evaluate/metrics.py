"""Point-forecast accuracy metrics for return targets.

MAPE is deliberately absent: the target is a return centered near zero, so
percentage error explodes and is meaningless. The honest metrics for this problem
are error magnitude (RMSE/MAE), out-of-sample R^2 (which can and often should go
negative, meaning worse than predicting the mean), and, because a trader only needs
the sign, directional accuracy and rank information coefficient.
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


def r2_oos(y_true: pd.Series, y_pred: np.ndarray, y_bench: np.ndarray | None = None) -> float:
    """Out-of-sample R^2; negative means the forecast is worse than the benchmark.

    With ``y_bench`` this is the Campbell-Thompson (2008) statistic used throughout
    the return-predictability literature: the denominator is the squared error of a
    genuine *ex ante* benchmark forecast, here the recursively-estimated historical
    mean. That is the number worth quoting.

    With ``y_bench=None`` the denominator falls back to the variance around the
    realized mean of the evaluation window. That mean is not knowable in advance,
    so the fallback flatters the benchmark and is reported only as a
    cross-check, never as "the" out-of-sample R^2.
    """
    if y_bench is None:
        yt, yp = _clean(y_true, y_pred)
        reference = np.full_like(yt, yt.mean())
    else:
        # One joint mask, so model and benchmark are scored on identical rows.
        yt = np.asarray(y_true, dtype=float)
        yp = np.asarray(y_pred, dtype=float)
        yb = np.asarray(y_bench, dtype=float)
        keep = np.isfinite(yt) & np.isfinite(yp) & np.isfinite(yb)
        yt, yp, reference = yt[keep], yp[keep], yb[keep]
    sse = float(np.sum((yt - yp) ** 2))
    sst = float(np.sum((yt - reference) ** 2))
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


def regression_metrics(
    y_true: pd.Series, y_pred: np.ndarray, y_bench: np.ndarray | None = None
) -> dict[str, float]:
    """Standard forecast metrics; ``y_bench`` supplies the R^2 reference forecast."""
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2_oos": r2_oos(y_true, y_pred, y_bench),
        "r2_vs_sample_mean": r2_oos(y_true, y_pred),
        "dir_acc": directional_accuracy(y_true, y_pred),
        "rank_ic": rank_ic(y_true, y_pred),
    }
