"""Figures 2 and 3: forecast accuracy and sign-timing across every setting."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import MODEL_ORDER, NEGATIVE, finish, label


def _grid(table: pd.DataFrame, value: str) -> tuple[np.ndarray, list[str], list[str]]:
    """Reshape a tidy column into a models x (asset, horizon) matrix."""
    sub = table[table["model"].isin(MODEL_ORDER)].copy()
    sub["setting"] = sub["asset"] + "\nh=" + sub["horizon"].astype(str) + "d"
    wide = sub.pivot_table(index="model", columns="setting", values=value, sort=False)
    rows = [m for m in MODEL_ORDER if m in wide.index]
    wide = wide.loc[rows]
    return wide.to_numpy(dtype=float), [label(m) for m in rows], list(wide.columns)


def plot_r2_heatmap(table: pd.DataFrame, out_path: Path) -> None:
    """Out-of-sample R^2 against the drift benchmark, one cell per setting."""
    values, rows, cols = _grid(table, "r2_oos")
    limit = float(np.nanmax(np.abs(values))) or 1.0

    fig, ax = plt.subplots(figsize=(1.05 * len(cols) + 2.6, 0.52 * len(rows) + 2.2))
    mesh = ax.imshow(values, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.grid(False)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if np.isnan(v):
                continue
            ax.text(
                j,
                i,
                f"{v:+.4f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if abs(v) > 0.55 * limit else "#222222",
            )

    fig.colorbar(mesh, ax=ax, shrink=0.82, label=r"$R^2_{OS}$ vs historical mean")
    ax.set_title("Out-of-sample $R^2$ against the drift benchmark")
    finish(
        fig,
        out_path,
        "Red is worse than an ex-ante historical-mean forecast, blue is better. That "
        "benchmark scores exactly 0 against itself, which fixes the scale.",
    )


def _binomial_band(n: int, alpha: float = 0.05) -> float:
    """Half-width of the two-sided normal band for a fair coin over n draws."""
    return 1.959963985 * math.sqrt(0.25 / max(n, 1))


def plot_directional_accuracy(table: pd.DataFrame, out_path: Path) -> None:
    """Hit rate per setting against the band a coin flip would occupy.

    Two corrections matter here. The band is drawn per column, since the number
    of out-of-sample forecasts differs by asset and horizon. And it uses the
    *effective* sample size ``n / h``: consecutive h-day forecasts are formed
    from overlapping windows, so counting each as an independent draw would make
    the band far too tight and manufacture significance.
    """
    values, rows, cols = _grid(table, "dir_acc")
    counts, _, _ = _grid(table, "n_oos")
    horizons, _, _ = _grid(table, "horizon")
    n_per_col = np.nanmax(counts, axis=0)
    h_per_col = np.nanmax(horizons, axis=0)
    n_eff = n_per_col / np.maximum(h_per_col, 1.0)

    fig, ax = plt.subplots(figsize=(1.35 * len(cols) + 2.6, 4.0))
    width = 0.8 / max(len(rows), 1)
    x = np.arange(len(cols))
    bands = np.array([_binomial_band(int(n)) for n in n_eff])

    for j, band in enumerate(bands):
        ax.add_patch(
            plt.Rectangle(
                (j - 0.46, 0.5 - band),
                0.92,
                2 * band,
                color=NEGATIVE,
                alpha=0.16,
                lw=0,
                zorder=0,
            )
        )

    for i, row in enumerate(rows):
        ax.bar(
            x + i * width - 0.4 + width / 2,
            values[i] - 0.5,
            width=width,
            bottom=0.5,
            label=row,
            zorder=2,
        )

    ax.axhline(0.5, color="#222222", lw=1.0, zorder=1)
    ax.set_xticks(x, cols)
    ax.set_ylabel("directional accuracy")
    ax.set_title("Directional accuracy against a coin flip", pad=26)
    ax.legend(ncol=min(len(rows), 5), loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7.5)
    fig.tight_layout()

    outside = int(np.nansum(np.abs(values - 0.5) > bands[None, :]))
    verdict = (
        "Every bar lands inside its band."
        if outside == 0
        else f"{outside} of {int(np.isfinite(values).sum())} bars fall outside."
    )
    finish(
        fig,
        out_path,
        "Shaded band is the 95% interval a fair coin would occupy, using the effective "
        f"sample size n/h ({int(np.nanmin(n_eff))}-{int(np.nanmax(n_eff))} independent "
        f"forecasts) rather than the raw count. {verdict}",
    )
